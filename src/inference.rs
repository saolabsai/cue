//! Native CPU inference for Cue's acoustic ONNX backends.

use crate::features::{log_mel, MEL_BINS, MEL_FRAMES};
use anyhow::{anyhow, ensure, Context, Result};
use ndarray::{arr0, Array2, Array3};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::{tensor::TensorElementType, value::ValueType};
use std::{
    path::{Path, PathBuf},
    sync::OnceLock,
};

/// Silero's supported frame length at 16 kHz (32 ms).
pub const VAD_FRAME_SAMPLES: usize = 512;
const VAD_CONTEXT_SAMPLES: usize = 64;
const MAX_JOINT_STATE_SIZE: i64 = 256;

fn initialize_runtime(runtime_path: Option<&Path>) -> Result<()> {
    let requested = runtime_path
        .map(Path::to_path_buf)
        .or_else(|| std::env::var_os("ORT_DYLIB_PATH").map(PathBuf::from));
    let requested = requested
        .map(|path| {
            ensure!(
                path.is_file(),
                "ONNX Runtime library does not exist: {}",
                path.display()
            );
            path.canonicalize()
                .context("resolve ONNX Runtime library path")
        })
        .transpose()?;
    static INITIALIZED: OnceLock<std::result::Result<Option<PathBuf>, String>> = OnceLock::new();
    let initialized = INITIALIZED.get_or_init(|| {
        // ort rc.9 panics if the native loader cannot open a library. Turn that
        // dependency failure into a caller-visible Result instead of unwinding.
        let result = std::panic::catch_unwind(|| {
            let builder = match &requested {
                Some(path) => ort::init_from(path.to_string_lossy()),
                None => ort::init(),
            };
            builder.with_name("sao-turn").commit()
        });
        match result {
            Ok(Ok(_)) => Ok(requested.clone()),
            Ok(Err(error)) => Err(format!("initialize ONNX Runtime: {error}")),
            Err(_) => Err("could not load ONNX Runtime; supply --runtime or ORT_DYLIB_PATH pointing to a compatible native library".into()),
        }
    });
    match initialized {
        Ok(loaded) => {
            if requested.is_some() {
                ensure!(loaded == &requested,
                    "ONNX Runtime is already initialized with a different library; use one runtime per process");
            }
            Ok(())
        }
        Err(message) => Err(anyhow!(message.clone())),
    }
}

pub(crate) fn session(
    model_path: &Path,
    runtime_path: Option<&Path>,
    threads: usize,
) -> Result<Session> {
    ensure!(
        threads > 0 && threads <= i32::MAX as usize,
        "inference threads must be in 1..=2147483647"
    );
    ensure!(
        model_path.is_file(),
        "model not found: {}; run python scripts/prepare_models.py",
        model_path.display()
    );
    initialize_runtime(runtime_path)?;
    Session::builder()?
        .with_parallel_execution(false)?
        .with_inter_threads(1)?
        .with_intra_threads(threads)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(model_path)
        .with_context(|| format!("load ONNX model {}", model_path.display()))
}

/// Cue's causal joint speech and completion model. One instance owns the
/// recurrent state of one stream, including silence between turns.
pub struct JointModel {
    session: Session,
    state: Array3<f32>,
}

impl JointModel {
    pub fn load(model_path: &Path, runtime_path: Option<&Path>, threads: usize) -> Result<Self> {
        ensure!(
            model_path.is_file(),
            "joint model not found: {}",
            model_path.display()
        );
        let session = session(model_path, runtime_path, threads)?;
        ensure!(
            session.inputs.len() == 2 && session.outputs.len() == 2,
            "joint model requires exactly two inputs and two outputs"
        );
        let input_type = |name: &str| {
            session
                .inputs
                .iter()
                .find(|input| input.name == name)
                .map(|input| &input.input_type)
        };
        check_joint_tensor(input_type("pcm"), "pcm", &[1, VAD_FRAME_SAMPLES as i64])?;
        let state_width = joint_state_width(input_type("state"))?;
        let state_shape = [1, 1, state_width];
        for (name, shape) in [("probabilities", &[1, 2][..]), ("next_state", &state_shape)] {
            check_joint_tensor(
                session
                    .outputs
                    .iter()
                    .find(|output| output.name == name)
                    .map(|output| &output.output_type),
                name,
                shape,
            )?;
        }
        Ok(Self {
            session,
            state: Array3::zeros((1, 1, state_width as usize)),
        })
    }

    /// Consume one frame of normalized 16 kHz PCM and return speech and
    /// completion probabilities in that order.
    pub fn process(&mut self, frame: &[f32]) -> Result<(f32, f32)> {
        ensure!(
            frame.len() == VAD_FRAME_SAMPLES,
            "joint model requires {VAD_FRAME_SAMPLES} samples at 16 kHz"
        );
        ensure!(
            frame
                .iter()
                .all(|sample| sample.is_finite() && (-1.0..=1.0).contains(sample)),
            "joint model requires finite normalized PCM in [-1, 1]"
        );
        let pcm = Array2::from_shape_vec((1, VAD_FRAME_SAMPLES), frame.to_vec())?;
        let outputs = self
            .session
            .run(ort::inputs! {
                "pcm" => pcm,
                "state" => self.state.view(),
            }?)
            .context("run Cue joint inference")?;
        let scores = outputs
            .get("probabilities")
            .context("missing joint probabilities")?
            .try_extract_tensor::<f32>()?;
        ensure!(
            scores.shape() == [1, 2],
            "invalid joint probability shape {:?}",
            scores.shape()
        );
        let speech = probability(scores[[0, 0]], "joint speech head")?;
        let completion = probability(scores[[0, 1]], "joint completion head")?;
        let state = outputs
            .get("next_state")
            .context("missing joint recurrent state")?
            .try_extract_tensor::<f32>()?;
        ensure!(
            state.shape() == self.state.shape() && state.iter().all(|value| value.is_finite()),
            "invalid joint recurrent state"
        );
        self.state
            .as_slice_mut()
            .expect("owned contiguous state")
            .copy_from_slice(
                state
                    .as_slice()
                    .context("non-contiguous joint recurrent state")?,
            );
        Ok((speech, completion))
    }

    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }
}

fn joint_state_width(value: Option<&ValueType>) -> Result<i64> {
    if let Some(ValueType::Tensor {
        ty: TensorElementType::Float32,
        dimensions,
        ..
    }) = value
    {
        match dimensions.as_slice() {
            [1, 1, width] if (1..=MAX_JOINT_STATE_SIZE).contains(width) => return Ok(*width),
            _ => {}
        }
    }
    Err(anyhow!(
        "joint model state must be float32 [1, 1, H] with H in 1..={MAX_JOINT_STATE_SIZE}, got {value:?}"
    ))
}

fn check_joint_tensor(value: Option<&ValueType>, name: &str, shape: &[i64]) -> Result<()> {
    ensure!(
        matches!(value, Some(ValueType::Tensor { ty: TensorElementType::Float32, dimensions, .. }) if dimensions == shape),
        "joint model tensor '{name}' must be float32 with shape {shape:?}, got {value:?}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joint_state_accepts_bounded_static_widths() {
        let tensor = |ty, dimensions: Vec<i64>| ValueType::Tensor {
            ty,
            dimension_symbols: vec![None; dimensions.len()],
            dimensions,
        };
        for width in [48, 96, MAX_JOINT_STATE_SIZE] {
            let state = tensor(TensorElementType::Float32, vec![1, 1, width]);
            assert_eq!(joint_state_width(Some(&state)).unwrap(), width);
        }
        for shape in [
            vec![1, 1, 0],
            vec![1, 1, -1],
            vec![1, 1, 257],
            vec![2, 1, 48],
        ] {
            let state = tensor(TensorElementType::Float32, shape);
            assert!(joint_state_width(Some(&state)).is_err());
        }
        let state = tensor(TensorElementType::Int64, vec![1, 1, 48]);
        assert!(joint_state_width(Some(&state)).is_err());
    }
}

fn probability(value: f32, model: &str) -> Result<f32> {
    ensure!(
        value.is_finite() && (0.0..=1.0).contains(&value),
        "{model} returned invalid probability {value}"
    );
    Ok(value)
}

/// Audio-only completion classifier with an optional score for the early gate.
pub struct SmartTurnModel {
    session: Session,
    early_output: Option<usize>,
}

impl SmartTurnModel {
    pub fn load(model_path: &Path, runtime_path: Option<&Path>, threads: usize) -> Result<Self> {
        let session = session(model_path, runtime_path, threads)?;
        ensure!(
            session.inputs.len() == 1 && session.inputs[0].name == "input_features",
            "expected Smart Turn ONNX input 'input_features'"
        );
        ensure!(
            !session.outputs.is_empty(),
            "Smart Turn model has no outputs"
        );
        let early_output = session
            .outputs
            .iter()
            .position(|output| output.name == "early_completion");
        if early_output.is_some() {
            ensure!(
                session.outputs.len() == 2
                    && session.outputs[0].name == "completion"
                    && early_output == Some(1),
                "staged completion outputs must be 'completion' followed by 'early_completion'"
            );
        }
        Ok(Self {
            session,
            early_output,
        })
    }

    /// Return probability of a completed turn for mono 16 kHz PCM.
    pub fn predict(&mut self, audio_16k: &[f32]) -> Result<f32> {
        self.predict_for_gate(audio_16k, false)
    }

    pub(crate) fn predict_for_gate(&mut self, audio_16k: &[f32], early: bool) -> Result<f32> {
        ensure!(
            !audio_16k.is_empty(),
            "Smart Turn requires nonempty 16 kHz audio"
        );
        let features = Array3::from_shape_vec((1, MEL_BINS, MEL_FRAMES), log_mel(audio_16k)?)?;
        let outputs = self
            .session
            .run(ort::inputs! { "input_features" => features }?)
            .context("run Smart Turn inference")?;
        let index = if early {
            self.early_output.unwrap_or(0)
        } else {
            0
        };
        let output = outputs[index].try_extract_tensor::<f32>()?;
        ensure!(
            output.len() == 1,
            "Smart Turn output must contain one probability, got {:?}",
            output.shape()
        );
        probability(
            *output.iter().next().context("empty Smart Turn output")?,
            "Smart Turn",
        )
    }
}

/// Stateful Silero VAD. Create one instance per audio stream and reset at stream
/// discontinuities. Context and recurrent state must persist across frames.
pub struct SileroVad {
    session: Session,
    context: [f32; VAD_CONTEXT_SAMPLES],
    state: Array3<f32>,
}

impl SileroVad {
    pub fn load(model_path: &Path, runtime_path: Option<&Path>, threads: usize) -> Result<Self> {
        let session = session(model_path, runtime_path, threads)?;
        ensure!(
            session.inputs.len() == 3
                && ["input", "state", "sr"]
                    .iter()
                    .all(|name| session.inputs.iter().any(|input| input.name == *name)),
            "expected Silero ONNX inputs 'input', 'state', 'sr'"
        );
        ensure!(
            session.outputs.len() == 2,
            "Silero model must provide probability and recurrent state"
        );
        Ok(Self {
            session,
            context: [0.0; VAD_CONTEXT_SAMPLES],
            state: Array3::zeros((2, 1, 128)),
        })
    }

    /// Consume exactly 512 fresh mono samples at 16 kHz and return speech probability.
    pub fn process(&mut self, frame: &[f32]) -> Result<f32> {
        ensure!(
            frame.len() == VAD_FRAME_SAMPLES,
            "Silero requires exactly {VAD_FRAME_SAMPLES} samples at 16 kHz, got {}",
            frame.len()
        );
        ensure!(
            frame.iter().all(|x| x.is_finite()),
            "audio contains NaN or infinity"
        );
        let mut samples = Vec::with_capacity(VAD_CONTEXT_SAMPLES + VAD_FRAME_SAMPLES);
        samples.extend_from_slice(&self.context);
        samples.extend_from_slice(frame);
        let input = Array2::from_shape_vec((1, samples.len()), samples)?;
        let outputs = self
            .session
            .run(ort::inputs! {
                "input" => input,
                "state" => self.state.view(),
                "sr" => arr0(16_000_i64),
            }?)
            .context("run Silero VAD inference")?;
        let output = outputs[0].try_extract_tensor::<f32>()?;
        ensure!(
            output.len() == 1,
            "Silero probability output has unexpected shape {:?}",
            output.shape()
        );
        let score = probability(
            *output.iter().next().context("empty Silero output")?,
            "Silero",
        )?;
        let next_state = outputs[1].try_extract_tensor::<f32>()?;
        ensure!(
            next_state.shape() == [2, 1, 128] && next_state.iter().all(|x| x.is_finite()),
            "Silero recurrent state is invalid"
        );
        self.state
            .as_slice_mut()
            .expect("owned contiguous state")
            .copy_from_slice(
                next_state
                    .as_slice()
                    .context("non-contiguous Silero state")?,
            );
        self.context
            .copy_from_slice(&frame[VAD_FRAME_SAMPLES - VAD_CONTEXT_SAMPLES..]);
        Ok(score)
    }

    pub fn reset(&mut self) {
        self.context.fill(0.0);
        self.state.fill(0.0);
    }
}
