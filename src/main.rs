use anyhow::{bail, ensure, Context as _, Result};
use clap::{Parser, Subcommand};
use sao_turn::evaluation::{evaluate, SessionResult};
use sao_turn::inference::{JointModel, SileroVad, SmartTurnModel};
use sao_turn::policy::{Config, Context, DecisionKind, Observation, TurnEngine};
use sao_turn::stream::{
    AcousticBackend, ComparisonMode, FrameResult, JointBackend, NeuralBackend, StreamDetector,
    FRAME_SAMPLES, SAMPLE_RATE,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::json;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(version, about = "Cue: streaming CPU turn detection")]
struct Cli {
    #[arg(
        long,
        global = true,
        default_value = "assets/models/smart-turn-v3.2-cpu.onnx"
    )]
    model: PathBuf,
    #[arg(long, global = true, default_value = "assets/models/silero-vad.onnx")]
    vad_model: PathBuf,
    /// Use one Cue joint speech/completion ONNX model for WAV, stream or benchmark.
    #[arg(long, global = true)]
    joint_model: Option<PathBuf>,
    /// ONNX Runtime shared library; alternatively set ORT_DYLIB_PATH.
    #[arg(long, global = true)]
    runtime: Option<PathBuf>,
    #[arg(long, global = true, default_value_t = 1)]
    threads: usize,
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Score the last eight seconds of a 16 kHz mono WAV using the neural model.
    Score { input: PathBuf },
    /// Process a WAV through VAD and both policies, emitting decisions as JSONL.
    Wav {
        input: PathBuf,
        #[arg(long)]
        context: Option<PathBuf>,
        /// Timestamped context updates, available at these times, not before.
        #[arg(long)]
        events: Option<PathBuf>,
        /// Save observations for deterministic policy replay; refuses overwrite.
        #[arg(long)]
        trace: Option<PathBuf>,
    },
    /// JSONL stdin packets: {"samples":[signed16...],"context":{...},"reset":false}.
    Stream,
    /// Replay recorded Observation JSONL without loading ONNX models.
    Replay {
        input: PathBuf,
        /// Treat scores as per-frame outputs from the joint model.
        #[arg(long)]
        joint_scores: bool,
    },
    /// Evaluate paired SessionResult JSONL without neural inference.
    Compare { input: PathBuf },
    /// Run a WAV manifest through both policies, then evaluate first endpoints.
    Benchmark {
        manifest: PathBuf,
        /// Write the paired session results; refuses overwrite.
        #[arg(long)]
        results: Option<PathBuf>,
    },
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ContextEvent {
    at_ms: u64,
    context: Context,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Packet {
    #[serde(default)]
    samples: Vec<i16>,
    context: Option<Context>,
    #[serde(default)]
    reset: bool,
}

// Serialize typed f32 scores directly, consistently with WAV/trace output.
// json! would first promote them to f64 and change their decimal representation.
#[derive(Serialize)]
struct PacketResponse {
    frames: Vec<FrameResult>,
    buffered_samples: usize,
}

#[derive(Serialize)]
struct ReplayResponse {
    decision: Option<sao_turn::policy::Decision>,
    baseline_decision: Option<sao_turn::policy::Decision>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestEntry {
    id: String,
    audio: PathBuf,
    /// null is a known-incomplete segment, not an unlabelled one.
    #[serde(deserialize_with = "required_optional")]
    reference_end_ms: Option<u64>,
    #[serde(default)]
    context: Context,
    #[serde(default)]
    events: Vec<ContextEvent>,
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    serde_json::from_reader(BufReader::new(File::open(path)?))
        .with_context(|| format!("parse {}", path.display()))
}

fn required_optional<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<u64>, D::Error> {
    Option::<u64>::deserialize(deserializer)
}

fn read_jsonl<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    let mut rows = Vec::new();
    for (i, line) in BufReader::new(File::open(path)?).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        rows.push(
            serde_json::from_str(&line)
                .with_context(|| format!("parse {} line {}", path.display(), i + 1))?,
        );
    }
    Ok(rows)
}

fn new_output(path: &Path) -> Result<BufWriter<File>> {
    Ok(BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .with_context(|| format!("create new output {}", path.display()))?,
    ))
}

fn emit(writer: &mut impl Write, value: &impl serde::Serialize) -> Result<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writeln!(writer)?;
    Ok(())
}

fn read_wav(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    ensure!(
        spec.channels == 1 && spec.sample_rate == SAMPLE_RATE as u32,
        "{} must be mono 16000 Hz; resample explicitly before inference",
        path.display()
    );
    ensure!(reader.duration() > 0, "empty WAV input");
    let audio = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|x| x.map(|x| f32::from(x) / 32768.0))
            .collect::<Result<Vec<_>, _>>()?,
        (hound::SampleFormat::Float, 32) => {
            reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?
        }
        _ => bail!("WAV must contain signed 16-bit PCM or normalized 32-bit float samples"),
    };
    ensure!(
        audio
            .iter()
            .all(|x| x.is_finite() && (-1.0..=1.0).contains(x)),
        "WAV contains nonfinite or non-normalized samples"
    );
    Ok(audio)
}

fn backend(cli: &Cli) -> Result<Box<dyn AcousticBackend>> {
    if let Some(path) = &cli.joint_model {
        return Ok(Box::new(JointBackend::new(JointModel::load(
            path,
            cli.runtime.as_deref(),
            cli.threads,
        )?)));
    }
    Ok(Box::new(NeuralBackend {
        turn: SmartTurnModel::load(&cli.model, cli.runtime.as_deref(), cli.threads)?,
        vad: SileroVad::load(&cli.vad_model, cli.runtime.as_deref(), cli.threads)?,
    }))
}

fn config(cli: &Cli) -> Result<Config> {
    cli.config
        .as_deref()
        .map(read_json)
        .transpose()
        .map(|c| c.unwrap_or_default())
}

fn validate_events(events: &[ContextEvent], duration_ms: u64) -> Result<()> {
    let mut previous = None;
    for event in events {
        ensure!(
            event.at_ms <= duration_ms,
            "context event lies beyond audio duration"
        );
        ensure!(
            previous.is_none_or(|time| event.at_ms > time),
            "context event timestamps must be strictly increasing"
        );
        previous = Some(event.at_ms);
    }
    Ok(())
}

fn run_audio(
    detector: &mut StreamDetector<Box<dyn AcousticBackend>>,
    audio: &[f32],
    initial: &Context,
    events: &[ContextEvent],
    mut on_frame: impl FnMut(&FrameResult, &Context) -> Result<()>,
) -> Result<()> {
    validate_events(events, audio.len() as u64 / 16)?;
    let mut context = initial.clone();
    let mut event_index = 0;
    for (i, chunk) in audio.chunks(FRAME_SAMPLES).enumerate() {
        let at_ms = ((i + 1) * FRAME_SAMPLES) as u64 / 16;
        while event_index < events.len() && events[event_index].at_ms <= at_ms {
            context = events[event_index].context.clone();
            event_index += 1;
        }
        for frame in detector.push(chunk, &context)? {
            on_frame(&frame, &context)?;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    ensure!(
        cli.joint_model.is_none()
            || matches!(
                &cli.command,
                Command::Wav { .. } | Command::Stream | Command::Benchmark { .. }
            ),
        "--joint-model is supported by wav, stream, and benchmark"
    );
    let cfg = config(&cli)?;
    TurnEngine::new(cfg.clone())?;
    let stdout = io::stdout();
    let mut out = stdout.lock();
    match &cli.command {
        Command::Score { input } => {
            let audio = read_wav(input)?;
            let mut model = SmartTurnModel::load(&cli.model, cli.runtime.as_deref(), cli.threads)?;
            // Separate first-call warmup from the timed inference.
            let warmup_start = Instant::now();
            model.predict(&audio)?;
            let warmup_ms = warmup_start.elapsed().as_secs_f64() * 1000.0;
            let started = Instant::now();
            let probability = model.predict(&audio)?;
            emit(
                &mut out,
                &json!({"model_path":cli.model, "probability":probability,
                "complete":probability > 0.5, "audio_samples":audio.len(),
                "warmup_ms":warmup_ms, "inference_ms":started.elapsed().as_secs_f64()*1000.0,
                "timing_boundary":"feature extraction and ONNX execution; excludes loading"}),
            )?;
        }
        Command::Wav {
            input,
            context,
            events,
            trace,
        } => {
            let audio = read_wav(input)?;
            let initial: Context = context
                .as_deref()
                .map(read_json)
                .transpose()?
                .unwrap_or_default();
            let events: Vec<ContextEvent> = events
                .as_deref()
                .map(read_jsonl)
                .transpose()?
                .unwrap_or_default();
            validate_events(&events, audio.len() as u64 / 16)?;
            let mut detector = StreamDetector::new(backend(&cli)?, cfg)?;
            let mut trace = trace.as_deref().map(new_output).transpose()?;
            let mut calls = 0;
            run_audio(
                &mut detector,
                &audio,
                &initial,
                &events,
                |frame, context| {
                    if frame.inference_ms.is_some() || frame.joint_inference_ms.is_some() {
                        calls += 1;
                    }
                    if frame.decision.is_some() || frame.baseline_decision.is_some() {
                        emit(&mut out, frame)?;
                    }
                    if let Some(writer) = &mut trace {
                        emit(
                            writer,
                            &Observation {
                                at_ms: frame.at_ms,
                                speech_probability: frame.speech_probability,
                                completion_probability: frame.completion_probability,
                                context: context.clone(),
                            },
                        )?;
                    }
                    Ok(())
                },
            )?;
            if let Some(writer) = &mut trace {
                writer.flush()?;
            }
            eprintln!(
                "{}",
                json!({"samples":audio.len(), "unprocessed_tail_samples":detector.buffered_samples(),
                "turn_model_calls":calls, "padding_added":false})
            );
        }
        Command::Stream => {
            let mut detector = StreamDetector::new(backend(&cli)?, cfg)?;
            let mut context = Context::default();
            for (line_no, line) in io::stdin().lock().lines().enumerate() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                ensure!(line.len() <= 2_000_000, "stream packet exceeds 2 MB");
                let packet: Packet = serde_json::from_str(&line)
                    .with_context(|| format!("stream packet {}", line_no + 1))?;
                ensure!(
                    packet.samples.len() <= SAMPLE_RATE * 10,
                    "packet exceeds ten seconds of audio"
                );
                if packet.reset {
                    detector.reset();
                    context = Context::default();
                }
                if let Some(update) = packet.context {
                    context = update;
                }
                let audio: Vec<f32> = packet
                    .samples
                    .iter()
                    .map(|&x| f32::from(x) / 32768.0)
                    .collect();
                let frames = detector.push(&audio, &context)?;
                emit(
                    &mut out,
                    &PacketResponse {
                        frames,
                        buffered_samples: detector.buffered_samples(),
                    },
                )?;
                out.flush()?;
            }
        }
        Command::Replay {
            input,
            joint_scores,
        } => {
            let mut candidate = TurnEngine::new(cfg.clone())?;
            let mut baseline = TurnEngine::new(cfg)?;
            let mut baseline_finished = false;
            for observation in read_jsonl::<Observation>(input)? {
                let mut unscored = observation.clone();
                let mut raw = observation.clone();
                raw.context = Context::default();
                if !joint_scores {
                    unscored.completion_probability = None;
                    raw.completion_probability = None;
                }
                let mut decision = candidate.update(unscored)?;
                let mut baseline_decision = if baseline_finished {
                    None
                } else {
                    baseline.update(raw.clone())?
                };
                let actionable = |decision: &Option<sao_turn::policy::Decision>| {
                    decision.as_ref().is_some_and(|d| {
                        matches!(
                            d.kind,
                            DecisionKind::End | DecisionKind::Backchannel | DecisionKind::Interrupt
                        )
                    })
                };
                if !joint_scores && observation.completion_probability.is_some() {
                    if !actionable(&decision) && candidate.needs_prediction(observation.at_ms) {
                        decision = candidate.update(observation.clone())?.or(decision);
                    }
                    if !baseline_finished
                        && !actionable(&baseline_decision)
                        && baseline.needs_prediction(observation.at_ms)
                    {
                        raw.completion_probability = observation.completion_probability;
                        baseline_decision = baseline.update(raw)?.or(baseline_decision);
                    }
                }
                if baseline_decision
                    .as_ref()
                    .is_some_and(|d| d.kind == DecisionKind::End)
                {
                    baseline_finished = true;
                }
                if decision.as_ref().is_some_and(|d| {
                    matches!(d.kind, DecisionKind::End | DecisionKind::Backchannel)
                }) {
                    baseline.reset();
                    baseline_finished = false;
                }
                if decision.is_some() || baseline_decision.is_some() {
                    emit(
                        &mut out,
                        &ReplayResponse {
                            decision,
                            baseline_decision,
                        },
                    )?;
                }
            }
        }
        Command::Compare { input } => {
            let rows = read_jsonl::<SessionResult>(input)?;
            emit(&mut out, &evaluate(&rows)?)?;
        }
        Command::Benchmark { manifest, results } => {
            let entries = read_jsonl::<ManifestEntry>(manifest)?;
            ensure!(!entries.is_empty(), "empty benchmark manifest");
            let mut ids = std::collections::HashSet::new();
            for entry in &entries {
                ensure!(
                    !entry.id.trim().is_empty() && ids.insert(&entry.id),
                    "empty or duplicate session id"
                );
            }
            let parent = manifest.parent().unwrap_or(Path::new("."));
            let mut detector = StreamDetector::new_with_comparison_mode(
                backend(&cli)?,
                cfg.clone(),
                ComparisonMode::Segment,
            )?;
            let mut writer = results.as_deref().map(new_output).transpose()?;
            let mut rows = Vec::new();
            let mut acoustic_frames = 0_u64;
            let mut vad_ms = 0.0;
            let mut completion_ms = 0.0;
            let mut joint_ms = 0.0;
            for entry in entries {
                ensure!(entry.context.text.is_empty(),
                    "benchmark initial context cannot include a transcript; supply timestamped events to avoid future-text leakage");
                let path = if entry.audio.is_absolute() {
                    entry.audio
                } else {
                    parent.join(entry.audio)
                };
                let audio = read_wav(&path)?;
                ensure!(
                    entry
                        .reference_end_ms
                        .is_none_or(|t| t <= audio.len() as u64 / 16),
                    "reference endpoint beyond observation window in {}",
                    entry.id
                );
                detector.reset();
                let mut row = SessionResult {
                    id: entry.id,
                    reference_end_ms: entry.reference_end_ms,
                    baseline_end_ms: None,
                    candidate_end_ms: None,
                    baseline_inference_ms: vec![],
                    candidate_inference_ms: vec![],
                };
                run_audio(
                    &mut detector,
                    &audio,
                    &entry.context,
                    &entry.events,
                    |frame, _| {
                        acoustic_frames += 1;
                        vad_ms += frame.vad_ms;
                        completion_ms += frame.inference_ms.unwrap_or_default();
                        if let Some(ms) = frame.joint_inference_ms {
                            joint_ms += ms;
                            row.baseline_inference_ms.push(ms);
                            row.candidate_inference_ms.push(ms);
                        }
                        if frame
                            .baseline_decision
                            .as_ref()
                            .is_some_and(|d| d.kind == DecisionKind::End)
                            && row.baseline_end_ms.is_none()
                        {
                            row.baseline_end_ms = Some(frame.at_ms);
                        }
                        if frame
                            .decision
                            .as_ref()
                            .is_some_and(|d| d.kind == DecisionKind::End)
                            && row.candidate_end_ms.is_none()
                        {
                            row.candidate_end_ms = Some(frame.at_ms);
                        }
                        if let Some(ms) = frame.baseline_inference_ms {
                            row.baseline_inference_ms.push(ms);
                        }
                        if let Some(ms) = frame.candidate_inference_ms {
                            row.candidate_inference_ms.push(ms);
                        }
                        Ok(())
                    },
                )?;
                if let Some(writer) = &mut writer {
                    emit(writer, &row)?;
                    writer.flush()?;
                }
                rows.push(row);
            }
            let mut report = json!({"comparison":evaluate(&rows)?,
                "timing":"audio-clock endpoints; shared acoustic inference timings; excludes transport/STT and queueing",
                "threads":cli.threads,"config":cfg,"superiority_established":false,
                "acoustic_compute":{"frames":acoustic_frames,"vad_ms":vad_ms,
                    "completion_ms":completion_ms,"joint_ms":joint_ms,
                    "total_ms":vad_ms + completion_ms + joint_ms}});
            if let Some(path) = &cli.joint_model {
                report["baseline"] = json!(
                    "Cue joint model with identical thresholds and timeout; context disabled"
                );
                report["joint_model_path"] = json!(path);
                report["timing"] = json!("audio-clock endpoints; one shared joint inference call per frame; excludes transport/STT and queueing");
            } else {
                report["baseline"] = json!(
                    "Smart Turn v3.2 with identical VAD, thresholds and timeout; context disabled"
                );
                report["model_path"] = json!(cli.model);
                report["vad_model_path"] = json!(cli.vad_model);
            }
            emit(&mut out, &report)?;
        }
    }
    Ok(())
}
