//! Bounded PCM framing and shared acoustic observations for two policies.
//! Input is mono 16 kHz float PCM in [-1, 1].
use crate::inference::{JointModel, SileroVad, SmartTurnModel};
use crate::policy::{Config, Context, Decision, DecisionKind, Observation, TurnEngine};
use anyhow::{ensure, Result};
use serde::Serialize;
use std::collections::VecDeque;
use std::time::Instant;

pub const SAMPLE_RATE: usize = 16_000;
pub const FRAME_SAMPLES: usize = 512;
const WINDOW_SAMPLES: usize = SAMPLE_RATE * 8;

pub trait AcousticBackend {
    fn speech_probability(&mut self, frame: &[f32]) -> Result<f32>;
    fn completion_probability(&mut self, audio: &[f32]) -> Result<f32>;
    /// Single-output backends use the same score at either pause gate.
    fn completion_probability_for_gate(&mut self, audio: &[f32], _early: bool) -> Result<f32> {
        self.completion_probability(audio)
    }
    /// A completion score already computed by the most recent speech call.
    fn joint_completion_probability(&self) -> Option<f32> {
        None
    }
    fn reset(&mut self);
}

impl<B: AcousticBackend + ?Sized> AcousticBackend for Box<B> {
    fn speech_probability(&mut self, frame: &[f32]) -> Result<f32> {
        (**self).speech_probability(frame)
    }
    fn completion_probability(&mut self, audio: &[f32]) -> Result<f32> {
        (**self).completion_probability(audio)
    }
    fn completion_probability_for_gate(&mut self, audio: &[f32], early: bool) -> Result<f32> {
        (**self).completion_probability_for_gate(audio, early)
    }
    fn joint_completion_probability(&self) -> Option<f32> {
        (**self).joint_completion_probability()
    }
    fn reset(&mut self) {
        (**self).reset();
    }
}

pub struct NeuralBackend {
    pub turn: SmartTurnModel,
    pub vad: SileroVad,
}

impl AcousticBackend for NeuralBackend {
    fn speech_probability(&mut self, frame: &[f32]) -> Result<f32> {
        self.vad.process(frame)
    }
    fn completion_probability(&mut self, audio: &[f32]) -> Result<f32> {
        self.turn.predict(audio)
    }
    fn completion_probability_for_gate(&mut self, audio: &[f32], early: bool) -> Result<f32> {
        self.turn.predict_for_gate(audio, early)
    }
    fn reset(&mut self) {
        self.vad.reset();
    }
}

/// One recurrent model call per audio frame supplies both acoustic scores.
pub struct JointBackend {
    model: JointModel,
    completion: Option<f32>,
}

impl JointBackend {
    pub fn new(model: JointModel) -> Self {
        Self {
            model,
            completion: None,
        }
    }
}

impl AcousticBackend for JointBackend {
    fn speech_probability(&mut self, frame: &[f32]) -> Result<f32> {
        let (speech, completion) = self.model.process(frame)?;
        self.completion = Some(completion);
        Ok(speech)
    }
    fn completion_probability(&mut self, _: &[f32]) -> Result<f32> {
        self.completion
            .ok_or_else(|| anyhow::anyhow!("joint backend has no processed frame"))
    }
    fn joint_completion_probability(&self) -> Option<f32> {
        self.completion
    }
    fn reset(&mut self) {
        self.model.reset();
        self.completion = None;
    }
}

#[derive(Debug, Serialize)]
pub struct FrameResult {
    pub at_ms: u64,
    pub speech_probability: f32,
    pub completion_probability: Option<f32>,
    pub decision: Option<Decision>,
    pub baseline_decision: Option<Decision>,
    /// Legacy completion call including Rust features and ONNX, excluding loading.
    pub inference_ms: Option<f64>,
    /// Shared inference cost when the candidate actually requested a new score.
    pub candidate_inference_ms: Option<f64>,
    /// Shared inference cost when the diagnostic baseline requested a new score.
    pub baseline_inference_ms: Option<f64>,
    /// Separate legacy VAD call. Zero when inference supplies both scores.
    pub vad_ms: f64,
    /// One joint PCM-to-scores call, including preprocessing and state handling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub joint_inference_ms: Option<f64>,
    pub candidate_policy_ms: f64,
    pub baseline_policy_ms: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonMode {
    /// Continuous candidate stream. Baseline is a per-candidate-turn diagnostic,
    /// frozen after its first endpoint and reset when the candidate ends.
    /// It is not an independent multi-turn Smart Turn benchmark.
    Diagnostic,
    /// One annotated evaluation segment per reset. Each policy freezes after its
    /// first endpoint, and neither endpoint clears the shared acoustic history.
    Segment,
}

/// One instance per independent input stream. `push` is synchronous; put it on a
/// dedicated worker when integrating an async real-time transport.
pub struct StreamDetector<B: AcousticBackend> {
    backend: B,
    candidate: TurnEngine,
    baseline: TurnEngine,
    pending: Vec<f32>,
    window: VecDeque<f32>,
    samples_seen: u64,
    comparison_mode: ComparisonMode,
    candidate_finished: bool,
    baseline_finished: bool,
    poisoned: bool,
}

impl<B: AcousticBackend> StreamDetector<B> {
    pub fn new(backend: B, config: Config) -> Result<Self> {
        Self::new_with_comparison_mode(backend, config, ComparisonMode::Diagnostic)
    }

    pub fn new_with_comparison_mode(
        backend: B,
        config: Config,
        comparison_mode: ComparisonMode,
    ) -> Result<Self> {
        Ok(Self {
            backend,
            candidate: TurnEngine::new(config.clone())?,
            baseline: TurnEngine::new(config)?,
            pending: Vec::with_capacity(FRAME_SAMPLES),
            window: VecDeque::with_capacity(WINDOW_SAMPLES),
            samples_seen: 0,
            comparison_mode,
            candidate_finished: false,
            baseline_finished: false,
            poisoned: false,
        })
    }

    /// Context applies at each completed 32 ms frame. Invalid PCM changes no state;
    /// backend/policy errors poison the stream until `reset`.
    pub fn push(&mut self, samples: &[f32], context: &Context) -> Result<Vec<FrameResult>> {
        ensure!(
            !self.poisoned,
            "stream is poisoned after a processing failure; call reset before reuse"
        );
        ensure!(
            samples
                .iter()
                .all(|x| x.is_finite() && (-1.0..=1.0).contains(x)),
            "PCM must contain finite normalized samples in [-1, 1]"
        );
        let result = self.push_validated(samples, context);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn push_validated(&mut self, samples: &[f32], context: &Context) -> Result<Vec<FrameResult>> {
        let mut result = Vec::new();
        for &sample in samples {
            self.pending.push(sample);
            if self.pending.len() != FRAME_SAMPLES {
                continue;
            }
            self.samples_seen = self
                .samples_seen
                .checked_add(FRAME_SAMPLES as u64)
                .ok_or_else(|| anyhow::anyhow!("audio clock overflow"))?;
            let at_ms = self.samples_seen / (SAMPLE_RATE as u64 / 1000);
            let started = Instant::now();
            let speech_probability = self.backend.speech_probability(&self.pending)?;
            let acoustic_ms = started.elapsed().as_secs_f64() * 1000.0;
            let joint_score = self.backend.joint_completion_probability();
            let joint_inference_ms = joint_score.map(|_| acoustic_ms);
            ensure!(
                speech_probability.is_finite() && (0.0..=1.0).contains(&speech_probability),
                "VAD returned an invalid probability"
            );
            if let Some(score) = joint_score {
                ensure!(
                    score.is_finite() && (0.0..=1.0).contains(&score),
                    "joint model returned an invalid completion probability"
                );
            } else {
                self.window.extend(self.pending.iter().copied());
                while self.window.len() > WINDOW_SAMPLES {
                    self.window.pop_front();
                }
            }
            self.pending.clear();
            let observation = Observation {
                at_ms,
                speech_probability,
                completion_probability: joint_score,
                context: context.clone(),
            };
            let started = Instant::now();
            let mut decision = if self.candidate_finished {
                None
            } else {
                self.candidate.update(observation.clone())?
            };
            let mut candidate_policy_ms = started.elapsed().as_secs_f64() * 1000.0;
            let mut baseline_observation = observation.clone();
            baseline_observation.context = Context::default();
            let started = Instant::now();
            let mut baseline_decision = if self.baseline_finished {
                None
            } else {
                self.baseline.update(baseline_observation.clone())?
            };
            let mut baseline_policy_ms = started.elapsed().as_secs_f64() * 1000.0;
            let mut completion_probability = joint_score;
            let mut inference_ms = None;
            let mut candidate_inference_ms = None;
            let mut baseline_inference_ms = None;
            let candidate_needs_prediction = joint_score.is_none()
                && !self.candidate_finished
                && !preserve_action(&decision)
                && self.candidate.needs_prediction(at_ms);
            let baseline_needs_prediction = joint_score.is_none()
                && !self.baseline_finished
                && !preserve_action(&baseline_decision)
                && self.baseline.needs_prediction(at_ms);
            if candidate_needs_prediction || baseline_needs_prediction {
                // Active policies share the VAD clock and gate configuration.
                // Context can suppress a request without changing its stage.
                let early = if candidate_needs_prediction {
                    self.candidate.prediction_is_early(at_ms)
                } else {
                    self.baseline.prediction_is_early(at_ms)
                };
                debug_assert!(
                    !baseline_needs_prediction || early == self.baseline.prediction_is_early(at_ms)
                );
                let audio: Vec<f32> = self.window.iter().copied().collect();
                let started = Instant::now();
                let probability = self
                    .backend
                    .completion_probability_for_gate(&audio, early)?;
                inference_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
                ensure!(
                    probability.is_finite() && (0.0..=1.0).contains(&probability),
                    "turn model returned an invalid probability"
                );
                completion_probability = Some(probability);
                if candidate_needs_prediction {
                    let mut scored = observation;
                    scored.completion_probability = Some(probability);
                    let started = Instant::now();
                    // A scored pass may refine a provisional Hold. Policies that
                    // already ended are never updated again on this frame.
                    decision = self.candidate.update(scored)?.or(decision);
                    candidate_policy_ms += started.elapsed().as_secs_f64() * 1000.0;
                    candidate_inference_ms = inference_ms;
                }
                if baseline_needs_prediction {
                    baseline_observation.completion_probability = Some(probability);
                    let started = Instant::now();
                    baseline_decision = self
                        .baseline
                        .update(baseline_observation)?
                        .or(baseline_decision);
                    baseline_policy_ms += started.elapsed().as_secs_f64() * 1000.0;
                    baseline_inference_ms = inference_ms;
                }
            }
            if is_end(&baseline_decision) {
                self.baseline_finished = true;
            }
            if is_end(&decision) && self.comparison_mode == ComparisonMode::Segment {
                self.candidate_finished = true;
            }
            if self.comparison_mode == ComparisonMode::Diagnostic
                && decision.as_ref().is_some_and(|decision| {
                    matches!(decision.kind, DecisionKind::End | DecisionKind::Backchannel)
                })
            {
                self.window.clear();
                self.baseline.reset();
                self.baseline_finished = false;
            }
            result.push(FrameResult {
                at_ms,
                speech_probability,
                completion_probability,
                decision,
                baseline_decision,
                inference_ms,
                vad_ms: if joint_score.is_some() {
                    0.0
                } else {
                    acoustic_ms
                },
                joint_inference_ms,
                candidate_inference_ms,
                baseline_inference_ms,
                candidate_policy_ms,
                baseline_policy_ms,
            });
        }
        Ok(result)
    }

    pub fn reset(&mut self) {
        self.backend.reset();
        self.candidate.reset();
        self.baseline.reset();
        self.pending.clear();
        self.window.clear();
        self.samples_seen = 0;
        self.candidate_finished = false;
        self.baseline_finished = false;
        self.poisoned = false;
    }

    /// Samples not yet forming a whole 32 ms frame. They are never silently padded.
    pub fn buffered_samples(&self) -> usize {
        self.pending.len()
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
}

fn is_end(decision: &Option<Decision>) -> bool {
    decision
        .as_ref()
        .is_some_and(|decision| decision.kind == DecisionKind::End)
}

// A frame carries one action. In particular, a score must not replace an
// interruption with End before the caller has a chance to stop playback.
fn preserve_action(decision: &Option<Decision>) -> bool {
    decision.as_ref().is_some_and(|decision| {
        matches!(
            decision.kind,
            DecisionKind::End | DecisionKind::Backchannel | DecisionKind::Interrupt
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    struct SilentBackend;
    impl AcousticBackend for SilentBackend {
        fn speech_probability(&mut self, _: &[f32]) -> Result<f32> {
            Ok(0.0)
        }
        fn completion_probability(&mut self, _: &[f32]) -> Result<f32> {
            panic!("pure silence must not trigger turn inference")
        }
        fn reset(&mut self) {}
    }
    #[test]
    fn arbitrary_chunks_preserve_clock_and_do_not_infer_on_silence() {
        let mut engine = StreamDetector::new(SilentBackend, Config::default()).unwrap();
        let mut times = Vec::new();
        for size in [13, 499, 1000, 512, 24] {
            for frame in engine.push(&vec![0.0; size], &Context::default()).unwrap() {
                assert!(frame.decision.is_none());
                times.push(frame.at_ms);
            }
        }
        assert_eq!(times, vec![32, 64, 96, 128]);
        assert_eq!(engine.buffered_samples(), 0);
    }
    #[test]
    fn invalid_chunk_does_not_advance_state() {
        let mut engine = StreamDetector::new(SilentBackend, Config::default()).unwrap();
        let mut bad = vec![0.0; 1000];
        bad[999] = f32::NAN;
        assert!(engine.push(&bad, &Context::default()).is_err());
        assert!(!engine.is_poisoned());
        assert_eq!(engine.buffered_samples(), 0);
        assert_eq!(
            engine.push(&vec![0.0; 512], &Context::default()).unwrap()[0].at_ms,
            32
        );
    }

    #[test]
    fn joint_scores_update_during_a_pause_without_a_second_model_call() {
        struct JointFrames {
            frames: VecDeque<(f32, f32)>,
            completion: Option<f32>,
        }
        impl AcousticBackend for JointFrames {
            fn speech_probability(&mut self, _: &[f32]) -> Result<f32> {
                let (speech, completion) = self.frames.pop_front().unwrap();
                self.completion = Some(completion);
                Ok(speech)
            }
            fn completion_probability(&mut self, _: &[f32]) -> Result<f32> {
                panic!("joint backend already provided a completion score")
            }
            fn joint_completion_probability(&self) -> Option<f32> {
                self.completion
            }
            fn reset(&mut self) {
                self.completion = None;
            }
        }
        let backend = JointFrames {
            frames: [(0.9, 0.0), (0.9, 0.0), (0.0, 0.1), (0.0, 0.2), (0.0, 0.99)]
                .into_iter()
                .collect(),
            completion: None,
        };
        let mut detector = StreamDetector::new(backend, short_config()).unwrap();
        let frames = detector
            .push(&[0.0; FRAME_SAMPLES * 5], &Context::default())
            .unwrap();
        assert!(frames.iter().all(|frame| frame.joint_inference_ms.is_some()
            && frame.inference_ms.is_none()
            && frame.vad_ms == 0.0));
        assert!(frames[..4].iter().all(|frame| !is_end(&frame.decision)));
        assert!(is_end(&frames[4].decision));
        assert!(is_end(&frames[4].baseline_decision));
        assert_eq!(frames[4].at_ms, 160);
    }

    #[derive(Default)]
    struct ScriptedBackend {
        speech: VecDeque<f32>,
        scores: VecDeque<f32>,
        observed_audio: Vec<Vec<f32>>,
        observed_early_gates: Vec<bool>,
        fail_vad: bool,
        fail_model: bool,
        reset_calls: usize,
    }

    impl ScriptedBackend {
        fn new(speech: &[f32], scores: &[f32]) -> Self {
            Self {
                speech: speech.iter().copied().collect(),
                scores: scores.iter().copied().collect(),
                ..Self::default()
            }
        }
    }

    impl AcousticBackend for ScriptedBackend {
        fn speech_probability(&mut self, _: &[f32]) -> Result<f32> {
            if self.fail_vad {
                self.fail_vad = false;
                anyhow::bail!("scripted VAD failure");
            }
            Ok(self.speech.pop_front().unwrap_or(0.0))
        }
        fn completion_probability(&mut self, audio: &[f32]) -> Result<f32> {
            self.observed_audio.push(audio.to_vec());
            if self.fail_model {
                self.fail_model = false;
                anyhow::bail!("scripted model failure");
            }
            Ok(self.scores.pop_front().unwrap_or(1.0))
        }
        fn completion_probability_for_gate(&mut self, audio: &[f32], early: bool) -> Result<f32> {
            self.observed_early_gates.push(early);
            self.completion_probability(audio)
        }
        fn reset(&mut self) {
            self.reset_calls += 1;
        }
    }

    fn short_config() -> Config {
        Config {
            min_voiced_ms: 32,
            min_silence_ms: 32,
            max_silence_ms: 128,
            max_backchannel_ms: 128,
            ..Config::default()
        }
    }

    fn frame(engine: &mut StreamDetector<ScriptedBackend>, context: &Context) -> FrameResult {
        engine
            .push(&[0.25; FRAME_SAMPLES], context)
            .unwrap()
            .remove(0)
    }

    #[test]
    fn chunking_and_same_frame_scoring_preserve_turn_durations() {
        let speech = [0.9, 0.9, 0.0, 0.0, 0.9, 0.9, 0.0, 0.0];
        let mut whole =
            StreamDetector::new(ScriptedBackend::new(&speech, &[1.0, 1.0]), short_config())
                .unwrap();
        let mut chunked =
            StreamDetector::new(ScriptedBackend::new(&speech, &[1.0, 1.0]), short_config())
                .unwrap();
        let expected = whole.push(&vec![0.25; 4096], &Context::default()).unwrap();
        let mut actual = Vec::new();
        for size in [13, 499, 1000, 512, 24, 2048] {
            actual.extend(
                chunked
                    .push(&vec![0.25; size], &Context::default())
                    .unwrap(),
            );
        }
        let endpoints = |frames: Vec<FrameResult>| {
            frames
                .into_iter()
                .filter_map(|frame| frame.decision)
                .map(|decision| (decision.kind, decision.at_ms, decision.voiced_ms))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            endpoints(expected),
            vec![(DecisionKind::End, 128, 64), (DecisionKind::End, 256, 64)]
        );
        assert_eq!(
            endpoints(actual),
            vec![(DecisionKind::End, 128, 64), (DecisionKind::End, 256, 64)]
        );
        assert_eq!(chunked.backend.observed_audio.len(), 2);
    }

    #[test]
    fn early_and_late_checks_share_a_call_and_use_current_audio() {
        let backend = ScriptedBackend::new(&[0.9, 0.9, 0.0, 0.0, 0.0], &[0.9, 0.8]);
        let config = Config {
            min_silence_ms: 64,
            early_silence_ms: Some(32),
            early_completion_threshold: Some(0.99),
            ..short_config()
        };
        let mut detector =
            StreamDetector::new_with_comparison_mode(backend, config, ComparisonMode::Segment)
                .unwrap();
        let frames = detector
            .push(&[0.0; FRAME_SAMPLES * 5], &Context::default())
            .unwrap();
        assert_eq!(frames[3].completion_probability, Some(0.9));
        assert!(!is_end(&frames[3].decision));
        assert!(!is_end(&frames[3].baseline_decision));
        assert_eq!(frames[4].completion_probability, Some(0.8));
        assert!(is_end(&frames[4].decision));
        assert!(is_end(&frames[4].baseline_decision));
        let windows = &detector.backend.observed_audio;
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].len(), FRAME_SAMPLES * 4);
        assert_eq!(windows[1].len(), FRAME_SAMPLES * 5);
        assert_eq!(detector.backend.observed_early_gates, [true, false]);
    }

    #[test]
    fn segment_mode_preserves_baseline_audio_after_candidate_endpoint() {
        let backend = ScriptedBackend::new(
            &[
                0.9, 0.9, 0.0, 0.0, 0.0, 0.9, 0.9, 0.0, 0.0, 0.9, 0.9, 0.0, 0.0,
            ],
            &[0.1, 0.99, 0.99],
        );
        let mut engine = StreamDetector::new_with_comparison_mode(
            backend,
            short_config(),
            ComparisonMode::Segment,
        )
        .unwrap();
        let waiting = Context {
            text: "wait".to_owned(),
            ..Context::default()
        };
        for _ in 0..3 {
            frame(&mut engine, &waiting);
        }
        let baseline_score = frame(&mut engine, &waiting);
        assert!(baseline_score.baseline_inference_ms.is_some());
        assert!(baseline_score.candidate_inference_ms.is_none());
        let candidate_end = frame(&mut engine, &Context::default());
        assert!(is_end(&candidate_end.decision));
        assert!(candidate_end.candidate_inference_ms.is_some());
        assert!(candidate_end.baseline_inference_ms.is_none());
        for _ in 0..3 {
            assert!(frame(&mut engine, &Context::default()).decision.is_none());
        }
        let baseline_end = frame(&mut engine, &Context::default());
        assert!(is_end(&baseline_end.baseline_decision));
        assert!(baseline_end.decision.is_none());
        assert!(baseline_end.candidate_inference_ms.is_none());
        assert_eq!(
            engine
                .backend
                .observed_audio
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>(),
            vec![2048, 2560, 4608]
        );
        assert_eq!(engine.backend.observed_audio[2][0], 0.25);
        for _ in 0..4 {
            let held = frame(&mut engine, &Context::default());
            assert!(held.decision.is_none());
            assert!(held.baseline_decision.is_none());
            assert!(held.inference_ms.is_none());
        }
        engine.reset();
        assert!(!engine.candidate_finished);
        assert!(!engine.baseline_finished);
        assert!(engine.window.is_empty());
    }

    #[test]
    fn segment_mode_freezes_baseline_while_candidate_waits() {
        let backend = ScriptedBackend::new(&[0.9, 0.9, 0.0, 0.0, 0.9, 0.9, 0.0, 0.0], &[0.99]);
        let mut engine = StreamDetector::new_with_comparison_mode(
            backend,
            short_config(),
            ComparisonMode::Segment,
        )
        .unwrap();
        let waiting = Context {
            text: "wait".to_owned(),
            ..Context::default()
        };
        for _ in 0..3 {
            frame(&mut engine, &waiting);
        }
        assert!(is_end(&frame(&mut engine, &waiting).baseline_decision));
        for _ in 0..4 {
            let output = frame(&mut engine, &waiting);
            assert!(output.baseline_decision.is_none());
            assert!(output.inference_ms.is_none());
        }
        assert_eq!(engine.backend.observed_audio.len(), 1);
    }

    #[test]
    fn scored_pass_cannot_replace_interruption_with_endpoint() {
        let backend = ScriptedBackend::new(&[0.9, 0.9, 0.0, 0.0, 0.0], &[0.99, 0.99]);
        let mut engine = StreamDetector::new(backend, short_config()).unwrap();
        for _ in 0..3 {
            frame(&mut engine, &Context::default());
        }
        let stop = Context {
            agent_speaking: true,
            text: "stop".into(),
            text_final: true,
            ..Context::default()
        };
        let interrupted = frame(&mut engine, &stop);
        assert_eq!(interrupted.decision.unwrap().kind, DecisionKind::Interrupt);
        assert!(interrupted.candidate_inference_ms.is_none());
        assert!(interrupted.baseline_inference_ms.is_some());
        let ended = frame(&mut engine, &stop);
        assert!(is_end(&ended.decision));
        assert!(ended.candidate_inference_ms.is_some());
        assert!(ended.baseline_inference_ms.is_none());
    }

    #[test]
    fn backchannel_clears_consumed_audio_in_diagnostic_mode() {
        let backend = ScriptedBackend::new(&[0.9, 0.9, 0.0, 0.0], &[0.99]);
        let mut engine = StreamDetector::new(backend, short_config()).unwrap();
        let acknowledgement = Context {
            agent_speaking: true,
            text: "yeah".into(),
            text_final: true,
            ..Context::default()
        };
        for _ in 0..3 {
            frame(&mut engine, &acknowledgement);
        }
        let output = frame(&mut engine, &acknowledgement);
        assert_eq!(output.decision.unwrap().kind, DecisionKind::Backchannel);
        assert!(engine.window.is_empty());
        assert!(!engine.baseline_finished);
    }

    #[test]
    fn processing_failures_require_reset_but_reset_restores_framing() {
        let backend = ScriptedBackend {
            fail_vad: true,
            ..ScriptedBackend::default()
        };
        let mut engine = StreamDetector::new(backend, short_config()).unwrap();
        assert!(engine
            .push(&[0.0; FRAME_SAMPLES], &Context::default())
            .is_err());
        assert!(engine.is_poisoned());
        assert!(engine
            .push(&[0.0; FRAME_SAMPLES], &Context::default())
            .unwrap_err()
            .to_string()
            .contains("reset"));
        engine.reset();
        assert!(!engine.is_poisoned());
        assert_eq!(engine.buffered_samples(), 0);
        assert_eq!(frame(&mut engine, &Context::default()).at_ms, 32);
        assert_eq!(engine.backend.reset_calls, 1);
    }

    #[test]
    fn model_and_policy_failures_also_poison_the_stream() {
        let backend = ScriptedBackend {
            fail_model: true,
            ..ScriptedBackend::new(&[0.9, 0.9, 0.0, 0.0], &[])
        };
        let mut engine = StreamDetector::new(backend, short_config()).unwrap();
        for _ in 0..3 {
            frame(&mut engine, &Context::default());
        }
        assert!(engine
            .push(&[0.0; FRAME_SAMPLES], &Context::default())
            .is_err());
        assert!(engine.is_poisoned());
        engine.reset();
        let invalid = Context {
            expected_digits: Some(0),
            ..Context::default()
        };
        assert!(engine.push(&[0.0; FRAME_SAMPLES], &invalid).is_err());
        assert!(engine.is_poisoned());
    }

    #[test]
    fn invalid_backend_probability_poison_is_visible() {
        for invalid in [f32::NAN, 1.1, -0.1] {
            let mut engine =
                StreamDetector::new(ScriptedBackend::new(&[invalid], &[]), short_config()).unwrap();
            assert!(engine
                .push(&[0.0; FRAME_SAMPLES], &Context::default())
                .is_err());
            assert!(engine.is_poisoned());
        }
    }
}
