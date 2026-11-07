//! Context heuristics around acoustic scores, using a monotonic session clock.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub speech_start_threshold: f32,
    pub speech_end_threshold: f32,
    pub completion_threshold: f32,
    pub min_voiced_ms: u64,
    pub min_silence_ms: u64,
    /// Optional earlier acoustic check with a stricter completion threshold.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub early_silence_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub early_completion_threshold: Option<f32>,
    /// Hard silence cap, including explicit waits and incomplete-input guards.
    /// Reaching this cap emits a forced end, even without a completion score.
    pub max_silence_ms: u64,
    pub max_backchannel_ms: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            speech_start_threshold: 0.65,
            speech_end_threshold: 0.35,
            completion_threshold: 0.50,
            min_voiced_ms: 200,
            min_silence_ms: 200,
            early_silence_ms: None,
            early_completion_threshold: None,
            max_silence_ms: 8_000,
            max_backchannel_ms: 1_200,
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        probability("speech_start_threshold", self.speech_start_threshold)?;
        probability("speech_end_threshold", self.speech_end_threshold)?;
        probability("completion_threshold", self.completion_threshold)?;
        if self.speech_end_threshold >= self.speech_start_threshold {
            bail!("speech_end_threshold must be below speech_start_threshold");
        }
        if self.min_voiced_ms == 0 {
            bail!("min_voiced_ms must be positive");
        }
        if self.min_silence_ms == 0 || self.max_silence_ms < self.min_silence_ms {
            bail!("silence limits must satisfy 0 < min_silence_ms <= max_silence_ms");
        }
        match (self.early_silence_ms, self.early_completion_threshold) {
            (Some(gate), Some(threshold)) => {
                probability("early_completion_threshold", threshold)?;
                if gate == 0 || gate >= self.min_silence_ms {
                    bail!("early_silence_ms must be positive and below min_silence_ms");
                }
                if threshold < self.completion_threshold {
                    bail!("early_completion_threshold must be at least completion_threshold");
                }
            }
            (None, None) => {}
            _ => bail!("early_silence_ms and early_completion_threshold must be set together"),
        }
        if self.max_backchannel_ms < self.min_voiced_ms {
            bail!("max_backchannel_ms must be at least min_voiced_ms");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextMode {
    #[default]
    Conversation,
    Digits,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Context {
    pub agent_speaking: bool,
    pub mode: ContextMode,
    pub expected_digits: Option<usize>,
    /// Current-turn transcript only. The caller should clear it between turns.
    pub text: String,
    pub text_final: bool,
}

impl Context {
    pub fn validate(&self) -> Result<()> {
        if self.text.len() > 65_536 {
            bail!("context text must not exceed 65536 UTF-8 bytes");
        }
        if self
            .expected_digits
            .is_some_and(|count| !(1..=128).contains(&count))
        {
            bail!("expected_digits must be between 1 and 128 when supplied");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub at_ms: u64,
    pub speech_probability: f32,
    #[serde(default)]
    pub completion_probability: Option<f32>,
    #[serde(default)]
    pub context: Context,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionKind {
    /// Unchanged state is emitted as `None`; this variant is reserved for consumers.
    Continue,
    End,
    Wait,
    Backchannel,
    Interrupt,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Decision {
    pub kind: DecisionKind,
    pub at_ms: u64,
    pub reason: String,
    /// Completion probability supplied by the acoustic model.
    pub score: Option<f32>,
    pub forced: bool,
    pub turn_start_ms: u64,
    pub last_speech_ms: u64,
    pub voiced_ms: u64,
}

/// Policy inputs exposed for evaluation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Features {
    pub voiced_ms: u64,
    pub silence_ms: u64,
    pub speech_active: bool,
    pub agent_speaking: bool,
    pub digit_count: usize,
    pub expected_digits: Option<usize>,
    pub text_final: bool,
    pub explicit_wait: bool,
    pub unfinished_text: bool,
    pub exact_backchannel: bool,
    pub interruption_cue: bool,
    pub completion_probability: Option<f32>,
}

#[derive(Debug)]
pub struct TurnEngine {
    config: Config,
    last_at_ms: Option<u64>,
    turn_start_ms: Option<u64>,
    last_speech_ms: Option<u64>,
    voiced_ms: u64,
    speech_active: bool,
    completion_probability: Option<f32>,
    early_prediction_used: bool,
    context: Context,
    last_wait_reason: Option<&'static str>,
    interruption_emitted: bool,
    // Reject an unchanged terminal transcript carried accidentally into the
    // next turn. Explicit reset permits a genuinely repeated transcript.
    terminal_text: Option<String>,
}

impl TurnEngine {
    pub fn new(config: Config) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            last_at_ms: None,
            turn_start_ms: None,
            last_speech_ms: None,
            voiced_ms: 0,
            speech_active: false,
            completion_probability: None,
            early_prediction_used: false,
            context: Context::default(),
            last_wait_reason: None,
            interruption_emitted: false,
            terminal_text: None,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Clear the session clock and all turn/context state. Call this on an
    /// external new-turn boundary; it also permits an identical new transcript.
    pub fn reset(&mut self) {
        self.clear_turn();
        self.last_at_ms = None;
        self.terminal_text = None;
    }

    /// Whether the paused turn needs a score for the current silence gate.
    /// Each gate requests at most one score before speech resumes.
    /// Call after supplying the latest VAD/context observation. Calling update
    /// again at the same timestamp to attach the score is supported.
    pub fn needs_prediction(&self, at_ms: u64) -> bool {
        let Some(last_at) = self.last_at_ms else {
            return false;
        };
        if at_ms < last_at
            || self.turn_start_ms.is_none()
            || self.speech_active
            || self.voiced_ms < self.config.min_voiced_ms
            || self.completion_probability.is_some()
        {
            return false;
        }
        let silence = self.silence_ms(at_ms);
        (silence >= self.config.min_silence_ms || self.early_check_due(silence))
            && silence < self.config.max_silence_ms
            && self.guard_reason().is_none()
            && !self.is_backchannel()
    }

    /// Stage of a completion check already requested by `needs_prediction`.
    pub(crate) fn prediction_is_early(&self, at_ms: u64) -> bool {
        self.silence_ms(at_ms) < self.config.min_silence_ms
    }

    pub fn features(&self, at_ms: u64) -> Option<Features> {
        self.turn_start_ms?;
        if at_ms < self.last_at_ms? {
            return None;
        }
        let normalized = normalize(&self.context.text);
        Some(Features {
            voiced_ms: self.voiced_ms,
            silence_ms: if self.speech_active {
                0
            } else {
                at_ms.saturating_sub(self.last_speech_ms?)
            },
            speech_active: self.speech_active,
            agent_speaking: self.context.agent_speaking,
            digit_count: count_digits(&normalized),
            expected_digits: self.context.expected_digits,
            text_final: self.context.text_final,
            explicit_wait: explicit_wait(&normalized),
            unfinished_text: !self.context.text_final && unfinished_text(&normalized),
            exact_backchannel: backchannel(&normalized),
            interruption_cue: interruption(&normalized),
            completion_probability: self.completion_probability,
        })
    }

    /// Consume a point-in-time observation. Durations use the preceding VAD
    /// state for the interval since the previous observation. Invalid inputs
    /// are rejected before mutating state. Supply observations at a regular
    /// audio cadence; an omitted interval cannot be reconstructed from text.
    pub fn update(&mut self, observation: Observation) -> Result<Option<Decision>> {
        probability("speech_probability", observation.speech_probability)?;
        if let Some(score) = observation.completion_probability {
            probability("completion_probability", score)?;
        }
        if self.last_at_ms.is_some_and(|last| observation.at_ms < last) {
            bail!("observation timestamps must not move backwards");
        }
        observation.context.validate()?;

        let at_ms = observation.at_ms;
        let elapsed = self.last_at_ms.map_or(0, |last| at_ms - last);
        if self.speech_active {
            self.voiced_ms = self.voiced_ms.saturating_add(elapsed);
            self.last_speech_ms = Some(at_ms);
        }
        self.last_at_ms = Some(at_ms);

        // An explicit transcript clear is a caller-provided new-context signal,
        // including when it arrives during idle silence between two turns.
        if observation.context.text.trim().is_empty() {
            self.terminal_text = None;
        }

        let was_active = self.speech_active;
        self.speech_active = if was_active {
            observation.speech_probability > self.config.speech_end_threshold
        } else {
            observation.speech_probability >= self.config.speech_start_threshold
        };

        if self.speech_active {
            self.turn_start_ms.get_or_insert(at_ms);
            self.last_speech_ms = Some(at_ms);
            self.completion_probability = None;
            self.early_prediction_used = false;
            self.last_wait_reason = None;
        }

        // Pure silence cannot create a turn, retain a score, or emit an event.
        if self.turn_start_ms.is_none() {
            self.context = Context::default();
            self.completion_probability = None;
            return Ok(None);
        }

        self.context = observation.context;
        if let Some(previous) = &self.terminal_text {
            let current = self.context.text.trim();
            if current == previous {
                self.context.text.clear();
                self.context.text_final = false;
            } else if !current.is_empty() {
                self.terminal_text = None;
            }
        }

        // Never accept a score while speech is active, including a score that
        // arrives in the same observation as resumed speech.
        let silence_ms = self.silence_ms(at_ms);
        if !self.speech_active
            && (self.config.early_silence_ms.is_none() || silence_ms >= self.config.min_silence_ms)
        {
            if let Some(score) = observation.completion_probability {
                self.completion_probability = Some(score);
            }
        }

        if self.voiced_ms < self.config.min_voiced_ms {
            if !self.speech_active && self.silence_ms(at_ms) >= self.config.max_silence_ms {
                self.finish_turn();
            }
            return Ok(None);
        }

        if !self.interruption_emitted
            && self.context.agent_speaking
            && self.context.text_final
            && interruption(&normalize(&self.context.text))
        {
            self.interruption_emitted = true;
            return Ok(Some(self.decision(
                DecisionKind::Interrupt,
                at_ms,
                "explicit_interruption_while_agent_speaking",
                false,
            )));
        }

        if self.speech_active {
            return Ok(None);
        }
        if silence_ms < self.config.min_silence_ms {
            if let (Some(score), Some(threshold)) = (
                observation.completion_probability,
                self.config.early_completion_threshold,
            ) {
                if self.early_check_due(silence_ms)
                    && self.guard_reason().is_none()
                    && !self.is_backchannel()
                {
                    self.early_prediction_used = true;
                    if score >= threshold {
                        self.completion_probability = Some(score);
                        let decision =
                            self.decision(DecisionKind::End, at_ms, "early_model_complete", false);
                        self.finish_turn();
                        return Ok(Some(decision));
                    }
                }
            }
            return Ok(None);
        }

        if self.is_backchannel() {
            let decision = self.decision(
                DecisionKind::Backchannel,
                at_ms,
                "short_final_backchannel_while_agent_speaking",
                false,
            );
            self.finish_turn();
            return Ok(Some(decision));
        }

        if silence_ms >= self.config.max_silence_ms {
            let reason = if self.guard_reason().is_some() {
                "context_wait_timeout"
            } else {
                "maximum_silence_timeout"
            };
            let decision = self.decision(DecisionKind::End, at_ms, reason, true);
            self.finish_turn();
            return Ok(Some(decision));
        }

        if let Some(reason) = self.guard_reason() {
            if self.last_wait_reason != Some(reason) {
                self.last_wait_reason = Some(reason);
                return Ok(Some(self.decision(
                    DecisionKind::Wait,
                    at_ms,
                    reason,
                    false,
                )));
            }
            return Ok(None);
        }
        self.last_wait_reason = None;

        if self
            .completion_probability
            .is_some_and(|score| score >= self.config.completion_threshold)
        {
            let decision = self.decision(DecisionKind::End, at_ms, "model_complete", false);
            self.finish_turn();
            return Ok(Some(decision));
        }
        Ok(None)
    }

    fn silence_ms(&self, at_ms: u64) -> u64 {
        at_ms.saturating_sub(self.last_speech_ms.unwrap_or(at_ms))
    }

    fn early_check_due(&self, silence_ms: u64) -> bool {
        !self.early_prediction_used
            && self
                .config
                .early_silence_ms
                .is_some_and(|gate| silence_ms >= gate)
    }

    fn guard_reason(&self) -> Option<&'static str> {
        let text = normalize(&self.context.text);
        if explicit_wait(&text) {
            return Some("explicit_wait_request");
        }
        if self.context.mode == ContextMode::Digits {
            if let Some(expected) = self.context.expected_digits {
                if count_digits(&text) < expected {
                    return Some("incomplete_digit_sequence");
                }
            }
        }
        if !self.context.text_final && unfinished_text(&text) {
            return Some("unfinished_partial_transcript");
        }
        None
    }

    fn is_backchannel(&self) -> bool {
        self.context.agent_speaking
            && self.context.text_final
            && self.voiced_ms <= self.config.max_backchannel_ms
            && self.guard_reason().is_none()
            && backchannel(&normalize(&self.context.text))
    }

    fn decision(&self, kind: DecisionKind, at_ms: u64, reason: &str, forced: bool) -> Decision {
        Decision {
            kind,
            at_ms,
            reason: reason.to_owned(),
            score: self.completion_probability,
            forced,
            turn_start_ms: self
                .turn_start_ms
                .expect("decisions require an active turn"),
            last_speech_ms: self.last_speech_ms.expect("decisions require voiced audio"),
            voiced_ms: self.voiced_ms,
        }
    }

    fn finish_turn(&mut self) {
        let transcript = self.context.text.trim();
        if !transcript.is_empty() {
            self.terminal_text = Some(transcript.to_owned());
        }
        self.clear_turn();
    }

    fn clear_turn(&mut self) {
        self.turn_start_ms = None;
        self.last_speech_ms = None;
        self.voiced_ms = 0;
        self.speech_active = false;
        self.completion_probability = None;
        self.early_prediction_used = false;
        self.context = Context::default();
        self.last_wait_reason = None;
        self.interruption_emitted = false;
    }
}

fn probability(name: &str, value: f32) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        bail!("{name} must be a finite probability in [0, 1]");
    }
    Ok(())
}

fn normalize(text: &str) -> String {
    text.chars()
        .flat_map(char::to_lowercase)
        .map(|character| {
            if character.is_alphanumeric() {
                character
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn explicit_wait(text: &str) -> bool {
    matches!(
        text,
        "wait"
            | "please wait"
            | "wait please"
            | "wait a second"
            | "wait a moment"
            | "hold on"
            | "hang on"
            | "one moment"
            | "one second"
            | "give me a moment"
            | "give me a second"
    )
}

fn interruption(text: &str) -> bool {
    matches!(
        text,
        "stop"
            | "stop talking"
            | "please stop"
            | "stop please"
            | "let me speak"
            | "excuse me"
            | "wait"
            | "please wait"
            | "hold on"
    )
}

fn backchannel(text: &str) -> bool {
    matches!(
        text,
        "yes"
            | "yeah"
            | "yep"
            | "uh huh"
            | "mm hmm"
            | "mhm"
            | "okay"
            | "ok"
            | "right"
            | "i see"
            | "sure"
            | "got it"
    )
}

fn unfinished_text(text: &str) -> bool {
    matches!(
        text.split_whitespace().last(),
        Some("and" | "or" | "but" | "because" | "so" | "um" | "uh" | "erm")
    ) || ["i mean", "let me"].iter().any(|suffix| {
        text == *suffix
            || text
                .strip_suffix(*suffix)
                .is_some_and(|prefix| prefix.ends_with(' '))
    })
}

fn count_digits(text: &str) -> usize {
    let numeric = text.chars().filter(char::is_ascii_digit).count();
    let spoken = text
        .split_whitespace()
        .filter(|token| {
            matches!(
                *token,
                "zero"
                    | "oh"
                    | "one"
                    | "two"
                    | "three"
                    | "four"
                    | "five"
                    | "six"
                    | "seven"
                    | "eight"
                    | "nine"
            )
        })
        .count();
    numeric + spoken
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> TurnEngine {
        TurnEngine::new(Config {
            min_voiced_ms: 100,
            min_silence_ms: 200,
            max_silence_ms: 1_000,
            max_backchannel_ms: 500,
            ..Config::default()
        })
        .unwrap()
    }

    fn observation(at_ms: u64, speech: f32, score: Option<f32>, text: &str) -> Observation {
        Observation {
            at_ms,
            speech_probability: speech,
            completion_probability: score,
            context: Context {
                text: text.to_owned(),
                ..Context::default()
            },
        }
    }

    fn voiced(engine: &mut TurnEngine, context: Context) {
        for (at_ms, speech_probability) in [(0, 0.9), (100, 0.9), (200, 0.0)] {
            engine
                .update(Observation {
                    at_ms,
                    speech_probability,
                    completion_probability: None,
                    context: context.clone(),
                })
                .unwrap();
        }
    }

    #[test]
    fn silence_never_creates_a_response_or_retains_a_score() {
        let mut engine = engine();
        for at in [0, 100, 5_000, 20_000] {
            assert!(engine
                .update(observation(at, 0.0, Some(1.0), "wait"))
                .unwrap()
                .is_none());
            assert!(!engine.needs_prediction(at));
            assert!(engine.features(at).is_none());
        }
    }

    #[test]
    fn completion_needs_speech_and_silence_and_ends_only_once() {
        let mut engine = engine();
        voiced(&mut engine, Context::default());
        assert!(!engine.needs_prediction(399));
        assert!(engine.needs_prediction(400));
        let end = engine
            .update(observation(400, 0.0, Some(0.9), "done"))
            .unwrap()
            .unwrap();
        assert_eq!(end.kind, DecisionKind::End);
        assert_eq!(end.turn_start_ms, 0);
        assert_eq!(end.last_speech_ms, 200);
        assert_eq!(end.voiced_ms, 200);
        assert!(!end.forced);
        assert_eq!(end.score, Some(0.9));
        assert!(engine
            .update(observation(500, 0.0, Some(0.9), "done"))
            .unwrap()
            .is_none());
        assert!(!engine.needs_prediction(500));
    }

    #[test]
    fn early_rejection_requires_a_fresh_score_at_the_late_gate() {
        let mut engine = TurnEngine::new(Config {
            early_silence_ms: Some(32),
            early_completion_threshold: Some(0.99),
            ..engine().config().clone()
        })
        .unwrap();
        voiced(&mut engine, Context::default());
        assert!(!engine.needs_prediction(231));
        assert!(engine.needs_prediction(232));
        assert!(engine
            .update(observation(232, 0.0, Some(0.9), ""))
            .unwrap()
            .is_none());
        assert!(!engine.needs_prediction(264));
        assert!(engine
            .update(observation(400, 0.0, None, ""))
            .unwrap()
            .is_none());
        assert!(engine.needs_prediction(400));
        assert!(engine
            .update(observation(400, 0.0, Some(0.1), ""))
            .unwrap()
            .is_none());
        assert!(!engine.needs_prediction(432));
        engine.update(observation(464, 0.9, None, "")).unwrap();
        engine.update(observation(496, 0.0, None, "")).unwrap();
        assert!(engine.needs_prediction(528));
        let decision = engine
            .update(observation(528, 0.0, Some(0.99), ""))
            .unwrap()
            .unwrap();
        assert_eq!(decision.kind, DecisionKind::End);
        assert_eq!(decision.reason, "early_model_complete");
        assert_eq!(decision.voiced_ms, 232);
    }

    #[test]
    fn early_gate_respects_context_and_uses_late_score_if_hold_clears_late() {
        let mut engine = TurnEngine::new(Config {
            early_silence_ms: Some(32),
            early_completion_threshold: Some(0.99),
            ..engine().config().clone()
        })
        .unwrap();
        voiced(
            &mut engine,
            Context {
                text: "wait".into(),
                ..Context::default()
            },
        );
        assert!(!engine.needs_prediction(232));
        assert!(engine
            .update(observation(232, 0.0, Some(1.0), "wait"))
            .unwrap()
            .is_none());
        assert!(engine
            .update(observation(400, 0.0, None, "done"))
            .unwrap()
            .is_none());
        assert!(engine.needs_prediction(400));
        let decision = engine
            .update(observation(400, 0.0, Some(0.5), "done"))
            .unwrap()
            .unwrap();
        assert_eq!(decision.kind, DecisionKind::End);
        assert_eq!(decision.reason, "model_complete");
    }

    #[test]
    fn resumed_speech_invalidates_even_a_score_arriving_with_that_speech() {
        let mut engine = engine();
        voiced(&mut engine, Context::default());
        engine
            .update(observation(250, 0.0, Some(0.99), ""))
            .unwrap();
        engine
            .update(observation(300, 0.9, Some(0.99), ""))
            .unwrap();
        assert_eq!(engine.features(300).unwrap().completion_probability, None);
        engine.update(observation(400, 0.0, None, "")).unwrap();
        assert!(engine
            .update(observation(600, 0.0, None, ""))
            .unwrap()
            .is_none());
        assert!(engine.needs_prediction(600));
        assert!(engine
            .update(observation(600, 0.0, Some(0.95), ""))
            .unwrap()
            .is_some());
    }

    #[test]
    fn hysteresis_does_not_chatter_and_blips_cannot_end() {
        let mut engine = engine();
        engine.update(observation(0, 0.5, None, "")).unwrap();
        assert!(engine.features(0).is_none());
        engine.update(observation(10, 0.9, None, "")).unwrap();
        engine.update(observation(20, 0.5, None, "")).unwrap();
        assert!(engine.features(20).unwrap().speech_active);
        engine.update(observation(30, 0.1, None, "")).unwrap();
        assert!(engine
            .update(observation(300, 0.5, Some(1.0), ""))
            .unwrap()
            .is_none());
        assert!(!engine.needs_prediction(300));
        assert!(engine
            .update(observation(1_030, 0.0, None, ""))
            .unwrap()
            .is_none());
        assert!(engine.features(1_030).is_none());
    }

    #[test]
    fn invalid_inputs_do_not_advance_state() {
        let mut engine = engine();
        engine.update(observation(100, 0.9, None, "")).unwrap();
        for bad in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
            assert!(engine.update(observation(200, bad, None, "")).is_err());
            assert!(engine.update(observation(200, 0.0, Some(bad), "")).is_err());
        }
        assert!(engine.update(observation(99, 0.0, None, "")).is_err());
        assert!(engine.update(observation(100, 0.9, None, "")).is_ok());
        assert_eq!(engine.features(100).unwrap().voiced_ms, 0);
        assert!(!engine.needs_prediction(99));
        assert!(engine.features(99).is_none());
    }

    #[test]
    fn explicit_wait_emits_once_then_has_a_bounded_timeout() {
        let mut engine = engine();
        voiced(
            &mut engine,
            Context {
                text: "Please wait!".into(),
                text_final: true,
                ..Context::default()
            },
        );
        let wait = engine
            .update(observation(400, 0.0, Some(0.99), "Please wait!"))
            .unwrap()
            .unwrap();
        assert_eq!(wait.kind, DecisionKind::Wait);
        assert_eq!(wait.reason, "explicit_wait_request");
        assert!(!wait.forced);
        assert!(!engine.needs_prediction(400));
        assert!(engine
            .update(observation(700, 0.0, None, "Please wait!"))
            .unwrap()
            .is_none());
        let end = engine
            .update(observation(1_200, 0.0, None, "Please wait!"))
            .unwrap()
            .unwrap();
        assert_eq!(end.kind, DecisionKind::End);
        assert_eq!(end.reason, "context_wait_timeout");
        assert!(end.forced);
    }

    #[test]
    fn absent_or_low_scores_use_explicit_timeout_not_false_confidence() {
        for score in [None, Some(0.1)] {
            let mut engine = engine();
            voiced(&mut engine, Context::default());
            assert!(engine
                .update(observation(400, 0.0, score, ""))
                .unwrap()
                .is_none());
            let end = engine
                .update(observation(1_200, 0.0, None, ""))
                .unwrap()
                .unwrap();
            assert!(end.forced);
            assert_eq!(end.reason, "maximum_silence_timeout");
            assert_eq!(end.score, score);
        }
    }

    #[test]
    fn digits_guard_releases_on_context_change() {
        let mut engine = engine();
        let context = Context {
            mode: ContextMode::Digits,
            expected_digits: Some(4),
            text: "one 23".into(),
            text_final: true,
            ..Context::default()
        };
        voiced(&mut engine, context.clone());
        let mut sample = observation(400, 0.0, Some(0.99), "");
        sample.context = context;
        let wait = engine.update(sample.clone()).unwrap().unwrap();
        assert_eq!(wait.reason, "incomplete_digit_sequence");
        assert_eq!(engine.features(400).unwrap().digit_count, 3);
        sample.at_ms = 500;
        sample.context.expected_digits = Some(3);
        sample.completion_probability = None;
        assert_eq!(
            engine.update(sample).unwrap().unwrap().kind,
            DecisionKind::End
        );
    }

    #[test]
    fn trailing_partial_guard_disappears_when_text_becomes_final() {
        let mut engine = engine();
        voiced(&mut engine, Context::default());
        let wait = engine
            .update(observation(400, 0.0, Some(0.99), "red and"))
            .unwrap()
            .unwrap();
        assert_eq!(wait.reason, "unfinished_partial_transcript");
        let mut sample = observation(500, 0.0, None, "red and");
        sample.context.text_final = true;
        assert_eq!(
            engine.update(sample).unwrap().unwrap().kind,
            DecisionKind::End
        );
    }

    #[test]
    fn yes_is_backchannel_only_while_agent_is_speaking() {
        for agent_speaking in [false, true] {
            let mut engine = engine();
            let context = Context {
                agent_speaking,
                text: "Yes.".into(),
                text_final: true,
                ..Context::default()
            };
            voiced(&mut engine, context.clone());
            let mut sample = observation(400, 0.0, Some(0.99), "");
            sample.context = context;
            let decision = engine.update(sample).unwrap().unwrap();
            assert_eq!(
                decision.kind,
                if agent_speaking {
                    DecisionKind::Backchannel
                } else {
                    DecisionKind::End
                }
            );
        }
    }

    #[test]
    fn backchannel_requires_exact_final_short_text() {
        for (text, text_final) in [
            ("yeah but no", true),
            ("yeah", false),
            ("unrecognized", true),
        ] {
            let mut engine = engine();
            let context = Context {
                agent_speaking: true,
                text: text.into(),
                text_final,
                ..Context::default()
            };
            voiced(&mut engine, context.clone());
            let mut sample = observation(400, 0.0, None, "");
            sample.context = context;
            assert!(engine.update(sample).unwrap().is_none());
            assert!(engine.needs_prediction(400));
        }
        let mut engine = engine();
        let context = Context {
            agent_speaking: true,
            text: "yeah".into(),
            text_final: true,
            ..Context::default()
        };
        for (at_ms, speech_probability) in [(0, 0.9), (600, 0.0), (800, 0.0)] {
            assert!(engine
                .update(Observation {
                    at_ms,
                    speech_probability,
                    completion_probability: None,
                    context: context.clone()
                })
                .unwrap()
                .is_none());
        }
        assert!(engine.needs_prediction(800));
    }

    #[test]
    fn interrupt_only_while_agent_speaks_and_only_once() {
        for agent_speaking in [false, true] {
            let mut engine = engine();
            let context = Context {
                agent_speaking,
                text: "Stop!".into(),
                text_final: true,
                ..Context::default()
            };
            let mut decisions = Vec::new();
            for at_ms in [0, 100, 200] {
                if let Some(decision) = engine
                    .update(Observation {
                        at_ms,
                        speech_probability: 0.9,
                        completion_probability: None,
                        context: context.clone(),
                    })
                    .unwrap()
                {
                    decisions.push(decision);
                }
            }
            assert_eq!(decisions.len(), usize::from(agent_speaking));
            if agent_speaking {
                assert_eq!(decisions[0].kind, DecisionKind::Interrupt);
                assert_eq!(decisions[0].score, None);
            }
        }
    }

    #[test]
    fn stale_terminal_text_is_suppressed_until_changed_or_reset() {
        let mut engine = engine();
        voiced(&mut engine, Context::default());
        engine
            .update(observation(1_200, 0.0, None, "wait"))
            .unwrap()
            .unwrap();
        engine
            .update(observation(1_300, 0.9, None, "wait"))
            .unwrap();
        engine
            .update(observation(1_500, 0.0, None, "wait"))
            .unwrap();
        assert!(!engine.features(1_500).unwrap().explicit_wait);
        assert!(engine.needs_prediction(1_700));
        let end = engine
            .update(observation(1_700, 0.0, Some(0.9), "wait"))
            .unwrap()
            .unwrap();
        assert_eq!(end.reason, "model_complete");
        engine.reset();
        voiced(&mut engine, Context::default());
        assert_eq!(
            engine
                .update(observation(400, 0.0, None, "wait"))
                .unwrap()
                .unwrap()
                .kind,
            DecisionKind::Wait
        );
    }

    #[test]
    fn clearing_transcript_between_turns_permits_the_same_words_again() {
        let mut engine = engine();
        voiced(&mut engine, Context::default());
        engine
            .update(observation(1_200, 0.0, None, "wait"))
            .unwrap()
            .unwrap();
        engine.update(observation(1_250, 0.0, None, "")).unwrap();
        engine
            .update(observation(1_300, 0.9, None, "wait"))
            .unwrap();
        engine
            .update(observation(1_500, 0.0, None, "wait"))
            .unwrap();
        assert!(engine.features(1_500).unwrap().explicit_wait);
        let wait = engine
            .update(observation(1_700, 0.0, None, "wait"))
            .unwrap()
            .unwrap();
        assert_eq!(wait.kind, DecisionKind::Wait);
        assert_eq!(wait.reason, "explicit_wait_request");
    }

    #[test]
    fn configuration_and_context_are_validated() {
        assert!(TurnEngine::new(Config {
            speech_start_threshold: f32::NAN,
            ..Config::default()
        })
        .is_err());
        assert!(TurnEngine::new(Config {
            speech_end_threshold: 0.9,
            ..Config::default()
        })
        .is_err());
        assert!(TurnEngine::new(Config {
            max_silence_ms: 1,
            ..Config::default()
        })
        .is_err());
        assert!(TurnEngine::new(Config {
            min_voiced_ms: 0,
            ..Config::default()
        })
        .is_err());
        let mut engine = engine();
        let mut sample = observation(0, 0.9, None, "");
        sample.context.expected_digits = Some(0);
        assert!(engine.update(sample.clone()).is_err());
        sample.context.expected_digits = Some(129);
        assert!(engine.update(sample.clone()).is_err());
        sample.context.expected_digits = None;
        sample.context.text = "a".repeat(65_537);
        assert!(engine.update(sample).is_err());
        assert!(engine.features(0).is_none());
        assert!(serde_json::from_str::<Config>(r#"{"completion_threshhold":0.9}"#).is_err());
        assert!(serde_json::from_str::<Context>(r#"{"text_fnal":true}"#).is_err());
        assert!(serde_json::from_str::<Observation>(
            r#"{"at_ms":0,"speech_probability":0.0,"contextt":{}}"#
        )
        .is_err());
    }
}
