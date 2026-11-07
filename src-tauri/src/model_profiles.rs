//! Per-model decoding profiles for Shivir's Uzbek fine-tunes.
//!
//! The models are stock Whisper medium checkpoints fine-tuned on Uzbek. They
//! need no retraining, only the right decoding: Uzbek forced (their leftover
//! config forces English and auto-detect drifts), and Kotib's tested
//! thresholds and temperature fallback. Values and evidence:
//! research/uzbek-models.md ("Decoding defaults differ per model") and
//! research/codex-codebases.md (Kotib: no-speech 0.25, logprob -1.0, fallback
//! step 0.2). Keyed by the GGUF `stt.variant` our converter writes
//! (tools/models/convert_uzbek.py).
//!
//! shortcut: transcribe-cpp 0.4 has no beam search or n-gram blocking, so
//! Kotib's beam 5 and NavAI/Zehnova's no_repeat_ngram are not applied; revisit
//! after the benchmark.

use transcribe_cpp::{RunExtension, TimestampKind, WhisperRunOptions};

pub struct ModelProfile {
    /// Language the model must always be told, whatever the user picked.
    pub language: &'static str,
    /// The fine-tunes were trained without timestamp tokens; decoding with
    /// them (the engine's Auto) drops the first word in most clips. Measured
    /// on 20 FLEURS uz clips, NavAI: 23.6 % WER with, 12.1 % without
    /// (research/uzbek-models.md, "Our FLEURS measurement").
    pub timestamps: TimestampKind,
    pub whisper: WhisperRunOptions,
}

const UZBEK_VARIANTS: &[&str] = &[
    "kotib-uzbek-stt-v1",
    "rubaistt-v2-medium",
    "navai-whisper-medium-uzbek",
    "zehnova-uzbek-stt",
];

pub fn for_variant(variant: &str) -> Option<ModelProfile> {
    UZBEK_VARIANTS.contains(&variant).then(|| ModelProfile {
        language: "uz",
        timestamps: TimestampKind::None,
        whisper: WhisperRunOptions {
            temperature: Some(0.0),
            temperature_inc: Some(0.2),
            logprob_thold: Some(-1.0),
            no_speech_thold: Some(0.25),
            ..Default::default()
        },
    })
}

impl ModelProfile {
    pub fn run_extension(&self) -> RunExtension {
        RunExtension::Whisper(self.whisper.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uzbek_fine_tunes_force_uzbek_and_kotib_thresholds() {
        let p = for_variant("kotib-uzbek-stt-v1").expect("profile");
        assert_eq!(p.language, "uz");
        assert_eq!(p.whisper.temperature_inc, Some(0.2));
        assert_eq!(p.whisper.no_speech_thold, Some(0.25));
        assert!(matches!(p.timestamps, TimestampKind::None));
        assert!(for_variant("whisper-medium").is_none());
    }
}
