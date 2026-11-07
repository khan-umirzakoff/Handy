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

    /// WER through the app's own decoding (profile options, then the Uzbek
    /// cleanup) on FLEURS uz dev, the set `tools/models/eval_fleurs_uz.py`
    /// scores with the CLI. Raw audio, no VAD. Run by hand:
    /// SHIVIR_FLEURS_DIR=<dir with dev.tsv, dev/> SHIVIR_MODELS=<a.gguf,b.gguf>
    /// cargo test --lib fleurs_app_path -- --ignored --nocapture
    #[test]
    #[ignore]
    fn fleurs_app_path() {
        use std::path::PathBuf;
        let dir = PathBuf::from(std::env::var("SHIVIR_FLEURS_DIR").expect("SHIVIR_FLEURS_DIR"));
        let n: usize = std::env::var("SHIVIR_N").map_or(100, |v| v.parse().unwrap());
        let tsv = std::fs::read_to_string(dir.join("dev.tsv")).unwrap();
        let mut rows: Vec<(String, String)> = tsv
            .lines()
            .map(|l| l.split('\t').collect::<Vec<_>>())
            .filter(|c| dir.join("dev").join(c[1]).exists())
            .map(|c| (c[1].to_string(), c[3].to_string()))
            .collect();
        rows.sort();
        rows.truncate(n);
        let norm = |s: &str| -> Vec<String> {
            s.to_lowercase()
                .chars()
                .map(|c| match c {
                    '\'' | '\u{2018}' | '\u{2019}' | '`' | '\u{b4}' | '\u{2bb}' | '\u{2bc}' => '\'',
                    c if c.is_alphanumeric() || c.is_whitespace() => c,
                    _ => ' ',
                })
                .collect::<String>()
                .split_whitespace()
                .map(str::to_string)
                .collect()
        };
        fn edits<T: PartialEq>(a: &[T], b: &[T]) -> usize {
            let mut prev: Vec<usize> = (0..=b.len()).collect();
            for (i, x) in a.iter().enumerate() {
                let mut cur = vec![i + 1];
                for (j, y) in b.iter().enumerate() {
                    cur.push(
                        (prev[j + 1] + 1)
                            .min(cur[j] + 1)
                            .min(prev[j] + usize::from(x != y)),
                    );
                }
                prev = cur;
            }
            prev[b.len()]
        }
        for model_path in std::env::var("SHIVIR_MODELS")
            .expect("SHIVIR_MODELS")
            .split(',')
        {
            let model = transcribe_cpp::Model::load(model_path).unwrap();
            let profile = for_variant(&model.variant()).expect("an Uzbek profile");
            let options = transcribe_cpp::RunOptions {
                language: Some(profile.language.to_string()),
                timestamps: profile.timestamps,
                family: Some(profile.run_extension()),
                ..Default::default()
            };
            let mut session = model.session().unwrap();
            let (mut we, mut wn, mut ce, mut cn) = (0, 0, 0, 0);
            let started = std::time::Instant::now();
            for (file, reference) in &rows {
                let mut wav = hound::WavReader::open(dir.join("dev").join(file)).unwrap();
                assert_eq!(wav.spec().sample_rate, 16_000);
                let pcm: Vec<f32> = match wav.spec().sample_format {
                    hound::SampleFormat::Float => {
                        wav.samples::<f32>().map(Result::unwrap).collect()
                    }
                    hound::SampleFormat::Int => wav
                        .samples::<i32>()
                        .map(|s| s.unwrap() as f32 / 32768.0)
                        .collect(),
                };
                let text =
                    crate::uzbek_text::clean(session.run(&pcm, &options).unwrap().text.trim());
                let (r, h) = (norm(reference), norm(&text));
                we += edits(&r, &h);
                wn += r.len();
                let (rc, hc): (Vec<char>, Vec<char>) =
                    (r.join(" ").chars().collect(), h.join(" ").chars().collect());
                ce += edits(&rc, &hc);
                cn += rc.len();
            }
            println!(
                "{model_path}: WER {:.2} % CER {:.2} % ({} clips, {:.0} s)",
                100.0 * we as f64 / wn as f64,
                100.0 * ce as f64 / cn as f64,
                rows.len(),
                started.elapsed().as_secs_f64()
            );
        }
    }
}
