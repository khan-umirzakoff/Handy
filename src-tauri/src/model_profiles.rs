//! Per-model decoding profiles for Shivir's Uzbek fine-tunes.
//!
//! The models are stock Whisper medium checkpoints fine-tuned on Uzbek. They
//! need no retraining, only the right decoding: Uzbek forced (their leftover
//! config forces English and auto-detect drifts), and Kotib's tested
//! thresholds, greedy without temperature fallback. Values and evidence:
//! research/uzbek-models.md ("Decoding defaults differ per model") and
//! research/codex-codebases.md (Kotib: no-speech 0.25, logprob -1.0, fallback
//! step 0.2; we keep the thresholds but drop the fallback, see below). Keyed by the GGUF `stt.variant` our converter writes
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
    /// Silence put before each take. VAD hands over speech that starts about
    /// 0.4 s in; NavAI then dropped the take's first word and added words to
    /// one-word takes ("baʼzida" -> "baʼzida uchta"); with 1 s of lead-in it
    /// wrote all three of the owner's takes right (300 ms was not enough).
    pub lead_in_ms: u32,
    /// Scale each take so its peak reaches 0.95 (at most ×40), as Kotib does
    /// before every transcription (`references/kotib/src/audio_util.swift:54-63`,
    /// called at `src/ilova.swift:379`): quiet or distant speech gets through.
    pub normalize_peak: bool,
    pub whisper: WhisperRunOptions,
}

/// Kotib's peak normalisation: gain `min(0.95 / peak, 40)`; a take whose peak
/// is at most 0.0001 (silence) is left alone.
pub fn normalize_peak(samples: &mut [f32]) {
    let peak = samples.iter().fold(0.0f32, |p, s| p.max(s.abs()));
    if peak <= 0.0001 {
        return;
    }
    let gain = (0.95 / peak).min(40.0);
    samples.iter_mut().for_each(|s| *s *= gain);
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
        lead_in_ms: 1000,
        normalize_peak: true,
        whisper: WhisperRunOptions {
            temperature: Some(0.0),
            // No temperature fallback: without Kotib's beam search it samples
            // junk onto short takes (a 2 s "performance" came back as "fre
            // bounds, performance"), and it changed nothing on FLEURS.
            temperature_inc: Some(0.0),
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
        assert_eq!(p.whisper.temperature_inc, Some(0.0));
        assert_eq!(p.whisper.no_speech_thold, Some(0.25));
        assert!(matches!(p.timestamps, TimestampKind::None));
        assert!(p.normalize_peak);
        assert!(for_variant("whisper-medium").is_none());
    }

    #[test]
    fn peak_normalisation_matches_kotib() {
        let mut loud = vec![0.5, -0.25];
        normalize_peak(&mut loud);
        assert_eq!(loud, vec![0.95, -0.475]);
        let mut quiet = vec![0.001, -0.0005];
        normalize_peak(&mut quiet);
        // Capped at x40, not scaled to 0.95.
        assert!((quiet[0] - 0.04).abs() < 1e-6 && (quiet[1] + 0.02).abs() < 1e-6);
        let mut silent = vec![0.00005, 0.0];
        normalize_peak(&mut silent);
        assert_eq!(silent, vec![0.00005, 0.0]);
    }

    /// Replays recorded takes (the app's `recordings/*.wav`) through decoding
    /// variants, to see which knob produces a bad transcript. Run by hand:
    /// SHIVIR_CLIPS=<a.wav,b.wav> SHIVIR_MODELS=<a.gguf,b.gguf>
    /// cargo test --lib replay_clips -- --ignored --nocapture
    #[test]
    #[ignore]
    fn replay_clips() {
        let load = |path: &str| -> Vec<f32> {
            let mut wav = hound::WavReader::open(path).unwrap();
            assert_eq!(wav.spec().sample_rate, 16_000);
            match wav.spec().sample_format {
                hound::SampleFormat::Float => wav.samples::<f32>().map(Result::unwrap).collect(),
                hound::SampleFormat::Int => wav
                    .samples::<i32>()
                    .map(|s| s.unwrap() as f32 / 32768.0)
                    .collect(),
            }
        };
        let profile = for_variant("kotib-uzbek-stt-v1").unwrap();
        let plain = WhisperRunOptions::default();
        let with_fallback = WhisperRunOptions {
            temperature_inc: Some(0.2),
            ..profile.whisper.clone()
        };
        let variants: Vec<(&str, TimestampKind, WhisperRunOptions)> = vec![
            (
                "profile (app now)",
                TimestampKind::None,
                profile.whisper.clone(),
            ),
            (
                "profile, timestamps on",
                TimestampKind::Auto,
                profile.whisper.clone(),
            ),
            (
                "profile + Kotib fallback 0.2",
                TimestampKind::None,
                with_fallback,
            ),
            ("plain, timestamps on", TimestampKind::Auto, plain.clone()),
            ("plain, timestamps off", TimestampKind::None, plain),
        ];
        // Silence before each clip: the profile's lead-in, or SHIVIR_PAD_MS.
        let pad = std::env::var("SHIVIR_PAD_MS").map_or(
            for_variant("kotib-uzbek-stt-v1").unwrap().lead_in_ms as usize * 16,
            |v| v.parse::<usize>().unwrap() * 16,
        );
        let clips: Vec<(String, Vec<f32>)> = std::env::var("SHIVIR_CLIPS")
            .expect("SHIVIR_CLIPS")
            .split(',')
            .map(|p| {
                // SHIVIR_GAIN scales the take (a quieter mic); SHIVIR_NO_NORM
                // skips the profile's peak normalisation.
                let gain: f32 = std::env::var("SHIVIR_GAIN").map_or(1.0, |v| v.parse().unwrap());
                let mut take: Vec<f32> = load(p).into_iter().map(|s| s * gain).collect();
                if std::env::var("SHIVIR_NO_NORM").is_err() {
                    normalize_peak(&mut take);
                }
                let mut pcm = vec![0.0; pad];
                pcm.extend(take);
                (p.rsplit('/').next().unwrap().to_string(), pcm)
            })
            .collect();
        for model_path in std::env::var("SHIVIR_MODELS")
            .expect("SHIVIR_MODELS")
            .split(',')
        {
            let model = transcribe_cpp::Model::load(model_path).unwrap();
            let mut session = model.session().unwrap();
            println!("== {}", model_path.rsplit('/').next().unwrap());
            for (name, pcm) in &clips {
                println!("  {name} ({:.2} s)", pcm.len() as f32 / 16_000.0);
                for (label, timestamps, whisper) in &variants {
                    let options = transcribe_cpp::RunOptions {
                        language: Some("uz".to_string()),
                        timestamps: *timestamps,
                        family: Some(RunExtension::Whisper(whisper.clone())),
                        ..Default::default()
                    };
                    let text = session.run(pcm, &options).unwrap().text;
                    println!("    {label:24} {}", text.trim());
                }
            }
        }
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
                // As the app prepares a take: normalise, then the lead-in.
                let mut take: Vec<f32> = match wav.spec().sample_format {
                    hound::SampleFormat::Float => {
                        wav.samples::<f32>().map(Result::unwrap).collect()
                    }
                    hound::SampleFormat::Int => wav
                        .samples::<i32>()
                        .map(|s| s.unwrap() as f32 / 32768.0)
                        .collect(),
                };
                if profile.normalize_peak {
                    normalize_peak(&mut take);
                }
                let mut pcm = vec![0.0f32; profile.lead_in_ms as usize * 16];
                pcm.extend(take);
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
