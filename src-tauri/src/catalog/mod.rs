//! The bundled, offline model catalog.
//!
//! `catalog.json` is a verbatim copy of the `@handy-computer/models` package's
//! catalog (the version pinned in `package.json`; `bun run check:catalog`
//! keeps the two identical). It lists every model transcribe.cpp publishes,
//! with pinned revisions, per-quant hashes and benchmark results, and marks
//! the curated subset. It is compiled into the binary so Handy ships a
//! complete model list with zero network access.
//!
//! `shivir.json` adds Shivir's own models (the Uzbek fine-tunes) in the same
//! shape; it is appended at load here and in `src/lib/models/catalog.ts`.
//! Handy's mirror does not host them.
//!
//! Every supported transcription model is offered for download and mirrored
//! at the package's default quant, independently of recommendation ranking.
//!
//! Each entry is normalised into a [`ModelDescriptor`] — the same source-agnostic
//! shape every other producer (HF discovery, on-disk scans, the legacy table)
//! yields — so the catalog is "just another producer". Its explicit capabilities
//! become a [`CapabilityProbe`] with confident `Some(..)` values; the runtime
//! `GgufHeaderProber` is the same shape with `None` where a header omits a key,
//! which is why the two are interchangeable (the catalog is a baked probe).

use std::collections::{BTreeSet, HashMap, HashSet};

use once_cell::sync::Lazy;
use serde::Deserialize;

use crate::managers::model::{default_quant_file, ModelDescriptor, ModelSource, QuantFile};
use crate::managers::model_capabilities::{CapabilityProbe, Compatibility};

/// Catalog schema this loader understands. The package bumps it on any
/// breaking change; additive fields are ignored.
const SCHEMA: u32 = 2;

/// Base URLs tried in order when the Hugging Face download fails. The full
/// file URL is `{mirror}/{repo_id}/{revision}/{filename}` — the same three
/// values that form the HF resolve URL, so a mirror is a plain static host.
/// `scripts/mirror_models.py` populates it from this catalog.
const MIRRORS: &[&str] = &["https://blob.handy.computer"];

/// The list order and onboarding set, keyed by catalog id: Shivir's Uzbek
/// models first (an Uzbek-first app; Zehnova is unscored, so not badged), then
/// Handy's order from its previous generator.
/// `(id, rank, recommended)`: rank is the editorial sort position (lower =
/// higher); recommended is the small badged onboarding subset.
const HANDY_RANKING: &[(&str, u32, bool)] = &[
    ("rubaistt-v2-medium", 1, true),
    ("kotib-uzbek-stt-v1", 2, true),
    ("navai-whisper-medium-uzbek", 3, true),
    ("zehnova-uzbek-stt", 4, false),
    ("parakeet-unified-en-0.6b", 5, true),
    ("nemotron-3.5-asr-streaming-0.6b", 6, true),
    ("canary-180m-flash", 7, true),
    ("cohere-transcribe-03-2026", 8, true),
    ("whisper-medium", 9, true),
    ("Voxtral-Mini-4B-Realtime-2602", 10, false),
    ("parakeet-tdt-0.6b-v3", 11, false),
    ("parakeet-tdt-0.6b-v2", 12, false),
    ("Qwen3-ASR-0.6B", 13, false),
    ("Fun-ASR-MLT-Nano-2512", 14, false),
];

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CatalogRoot {
    schema: u32,
    /// The transcribe.cpp release the catalog was built from.
    #[cfg_attr(not(test), allow(dead_code))]
    transcribe_cpp: String,
    models: Vec<CatalogModel>,
    /// Catalog id of the spoken-language detector for transcription models
    /// without their own language detection.
    #[serde(default)]
    language_detector: Option<String>,
    /// Benchmark results keyed by catalog id.
    #[serde(default)]
    results: HashMap<String, ModelResults>,
}

/// One model as written in `catalog.json`. Only the fields Handy needs are
/// declared; serde ignores the rest (license, parameters, …).
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CatalogModel {
    /// Stable package id: the HF repo name without `-gguf`.
    id: String,
    name: String,
    /// Editorial copy; empty for models that aren't curated.
    description: String,
    /// HF repo id, e.g. `handy-computer/whisper-small-gguf`.
    repository: String,
    /// Commit the files and hashes are pinned to. Both HF acquisition and
    /// mirror keys use it, so downloaded bytes provably match the hashes.
    revision: String,
    /// The loader architecture (`general.architecture`).
    family: String,
    languages: Vec<String>,
    capabilities: CatalogCaps,
    /// Key of the default file in `files`.
    quant: String,
    /// `asr`, `langid`, `vad` or `diarize`; only `asr` models transcribe.
    role: String,
    /// Oldest transcribe.cpp release that can load these files.
    #[cfg_attr(not(test), allow(dead_code))]
    min_engine: Option<String>,
    /// Every published file, keyed by quant.
    files: HashMap<String, CatalogFile>,
    /// From `shivir.json`: our own repos, not on Handy's mirror.
    #[serde(skip)]
    shivir: bool,
    /// Not uploaded yet (`shivir.json`): listed only while its file is on
    /// disk, never offered for download.
    #[serde(default)]
    unpublished: bool,
}

/// `shivir.json`: Shivir's models and results in the catalog's own shapes.
#[derive(Deserialize)]
struct ShivirCatalog {
    models: Vec<CatalogModel>,
    results: HashMap<String, ModelResults>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CatalogCaps {
    streaming: bool,
    translate: bool,
    language_detection: bool,
    #[serde(default)]
    timestamps: bool,
}

#[derive(Deserialize)]
struct CatalogFile {
    filename: String,
    size: u64,
    sha256: String,
}

#[derive(Deserialize, Default)]
struct ModelResults {
    /// Error percent per language identity, measured at the default quant.
    #[serde(default)]
    accuracy: HashMap<String, AccuracyCell>,
    #[serde(default)]
    performance: Performance,
}

#[derive(Deserialize)]
struct AccuracyCell {
    error: f64,
}

#[derive(Deserialize, Default)]
struct Performance {
    /// Real-time factor on the benchmark GPU (Vulkan).
    accelerated: Option<f64>,
    cpu: Option<f64>,
}

impl CatalogModel {
    fn default_file(&self) -> Option<&CatalogFile> {
        self.files.get(&self.quant)
    }

    /// Handy uses a flat models directory for mirror downloads, so only
    /// transcription models with root-level default files can be offered.
    /// Must match `offered_models` in `scripts/mirror_models.py`.
    fn is_offered(&self) -> bool {
        self.role == "asr"
            && self
                .default_file()
                .is_some_and(|file| !file.filename.contains('/'))
    }

    /// Files ordered largest first, so the list reads F32 → Q4 like the HF repo.
    fn quant_files(&self) -> Vec<QuantFile> {
        let mut files: Vec<QuantFile> = self
            .files
            .iter()
            .map(|(quant, file)| QuantFile {
                filename: file.filename.clone(),
                quant: quant.clone(),
                size_bytes: file.size,
            })
            .collect();
        files.sort_by_key(|f| std::cmp::Reverse(f.size_bytes));
        files
    }
}

// ───────────────────────── scores ─────────────────────────
// The list's 0–100 bars, from the package's benchmarks with the formulas the
// previous generator used.

/// speed = 100·(1 − e^(−rtf/8)), growing toward 100.
const SPEED_SCALE: f64 = 8.0;
/// accuracy = 100·e^(−error/15), decaying from 100.
const ACCURACY_SCALE: f64 = 15.0;
/// No benchmark speed (too heavy to measure): assume slow.
const SPEED_FLOOR: f64 = 10.0;

fn speed_score(results: Option<&ModelResults>) -> f32 {
    let rtf = results.and_then(|r| r.performance.accelerated.or(r.performance.cpu));
    let score = match rtf {
        Some(rtf) if rtf > 0.0 => 100.0 * (1.0 - (-rtf / SPEED_SCALE).exp()),
        _ => SPEED_FLOOR,
    };
    score.round() as f32
}

/// Accuracy on English when measured, else on the model's first language.
fn accuracy_score(model: &CatalogModel, results: Option<&ModelResults>) -> f32 {
    let Some(results) = results else {
        return 0.0;
    };
    let headline = std::iter::once("en".to_string())
        .chain(model.languages.first().map(|l| language_identity(l)))
        .find_map(|language| results.accuracy.get(&language));
    headline
        .map(|cell| (100.0 * (-cell.error / ACCURACY_SCALE).exp()).round() as f32)
        .unwrap_or(0.0)
}

/// The package's language identity (its `languageIdentity`): base code, with
/// the aliases its benchmark keys use.
pub(crate) fn language_identity(language: &str) -> String {
    crate::managers::model::canonical_language_code(&language.to_ascii_lowercase()).to_string()
}

// ───────────────────────── descriptions ─────────────────────────

/// A factual one-liner for models the package has no editorial copy for.
fn generated_description(model: &CatalogModel) -> String {
    let mut features = Vec::new();
    if model.capabilities.translate {
        features.push("translation");
    }
    if model.capabilities.language_detection {
        features.push("auto language detection");
    }
    if model.capabilities.streaming {
        features.push("streaming");
    }
    if model.capabilities.timestamps {
        features.push("timestamps");
    }

    let languages: BTreeSet<String> = model
        .languages
        .iter()
        .map(|l| language_identity(l))
        .collect();
    let base = if languages.len() > 1 {
        format!("{}-language speech-to-text", languages.len())
    } else {
        let name = match languages.iter().next().map(String::as_str) {
            None | Some("en") => "English",
            Some("ar") => "Arabic",
            Some("ja") => "Japanese",
            Some("ko") => "Korean",
            Some("ru") => "Russian",
            Some("uk") => "Ukrainian",
            Some("vi") => "Vietnamese",
            Some("zh") => "Chinese",
            Some(other) => other,
        };
        format!("{name} speech-to-text")
    };

    if features.is_empty() {
        format!("{base}.")
    } else {
        format!("{base} with {}.", features.join(", "))
    }
}

impl From<&CatalogModel> for ModelDescriptor {
    fn from(m: &CatalogModel) -> Self {
        // The default download file. Its name is folded into the id so a catalog
        // entry collides (dedups) with the very same file later discovered in
        // the HF cache — both compute `"{repo_id}/{filename}"`.
        let files = m.quant_files();
        let default_filename = default_quant_file(&files, Some(&m.quant))
            .map(|f| f.filename.clone())
            .unwrap_or_default();
        let ranking = HANDY_RANKING.iter().find(|(id, _, _)| *id == m.id);
        let results = ROOT.results.get(&m.id);

        ModelDescriptor {
            id: format!("{}/{}", m.repository, default_filename),
            source: ModelSource::HuggingFace {
                repo_id: m.repository.clone(),
                // Acquire at the pin: `resolve/<sha>` is immutable (CDN-friendly)
                // and guarantees the bytes match the catalog's hashes.
                revision: m.revision.clone(),
            },
            name: m.name.clone(),
            description: if m.description.is_empty() {
                generated_description(m)
            } else {
                m.description.clone()
            },
            caps: CapabilityProbe {
                verdict: Compatibility::Compatible, // published by transcribe.cpp
                display_name: None,
                architecture: Some(m.family.clone()),
                variant: None,
                languages: Some(m.languages.clone()),
                supports_streaming: Some(m.capabilities.streaming),
                supports_translation: Some(m.capabilities.translate),
                supports_language_detect: Some(m.capabilities.language_detection),
            },
            files,
            default_quant: Some(m.quant.clone()),
            // ModelInfo / the UI bars use 0.0–1.0.
            speed_score: speed_score(results) / 100.0,
            accuracy_score: accuracy_score(m, results) / 100.0,
            recommended_rank: ranking.map(|(_, rank, _)| *rank),
            recommended: ranking.is_some_and(|(_, _, recommended)| *recommended),
        }
    }
}

/// The raw parsed catalog. Kept alive (not consumed) so data that deliberately
/// stays out of [`ModelDescriptor`] (mirrors, results) can
/// be looked up separately.
static ROOT: Lazy<CatalogRoot> = Lazy::new(|| {
    let mut root: CatalogRoot = serde_json::from_str(include_str!("catalog.json"))
        .expect("bundled catalog.json is valid JSON matching the catalog schema");
    assert_eq!(
        root.schema, SCHEMA,
        "bundled catalog.json has an unsupported schema"
    );
    let shivir: ShivirCatalog = serde_json::from_str(include_str!("shivir.json"))
        .expect("bundled shivir.json is valid JSON matching the catalog schema");
    for mut model in shivir.models {
        assert!(
            root.models.iter().all(|m| m.id != model.id),
            "shivir.json reuses catalog id {}",
            model.id
        );
        model.shivir = true;
        root.models.push(model);
    }
    root.results.extend(shivir.results);
    root
});

/// Transcription models: the only role Handy can load. Language ID, VAD and
/// diarization models are published alongside them but produce no text.
fn transcription_models() -> impl Iterator<Item = &'static CatalogModel> {
    ROOT.models.iter().filter(|m| m.role == "asr")
}

/// Every language a transcription model in the catalog hears, as identities:
/// the languages the frontend's picker offers (its `LANGUAGES`).
static LANGUAGES: Lazy<HashSet<String>> = Lazy::new(|| {
    transcription_models()
        .flat_map(|m| m.languages.iter().map(|l| language_identity(l)))
        .collect()
});

/// Whether the picker offers this language identity.
pub fn offers_language(identity: &str) -> bool {
    LANGUAGES.contains(identity)
}

/// Every transcription model Handy can download, normalised into descriptors.
pub static CATALOG: Lazy<Vec<ModelDescriptor>> = Lazy::new(|| {
    transcription_models()
        .filter(|m| m.is_offered())
        .map(ModelDescriptor::from)
        .collect()
});

/// The spoken-language detector the catalog names (`languageDetector`): the
/// file Handy fetches, at the package's default quant, and the labels it can
/// tell apart.
pub struct LanguageDetectorFile {
    pub repo_id: String,
    pub revision: String,
    pub filename: String,
    pub size_bytes: u64,
    sha256: String,
    /// The detector's own codes (`iw`, `jw`, `tl`, `no` included).
    pub labels: Vec<String>,
}

pub static LANGUAGE_DETECTOR: Lazy<Option<LanguageDetectorFile>> = Lazy::new(|| {
    let id = ROOT.language_detector.as_deref()?;
    let model = ROOT
        .models
        .iter()
        .find(|m| m.id == id && m.role == "langid")?;
    let file = model.default_file()?;
    Some(LanguageDetectorFile {
        repo_id: model.repository.clone(),
        revision: model.revision.clone(),
        filename: file.filename.clone(),
        size_bytes: file.size,
        sha256: file.sha256.clone(),
        labels: model.languages.clone(),
    })
});

/// A mirror copy of a catalog model's file, with the expected content hash for
/// end-to-end verification. Mirrors are untrusted bit-pipes: the sha256 here
/// (from the catalog compiled into the binary) is the trust anchor.
pub struct MirrorFile {
    pub url: String,
    pub sha256: String,
    /// Catalog size — drives progress totals and resume sanity checks.
    pub size_bytes: u64,
}

/// Ordered mirror URLs for an offered catalog model's file — any listed quant,
/// not just the default — or empty when the model isn't offered from the
/// catalog. `model_id` is the registry id (`"{repo_id}/{filename}"`). (The
/// mirror only hosts default quants; a miss there just 404s and the caller
/// reports it, so listing every quant here costs nothing.)
pub fn mirror_fallbacks(model_id: &str) -> Vec<MirrorFile> {
    let Some((m, file)) = transcription_models()
        .filter(|m| m.is_offered() && !m.shivir)
        .find_map(|m| {
            m.files
                .values()
                .find(|f| format!("{}/{}", m.repository, f.filename) == model_id)
                .map(|f| (m, f))
        })
    else {
        return Vec::new();
    };
    mirror_files(
        &m.repository,
        &m.revision,
        &file.filename,
        &file.sha256,
        file.size,
    )
}

/// Ordered mirror URLs for the language detector's file.
pub fn language_detector_mirrors() -> Vec<MirrorFile> {
    LANGUAGE_DETECTOR.as_ref().map_or_else(Vec::new, |d| {
        mirror_files(
            &d.repo_id,
            &d.revision,
            &d.filename,
            &d.sha256,
            d.size_bytes,
        )
    })
}

fn mirror_files(
    repository: &str,
    revision: &str,
    filename: &str,
    sha256: &str,
    size_bytes: u64,
) -> Vec<MirrorFile> {
    MIRRORS
        .iter()
        .map(|base| MirrorFile {
            url: format!(
                "{}/{}/{}/{}",
                base.trim_end_matches('/'),
                repository,
                revision,
                filename
            ),
            sha256: sha256.to_string(),
            size_bytes,
        })
        .collect()
}

/// The catalog descriptor + specific `files[]` entry owning `filename`,
/// matched across every listed quant (not just the default) of every offered
/// transcription model. `repo_id`, when given, must also match — the HF-cache
/// scan uses it to keep a foreign repo that happens to reuse a catalog
/// filename from masquerading as ours.
pub fn file_in_catalog(
    filename: &str,
    repo_id: Option<&str>,
) -> Option<(&'static ModelDescriptor, &'static QuantFile)> {
    let catalog: &'static Vec<ModelDescriptor> = Lazy::force(&CATALOG);
    catalog.iter().find_map(|d| {
        if let Some(repo) = repo_id {
            match &d.source {
                ModelSource::HuggingFace { repo_id: r, .. } if r == repo => {}
                _ => return None,
            }
        }
        d.files
            .iter()
            .find(|f| f.filename == filename)
            .map(|f| (d, f))
    })
}

/// Whether a catalog repo has no uploaded files yet, so its entry may only
/// stand for a file already on disk.
pub fn is_unpublished(repo_id: &str) -> bool {
    ROOT.models
        .iter()
        .any(|m| m.unpublished && m.repository == repo_id)
}

/// Editorial recommended rank keyed by descriptor id (the same id the model
/// registry uses). Built once from the catalog.
static RANK_BY_ID: Lazy<HashMap<String, u32>> = Lazy::new(|| {
    CATALOG
        .iter()
        .filter_map(|d| d.recommended_rank.map(|r| (d.id.clone(), r)))
        .collect()
});

/// Recommended rank for a model id (lower = higher priority). Returns
/// `u32::MAX` for unranked/unknown ids so they sort last in an ascending sort.
pub fn rank_of(model_id: &str) -> u32 {
    RANK_BY_ID.get(model_id).copied().unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::managers::model_capabilities::KNOWN_ARCHES;

    #[test]
    fn catalog_parses_and_offers_models() {
        assert!(!CATALOG.is_empty(), "bundled catalog should offer models");
    }

    #[test]
    fn ids_are_unique() {
        let mut ids: Vec<&str> = CATALOG.iter().map(|d| d.id.as_str()).collect();
        ids.sort_unstable();
        let before = ids.len();
        ids.dedup();
        assert_eq!(before, ids.len(), "catalog descriptor ids must be unique");
    }

    #[test]
    fn names_a_language_detector() {
        let detector = LANGUAGE_DETECTOR
            .as_ref()
            .expect("catalog names a language detector");
        assert!(detector.filename.ends_with(".gguf"));
        assert!(detector.labels.iter().any(|l| l == "en"));
        assert!(detector.labels.iter().any(|l| l == "iw"));
    }

    #[test]
    fn only_transcription_models_are_loaded() {
        // Language ID (ecapa_tdnn), VAD (silero_vad) and diarization
        // (sortformer) models are published but produce no transcription text.
        for arch in ["ecapa_tdnn", "silero_vad", "sortformer"] {
            assert!(
                CATALOG
                    .iter()
                    .all(|d| d.caps.architecture.as_deref() != Some(arch)),
                "{arch} must not be in the transcription catalog"
            );
        }
    }

    #[test]
    fn all_flat_asr_models_are_offered_at_the_package_default() {
        for m in transcription_models()
            .filter(|m| m.default_file().is_some_and(|f| !f.filename.contains('/')))
        {
            let id = format!("{}/{}", m.repository, m.default_file().unwrap().filename);
            let desc = CATALOG
                .iter()
                .find(|d| d.id == id)
                .expect("ASR model is offered");
            assert_eq!(desc.default_quant.as_deref(), Some(m.quant.as_str()));
            assert_eq!(m.quant, "Q8_0");
        }
    }

    #[test]
    fn scores_are_normalised_0_to_1() {
        for d in CATALOG.iter() {
            assert!((0.0..=1.0).contains(&d.speed_score), "{} speed", d.id);
            assert!((0.0..=1.0).contains(&d.accuracy_score), "{} acc", d.id);
        }
    }

    #[test]
    fn every_description_is_filled() {
        for d in CATALOG.iter() {
            assert!(!d.description.is_empty(), "{}: empty description", d.id);
        }
    }

    #[test]
    fn handy_ranking_names_offered_models() {
        for (id, _, _) in HANDY_RANKING {
            let model = transcription_models()
                .find(|m| m.id == *id)
                .unwrap_or_else(|| panic!("ranked model {id} is not in the catalog"));
            assert!(model.is_offered(), "ranked model {id} is not offered");
        }
        assert_eq!(
            CATALOG.iter().filter(|d| d.recommended).count(),
            HANDY_RANKING.iter().filter(|(_, _, r)| *r).count()
        );
    }

    #[test]
    fn every_offered_model_has_mirror_fallbacks_with_hashes() {
        // The mirror fallback is the safety net for HF outages and blocked
        // networks; an offered entry without one silently loses that net.
        let ours: HashSet<String> = transcription_models()
            .filter(|m| m.shivir)
            .filter_map(|m| Some(format!("{}/{}", m.repository, m.default_file()?.filename)))
            .collect();
        for d in CATALOG.iter() {
            let mirrors = mirror_fallbacks(&d.id);
            if ours.contains(&d.id) {
                assert!(
                    mirrors.is_empty(),
                    "{}: Handy's mirror lacks our files",
                    d.id
                );
                continue;
            }
            assert!(!mirrors.is_empty(), "{}: no mirror fallbacks", d.id);
            for m in &mirrors {
                assert_eq!(m.sha256.len(), 64, "{}: mirror entry lacks a sha256", d.id);
                assert!(m.size_bytes > 0, "{}: mirror entry lacks a size", d.id);
                assert!(m.url.starts_with("https://"), "{}: bad url {}", d.id, m.url);
            }
        }
    }

    #[test]
    fn catalog_matches_the_engine_handy_ships() {
        // Every model must be loadable by the transcribe-cpp crate we link.
        let cargo = include_str!("../../Cargo.toml");
        let engine = cargo
            .lines()
            .find_map(|line| {
                line.strip_prefix("transcribe-cpp = { version = \"")
                    .and_then(|rest| rest.split('"').next())
            })
            .expect("Cargo.toml pins a transcribe-cpp version");
        let parse = |v: &str| -> Vec<u32> {
            v.trim_start_matches('v')
                .split(['.', '-', '+'])
                .take(3)
                .map(|p| p.parse().unwrap_or(0))
                .collect()
        };
        assert!(
            parse(&ROOT.transcribe_cpp) <= parse(engine),
            "catalog built for transcribe.cpp {} but Handy links {}",
            ROOT.transcribe_cpp,
            engine
        );
        for m in transcription_models() {
            if let Some(min) = &m.min_engine {
                assert!(
                    parse(min) <= parse(engine),
                    "{} needs transcribe.cpp {min}, Handy links {engine}",
                    m.id
                );
            }
        }
    }

    #[test]
    fn catalog_architectures_are_known_to_capability_probe() {
        let missing: BTreeSet<&str> = CATALOG
            .iter()
            .filter_map(|d| d.caps.architecture.as_deref())
            .filter(|arch| !KNOWN_ARCHES.contains(arch))
            .collect();

        assert!(
            missing.is_empty(),
            "catalog architecture(s) missing from KNOWN_ARCHES: {:?}",
            missing
        );
    }

    #[test]
    fn shivir_uzbek_models_join_the_catalog_first() {
        let (desc, file) = file_in_catalog("kotib-uzbek-stt-v1-Q8_0.gguf", None)
            .expect("a local Kotib file is recognised as the catalog model");
        assert_eq!(
            desc.id,
            "shivir/kotib-uzbek-stt-v1-gguf/kotib-uzbek-stt-v1-Q8_0.gguf"
        );
        assert_eq!(file.quant, "Q8_0");
        assert_eq!(
            desc.caps.languages.as_deref(),
            Some(&["uz".to_string()][..])
        );
        assert!(desc.recommended);
        assert!(offers_language("uz"));
        // Placeholder repos until the upload: never offered for download.
        assert!(is_unpublished("shivir/kotib-uzbek-stt-v1-gguf"));
        assert!(!is_unpublished("handy-computer/whisper-medium-gguf"));
        for id in [
            "rubaistt-v2-medium",
            "kotib-uzbek-stt-v1",
            "navai-whisper-medium-uzbek",
            "zehnova-uzbek-stt",
        ] {
            let m = transcription_models().find(|m| m.id == id).unwrap();
            assert!(m.shivir && m.is_offered(), "{id}");
            assert!(ROOT.results.contains_key(id), "{id}: no results entry");
        }
        // Uzbek leads the list.
        assert!(
            rank_of(&desc.id)
                < rank_of("handy-computer/whisper-medium-gguf/whisper-medium-Q8_0.gguf")
        );
    }

    #[test]
    fn generated_descriptions_read_naturally() {
        let gigaam = transcription_models()
            .find(|m| m.id == "gigaam-v3-ctc")
            .unwrap();
        assert!(generated_description(gigaam).starts_with("Russian speech-to-text"));
        let whisper = transcription_models()
            .find(|m| m.id == "whisper-tiny")
            .unwrap();
        assert!(generated_description(whisper).contains("-language speech-to-text with"));
    }
}
