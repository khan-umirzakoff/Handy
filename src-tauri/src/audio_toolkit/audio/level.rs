//! The bar's mic level, computed the way Wispr Flow's bar gets it
//! (research/pill-smoothness.md §2; `.re/flow-desktop/pretty/main/index.js:164197-164234`
//! and `:251330-251357`): chunk RMS in dBFS over an adaptive floor (the lowest
//! dB seen this take, never below −60 dB), 20 dB to full scale, averaged over
//! 150 ms. Silence sits at 0 and normal speech reaches 1 whatever the mic gain.

use crate::audio_toolkit::constants::WHISPER_SAMPLE_RATE;

const FLOOR_MIN_DB: f32 = -60.0;
const RANGE_DB: f32 = 20.0;
const REPORT_SAMPLES: usize = WHISPER_SAMPLE_RATE as usize * 150 / 1000;

/// One per take: a new meter starts with the floor at 0 dB, as Flow's does.
pub struct LevelMeter {
    floor_db: f32,
    sum: f32,
    chunks: usize,
    samples: usize,
}

impl LevelMeter {
    pub fn new() -> Self {
        Self {
            floor_db: 0.0,
            sum: 0.0,
            chunks: 0,
            samples: 0,
        }
    }

    /// Feeds one chunk of 16 kHz mono audio; returns the mean level (0..1)
    /// once 150 ms of audio has gone in.
    pub fn feed(&mut self, chunk: &[f32]) -> Option<f32> {
        if chunk.is_empty() {
            return None;
        }
        let rms = (chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32).sqrt();
        let db = 20.0 * (rms + 1e-10).log10();
        if db < self.floor_db {
            self.floor_db = db.max(FLOOR_MIN_DB);
        }
        self.sum += ((db - self.floor_db) / RANGE_DB).clamp(0.0, 1.0);
        self.chunks += 1;
        self.samples += chunk.len();
        if self.samples < REPORT_SAMPLES {
            return None;
        }
        let level = self.sum / self.chunks as f32;
        self.sum = 0.0;
        self.chunks = 0;
        // Carry the overshoot, so 512-sample chunks still average 150 ms.
        self.samples -= REPORT_SAMPLES;
        Some(level)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 512-sample chunk with the given RMS in dBFS.
    fn chunk(db: f32) -> Vec<f32> {
        let amp = 10f32.powf(db / 20.0) * std::f32::consts::SQRT_2;
        (0..512).map(|i| amp * (i as f32 * 0.3).sin()).collect()
    }

    /// Feeds 150 ms (5 chunks of 512) and returns the report.
    fn report(m: &mut LevelMeter, db: f32) -> f32 {
        (0..5).filter_map(|_| m.feed(&chunk(db))).last().unwrap()
    }

    #[test]
    fn silence_is_zero_and_speech_20_db_up_is_full() {
        let mut m = LevelMeter::new();
        assert!(report(&mut m, -50.0) < 0.01);
        assert!((report(&mut m, -30.0) - 1.0).abs() < 0.02);
        assert!((report(&mut m, -40.0) - 0.5).abs() < 0.02);
    }

    #[test]
    fn floor_stops_at_minus_60_db() {
        let mut m = LevelMeter::new();
        let muted = vec![0.0; 512];
        for _ in 0..5 {
            m.feed(&muted);
        }
        assert!((report(&mut m, -50.0) - 0.5).abs() < 0.02);
    }

    #[test]
    fn reports_every_150_ms() {
        let mut m = LevelMeter::new();
        // 75 × 512 samples = 2.4 s = 16 × 150 ms.
        let reports = (0..75).filter_map(|_| m.feed(&chunk(-40.0))).count();
        assert_eq!(reports, 16);
    }
}
