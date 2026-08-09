use anyhow::{Context, Result};
use std::io::{BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

/// Persistent whisper.cpp worker configured exactly like Audio-Matnga.
///
/// Keeping this in a child process avoids symbol collisions with Handy's
/// transcribe.cpp backend, while retaining the loaded Rubai model between
/// dictations.
pub struct RubaiEngine {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl RubaiEngine {
    pub fn new(helper_path: &Path, model_path: &Path) -> Result<Self> {
        let mut child = Command::new(helper_path)
            .arg(model_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("Failed to start Rubai helper at {:?}", helper_path))?;

        let stdin = child
            .stdin
            .take()
            .context("Rubai helper stdin was not available")?;
        let stdout = child
            .stdout
            .take()
            .context("Rubai helper stdout was not available")?;

        let mut engine = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        };
        let ready = engine.read_response()?;
        if ready != "ready" {
            anyhow::bail!("Unexpected Rubai helper startup response: {ready}");
        }
        Ok(engine)
    }

    pub fn transcribe(&mut self, audio: &[f32]) -> Result<String> {
        let sample_count = u32::try_from(audio.len()).context("Rubai audio is too long")?;
        let mut payload = Vec::with_capacity(4 + audio.len() * 4);
        payload.extend_from_slice(&sample_count.to_le_bytes());
        for sample in audio {
            payload.extend_from_slice(&sample.to_le_bytes());
        }
        self.stdin
            .write_all(&payload)
            .context("Failed to send audio to Rubai helper")?;
        self.stdin
            .flush()
            .context("Failed to flush audio to Rubai helper")?;
        self.read_response()
    }

    fn read_response(&mut self) -> Result<String> {
        let status = self.read_u32()?;
        let length = self.read_u32()? as usize;
        if length > MAX_RESPONSE_BYTES {
            anyhow::bail!("Rubai helper returned an invalid response length: {length}");
        }

        let mut bytes = vec![0; length];
        self.stdout
            .read_exact(&mut bytes)
            .context("Rubai helper response ended early")?;
        let message = String::from_utf8(bytes).context("Rubai helper returned invalid UTF-8")?;
        if status != 0 {
            anyhow::bail!(message);
        }
        Ok(message)
    }

    fn read_u32(&mut self) -> Result<u32> {
        let mut bytes = [0; 4];
        self.stdout
            .read_exact(&mut bytes)
            .context("Rubai helper stopped unexpectedly")?;
        Ok(u32::from_le_bytes(bytes))
    }
}

impl Drop for RubaiEngine {
    fn drop(&mut self) {
        let _ = self.stdin.write_all(&0u32.to_le_bytes());
        let _ = self.stdin.flush();
        if self.child.wait().is_err() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}
