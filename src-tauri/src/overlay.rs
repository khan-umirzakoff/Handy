use crate::input;
use crate::notices::Notice;
use crate::settings;
use crate::settings::{OverlayPosition, OverlayStyle};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Manager, PhysicalPosition, PhysicalSize};

#[cfg(not(target_os = "macos"))]
use log::debug;

#[cfg(not(target_os = "macos"))]
use tauri::WebviewWindowBuilder;

#[cfg(target_os = "macos")]
use tauri::WebviewUrl;

#[cfg(target_os = "macos")]
use tauri_nspanel::{tauri_panel, CollectionBehavior, PanelBuilder, PanelLevel, StyleMask};

#[cfg(target_os = "linux")]
use gtk_layer_shell::{Edge, KeyboardMode, Layer, LayerShell};

#[cfg(target_os = "linux")]
use std::env;

#[cfg(target_os = "macos")]
tauri_panel! {
    panel!(RecordingOverlayPanel {
        config: {
            can_become_key_window: false,
            is_floating_panel: true
        }
    })

    panel!(FlowBarTooltipPanel {
        config: {
            can_become_key_window: false,
            is_floating_panel: true
        }
    })
}

// The persistent Flow Bar keeps its native window tight around the painted
// surface. That is important because Tauri cannot do Electron-style per-pixel
// hit testing: a large transparent window would swallow clicks. Dimensions are
// therefore state- and dock-aware and mirror the independently implemented CSS.
const BOTTOM_REST_WIDTH: f64 = 76.0;
const BOTTOM_PTT_WIDTH: f64 = 103.0;
const BOTTOM_ACTIVE_WIDTH: f64 = 132.0;
const BOTTOM_HEIGHT: f64 = 54.0;
const SIDE_REST_WIDTH: f64 = 54.0;
const SIDE_REST_HEIGHT: f64 = 76.0;
const SIDE_PTT_HEIGHT: f64 = 103.0;
const SIDE_ACTIVE_WIDTH: f64 = 54.0;
const SIDE_ACTIVE_HEIGHT: f64 = 132.0;
const SIDE_WORK_WIDTH: f64 = 54.0;
const SIDE_WORK_HEIGHT: f64 = 132.0;
const RECOVERY_WIDTH: f64 = 400.0;
const RECOVERY_HEIGHT: f64 = 190.0;
const NOTICE_WIDTH: f64 = 360.0;
const NOTICE_HEIGHT: f64 = 124.0;
const TOOLTIP_WIDTH: f64 = 176.0;
const TOOLTIP_HEIGHT: f64 = 34.0;
const BUBBLE_TOOLTIP_GAP: f64 = 4.0;
const ACTION_TOOLTIP_GAP: f64 = 10.0;
const FLOW_BAR_PADDING: f64 = 6.0;
const FLOW_BAR_THICKNESS: f64 = 30.0;

// Actual is 394x118, just a little extra
const OVERLAY_STREAM_WIDTH: f64 = 400.0;
const OVERLAY_STREAM_HEIGHT: f64 = 120.0;

/// Overlay window size (logical points) for a given UI state and dock edge.
fn overlay_dimensions(state: &str, position: OverlayPosition) -> (f64, f64) {
    if state == "paste-recovery" {
        (RECOVERY_WIDTH, RECOVERY_HEIGHT)
    } else if state == "notice" {
        (NOTICE_WIDTH, NOTICE_HEIGHT)
    } else if state == "streaming" {
        (OVERLAY_STREAM_WIDTH, OVERLAY_STREAM_HEIGHT)
    } else if matches!(position, OverlayPosition::Bottom) {
        let width = match state {
            "recording-ptt" => BOTTOM_PTT_WIDTH,
            "recording-toggle" | "transcribing" | "processing" | "error" => BOTTOM_ACTIVE_WIDTH,
            _ => BOTTOM_REST_WIDTH,
        };
        (width, BOTTOM_HEIGHT)
    } else {
        match state {
            "recording-ptt" => (SIDE_ACTIVE_WIDTH, SIDE_PTT_HEIGHT),
            "recording-toggle" => (SIDE_ACTIVE_WIDTH, SIDE_ACTIVE_HEIGHT),
            "transcribing" | "processing" | "error" => (SIDE_WORK_WIDTH, SIDE_WORK_HEIGHT),
            _ => (SIDE_REST_WIDTH, SIDE_REST_HEIGHT),
        }
    }
}

static LAST_MIC_LEVEL_EMIT: AtomicU64 = AtomicU64::new(0);
const EMIT_THROTTLE_MS: u64 = 33; // ~30 FPS

// 0 resting, 1 recording-ptt, 2 streaming, 3 transcribing, 4 processing,
// 5 error, 6 recording-toggle, 7 paste-recovery, 8 notice.
// Geometry updates can happen independently of state events (settings change,
// monitor move), so the native side retains the last requested state.
static CURRENT_OVERLAY_STATE: AtomicU64 = AtomicU64::new(0);
const FLOW_BAR_MORPH_COLLAPSE_DELAY_MS: u64 = 110;

#[cfg(target_os = "macos")]
static FLOW_BAR_HIT_TEST_STARTED: AtomicBool = AtomicBool::new(false);

#[cfg(target_os = "macos")]
// No cursor sample has been applied when the panel is created. Starting false
// makes the first sample inside the resting catch rect a real transition.
static FLOW_BAR_ACCEPTS_MOUSE: AtomicBool = AtomicBool::new(false);

#[cfg(target_os = "macos")]
const FLOW_BAR_HIT_TEST_INTERVAL_MS: u64 = 50;

#[derive(Clone, Copy, Debug, PartialEq)]
struct FlowBarHitRect {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

impl FlowBarHitRect {
    fn contains(self, x: f64, y: f64) -> bool {
        x >= self.x && y >= self.y && x < self.x + self.width && y < self.y + self.height
    }
}

/// The independently implemented counterpart of Flow's native catch rectangle.
///
/// The renderer owns the visible pill, while this native rectangle prevents the
/// transparent part of the tight NSPanel from swallowing clicks meant for the
/// app underneath. Resting uses the exact painted hot area. Once the cursor has
/// entered it, the expanded ready pill's painted hot area becomes the hysteresis
/// region. The recovered 12 px pseudo-element extends beyond our tight Tauri
/// panel, so the native approximation is clamped to the ready hot area instead
/// of latching the whole transparent panel.
fn flowbar_hit_rect(
    state: &str,
    dock: OverlayPosition,
    panel_width: f64,
    panel_height: f64,
    already_accepting_mouse: bool,
) -> FlowBarHitRect {
    if state == "paste-recovery" || state == "notice" {
        return FlowBarHitRect {
            x: 0.0,
            y: 0.0,
            width: panel_width,
            height: panel_height,
        };
    }

    let (pill_width, pill_height, inline_padding): (f64, f64, f64) = match (state, dock) {
        ("recording-ptt", OverlayPosition::Bottom) => (73.0, 30.0, 10.0),
        ("recording-toggle" | "streaming", OverlayPosition::Bottom) => (102.5, 30.0, 10.0),
        ("transcribing" | "processing", OverlayPosition::Bottom) => (98.0, 30.0, 10.0),
        ("error", OverlayPosition::Bottom) => (91.0, 30.0, 10.0),
        ("resting", OverlayPosition::Bottom) if already_accepting_mouse => (50.0, 30.0, 10.0),
        ("resting", OverlayPosition::Bottom) => (40.0, 8.0, 10.0),
        ("recording-ptt", OverlayPosition::Left | OverlayPosition::Right) => (30.0, 73.0, 6.0),
        ("recording-toggle" | "streaming", OverlayPosition::Left | OverlayPosition::Right) => {
            (30.0, 102.5, 6.0)
        }
        ("transcribing" | "processing", OverlayPosition::Left | OverlayPosition::Right) => {
            (30.0, 98.0, 6.0)
        }
        ("error", OverlayPosition::Left | OverlayPosition::Right) => (30.0, 91.0, 6.0),
        ("resting", OverlayPosition::Left | OverlayPosition::Right) if already_accepting_mouse => {
            (30.0, 50.0, 6.0)
        }
        ("resting", OverlayPosition::Left | OverlayPosition::Right) => (8.0, 40.0, 6.0),
        (_, OverlayPosition::Bottom) => (40.0, 8.0, 10.0),
        (_, OverlayPosition::Left | OverlayPosition::Right) => (8.0, 40.0, 6.0),
    };

    let width = (pill_width + 2.0 * inline_padding).min(panel_width);
    let height = (pill_height + 2.0 * FLOW_BAR_PADDING).min(panel_height);
    let x = match dock {
        OverlayPosition::Bottom => (panel_width - width) / 2.0,
        OverlayPosition::Left => 0.0,
        OverlayPosition::Right => panel_width - width,
    };
    let y = match dock {
        OverlayPosition::Bottom => panel_height - height,
        OverlayPosition::Left | OverlayPosition::Right => (panel_height - height) / 2.0,
    };

    FlowBarHitRect {
        x,
        y,
        width,
        height,
    }
}

#[cfg(target_os = "macos")]
fn start_flowbar_hit_test(app_handle: &AppHandle) {
    if FLOW_BAR_HIT_TEST_STARTED.swap(true, Ordering::AcqRel) {
        return;
    }

    let app = app_handle.clone();
    tauri::async_runtime::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(
                FLOW_BAR_HIT_TEST_INTERVAL_MS,
            ))
            .await;

            let Some(window) = app.get_webview_window("recording_overlay") else {
                FLOW_BAR_ACCEPTS_MOUSE.store(false, Ordering::Relaxed);
                continue;
            };
            if !window.is_visible().unwrap_or(false) {
                continue;
            }

            // Enigo is deliberately initialized only after onboarding. Until it
            // exists, retain normal panel mouse handling instead of making the
            // overlay unreachable.
            let Some((cursor_x, cursor_y)) = input::get_cursor_position(&app) else {
                if !FLOW_BAR_ACCEPTS_MOUSE.swap(true, Ordering::AcqRel) {
                    let _ = window.set_ignore_cursor_events(false);
                    let _ = window.emit("flowbar-native-hover", true);
                }
                continue;
            };
            let Ok(scale) = window.scale_factor() else {
                continue;
            };
            let Ok(position) = window.outer_position() else {
                continue;
            };
            let Ok(size) = window.outer_size() else {
                continue;
            };
            let position = position.to_logical::<f64>(scale);
            let size = size.to_logical::<f64>(scale);
            let local_x = cursor_x as f64 - position.x;
            let local_y = cursor_y as f64 - position.y;
            let was_accepting = FLOW_BAR_ACCEPTS_MOUSE.load(Ordering::Relaxed);
            let rect = flowbar_hit_rect(
                current_overlay_state(),
                settings::get_settings(&app).overlay_position,
                size.width,
                size.height,
                was_accepting,
            );
            let should_accept = rect.contains(local_x, local_y);

            if should_accept != was_accepting {
                FLOW_BAR_ACCEPTS_MOUSE.store(should_accept, Ordering::Relaxed);
                log::debug!(
                    "Flow Bar native hit-test: accepts_mouse={} local=({:.1}, {:.1}) rect=({:.1}, {:.1}, {:.1}, {:.1})",
                    should_accept,
                    local_x,
                    local_y,
                    rect.x,
                    rect.y,
                    rect.width,
                    rect.height
                );
                if let Err(error) = window.set_ignore_cursor_events(!should_accept) {
                    log::debug!("Failed to update Flow Bar mouse hit-test: {error}");
                }
            }

            // WKWebView does not reliably synthesize mouseenter when an
            // already-hovered click-through NSPanel starts accepting events.
            // Flow's Electron window gets forwarded mouse movement here; our
            // native bridge reports the same catch-rect state explicitly.
            // Repeating the tiny boolean heartbeat also covers renderer reloads
            // that may have missed the first transition during startup.
            let _ = window.emit("flowbar-native-hover", should_accept);
        }
    });
}

fn overlay_state_code(state: &str) -> u64 {
    match state {
        "recording-ptt" => 1,
        "streaming" => 2,
        "transcribing" => 3,
        "processing" => 4,
        "error" => 5,
        "recording-toggle" => 6,
        "paste-recovery" => 7,
        "notice" => 8,
        _ => 0,
    }
}

fn overlay_state_from_code(code: u64) -> &'static str {
    match code {
        1 => "recording-ptt",
        2 => "streaming",
        3 => "transcribing",
        4 => "processing",
        5 => "error",
        6 => "recording-toggle",
        7 => "paste-recovery",
        8 => "notice",
        _ => "resting",
    }
}

fn current_overlay_state() -> &'static str {
    overlay_state_from_code(CURRENT_OVERLAY_STATE.load(Ordering::Relaxed))
}

#[cfg(target_os = "macos")]
const OVERLAY_EDGE_OFFSET: f64 = 0.0;

#[cfg(any(target_os = "windows", target_os = "linux"))]
const OVERLAY_EDGE_OFFSET: f64 = 8.0;

#[cfg(target_os = "linux")]
fn update_gtk_layer_shell_anchors(overlay_window: &tauri::webview::WebviewWindow) {
    let window_clone = overlay_window.clone();
    let _ = overlay_window.run_on_main_thread(move || {
        // Try to get the GTK window from the Tauri webview
        if let Ok(gtk_window) = window_clone.gtk_window() {
            let settings = settings::get_settings(window_clone.app_handle());
            match settings.overlay_position {
                OverlayPosition::Bottom => {
                    gtk_window.set_anchor(Edge::Bottom, true);
                    gtk_window.set_anchor(Edge::Top, false);
                    gtk_window.set_anchor(Edge::Left, false);
                    gtk_window.set_anchor(Edge::Right, false);
                }
                OverlayPosition::Left => {
                    gtk_window.set_anchor(Edge::Left, true);
                    gtk_window.set_anchor(Edge::Right, false);
                    gtk_window.set_anchor(Edge::Top, false);
                    gtk_window.set_anchor(Edge::Bottom, false);
                }
                OverlayPosition::Right => {
                    gtk_window.set_anchor(Edge::Right, true);
                    gtk_window.set_anchor(Edge::Left, false);
                    gtk_window.set_anchor(Edge::Top, false);
                    gtk_window.set_anchor(Edge::Bottom, false);
                }
            }
        }
    });
}

/// Returns true when the environment variable is set to a truthy value
/// (e.g. "1", "true", "yes", "on").
/// "0", "false", "no", "off" and empty string are treated as falsy (case-insensitive).
/// Returns false when the variable is not set.
#[cfg(target_os = "linux")]
fn env_flag_enabled(name: &str) -> bool {
    match env::var(name) {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "no" | "off"
        ),
        Err(_) => false,
    }
}

/// Initializes GTK layer shell for Linux overlay window
/// Returns true if layer shell was successfully initialized, false otherwise
#[cfg(target_os = "linux")]
fn init_gtk_layer_shell(overlay_window: &tauri::webview::WebviewWindow) -> bool {
    if env_flag_enabled("HANDY_NO_GTK_LAYER_SHELL") {
        debug!("Skipping GTK layer shell init (HANDY_NO_GTK_LAYER_SHELL is enabled)");
        return false;
    }

    if !gtk_layer_shell::is_supported() {
        return false;
    }

    // Try to get the GTK window from the Tauri webview
    if let Ok(gtk_window) = overlay_window.gtk_window() {
        // Initialize layer shell
        gtk_window.init_layer_shell();
        gtk_window.set_layer(Layer::Overlay);
        gtk_window.set_keyboard_mode(KeyboardMode::None);
        gtk_window.set_exclusive_zone(0);

        update_gtk_layer_shell_anchors(overlay_window);

        return true;
    }
    false
}

/// Forces a window to be topmost using Win32 API (Windows only)
/// This is more reliable than Tauri's set_always_on_top which can be overridden
#[cfg(target_os = "windows")]
fn force_overlay_topmost(overlay_window: &tauri::webview::WebviewWindow) {
    use windows::Win32::UI::WindowsAndMessaging::{
        SetWindowPos, HWND_TOPMOST, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOSIZE, SWP_SHOWWINDOW,
    };

    // Clone because run_on_main_thread takes 'static
    let overlay_clone = overlay_window.clone();

    // Make sure the Win32 call happens on the UI thread
    let _ = overlay_clone.clone().run_on_main_thread(move || {
        if let Ok(hwnd) = overlay_clone.hwnd() {
            unsafe {
                // Force Z-order: make this window topmost without changing size/pos or stealing focus
                let _ = SetWindowPos(
                    hwnd,
                    Some(HWND_TOPMOST),
                    0,
                    0,
                    0,
                    0,
                    SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
                );
            }
        }
    });
}

fn get_monitor_with_cursor(app_handle: &AppHandle) -> Option<tauri::Monitor> {
    if let Some(mouse_location) = input::get_cursor_position(app_handle) {
        if let Ok(monitors) = app_handle.available_monitors() {
            for monitor in monitors {
                // On Windows both the cursor (enigo -> GetCursorPos) and the
                // monitor bounds are physical pixels, so compare them directly.
                #[cfg(target_os = "windows")]
                if is_mouse_within_monitor(mouse_location, monitor.position(), monitor.size()) {
                    return Some(monitor);
                }

                // macOS/Linux: enigo returns logical coords, so scale the bounds down.
                #[cfg(not(target_os = "windows"))]
                {
                    let scale = monitor.scale_factor();
                    let pos = PhysicalPosition::new(
                        (monitor.position().x as f64 / scale) as i32,
                        (monitor.position().y as f64 / scale) as i32,
                    );
                    let size = PhysicalSize::new(
                        (monitor.size().width as f64 / scale) as u32,
                        (monitor.size().height as f64 / scale) as u32,
                    );
                    if is_mouse_within_monitor(mouse_location, &pos, &size) {
                        return Some(monitor);
                    }
                }
            }
        }
    }

    app_handle.primary_monitor().ok().flatten()
}

fn is_mouse_within_monitor(
    mouse_pos: (i32, i32),
    monitor_pos: &PhysicalPosition<i32>,
    monitor_size: &PhysicalSize<u32>,
) -> bool {
    let (mouse_x, mouse_y) = mouse_pos;
    let PhysicalPosition {
        x: monitor_x,
        y: monitor_y,
    } = *monitor_pos;
    let PhysicalSize {
        width: monitor_width,
        height: monitor_height,
    } = *monitor_size;

    mouse_x >= monitor_x
        && mouse_x < (monitor_x + monitor_width as i32)
        && mouse_y >= monitor_y
        && mouse_y < (monitor_y + monitor_height as i32)
}

/// Returns the docked Flow Bar position in logical coordinates.
///
/// macOS uses the visible work area so the bar clears the Dock and menu bar.
/// Other platforms use the monitor bounds plus a small safety inset because
/// Wayland work-area reporting is not consistent across compositors.
///
/// We must use LogicalPosition (not PhysicalPosition) because Tauri/tao
/// converts PhysicalPosition using the scale factor of the monitor the window
/// is *currently* on, which is wrong when moving cross-monitor. Windows uses
/// `place_windows_overlay` instead (no single logical space across mixed DPI).
fn calculate_overlay_position(
    app_handle: &AppHandle,
    width: f64,
    height: f64,
) -> Option<(f64, f64)> {
    let monitor = get_monitor_with_cursor(app_handle)?;
    let scale = monitor.scale_factor();
    #[cfg(not(target_os = "macos"))]
    let monitor_x = monitor.position().x as f64 / scale;
    #[cfg(not(target_os = "macos"))]
    let monitor_y = monitor.position().y as f64 / scale;
    #[cfg(not(target_os = "macos"))]
    let monitor_width = monitor.size().width as f64 / scale;
    #[cfg(not(target_os = "macos"))]
    let monitor_height = monitor.size().height as f64 / scale;

    #[cfg(target_os = "macos")]
    let (edge_x, edge_y, edge_width, edge_height) = {
        let work = monitor.work_area();
        (
            work.position.x as f64 / scale,
            work.position.y as f64 / scale,
            work.size.width as f64 / scale,
            work.size.height as f64 / scale,
        )
    };
    #[cfg(not(target_os = "macos"))]
    let (edge_x, edge_y, edge_width, edge_height) =
        (monitor_x, monitor_y, monitor_width, monitor_height);

    let settings = settings::get_settings(app_handle);

    let (x, y) = match settings.overlay_position {
        OverlayPosition::Bottom => (
            edge_x + (edge_width - width) / 2.0,
            edge_y + edge_height - height - OVERLAY_EDGE_OFFSET,
        ),
        OverlayPosition::Left => (
            edge_x + OVERLAY_EDGE_OFFSET,
            edge_y + (edge_height - height) / 2.0,
        ),
        OverlayPosition::Right => (
            edge_x + edge_width - width - OVERLAY_EDGE_OFFSET,
            edge_y + (edge_height - height) / 2.0,
        ),
    };

    Some((x, y))
}

/// Overlay rectangle in the destination monitor's physical pixels, so nothing
/// is converted through the window's previous-monitor DPI.
#[cfg(target_os = "windows")]
fn windows_overlay_bounds(
    monitor_position: PhysicalPosition<i32>,
    monitor_size: PhysicalSize<u32>,
    scale: f64,
    logical_width: f64,
    logical_height: f64,
    overlay_position: OverlayPosition,
) -> (i32, i32, i32, i32) {
    let width = (logical_width * scale).round().max(1.0) as i32;
    let height = (logical_height * scale).round().max(1.0) as i32;
    let (x, y) = match overlay_position {
        OverlayPosition::Bottom => (
            (monitor_position.x as f64 + (monitor_size.width as f64 - width as f64) / 2.0).round()
                as i32,
            (monitor_position.y as f64 + monitor_size.height as f64
                - height as f64
                - OVERLAY_EDGE_OFFSET * scale)
                .round() as i32,
        ),
        OverlayPosition::Left => (
            (monitor_position.x as f64 + OVERLAY_EDGE_OFFSET * scale).round() as i32,
            (monitor_position.y as f64 + (monitor_size.height as f64 - height as f64) / 2.0).round()
                as i32,
        ),
        OverlayPosition::Right => (
            (monitor_position.x as f64 + monitor_size.width as f64
                - width as f64
                - OVERLAY_EDGE_OFFSET * scale)
                .round() as i32,
            (monitor_position.y as f64 + (monitor_size.height as f64 - height as f64) / 2.0).round()
                as i32,
        ),
    };

    (x, y, width, height)
}

/// Moves and sizes the overlay in one native SetWindowPos, bypassing tao's
/// current-DPI logical conversion that mislands cross-monitor moves.
#[cfg(target_os = "windows")]
fn place_windows_overlay(
    app_handle: &AppHandle,
    overlay_window: &tauri::webview::WebviewWindow,
    logical_width: f64,
    logical_height: f64,
) -> Result<(), String> {
    use windows::Win32::UI::WindowsAndMessaging::{SetWindowPos, SWP_NOACTIVATE, SWP_NOZORDER};

    let monitor = get_monitor_with_cursor(app_handle)
        .ok_or_else(|| "failed to determine the monitor containing the cursor".to_string())?;
    let (x, y, width, height) = windows_overlay_bounds(
        *monitor.position(),
        *monitor.size(),
        monitor.scale_factor(),
        logical_width,
        logical_height,
        settings::get_settings(app_handle).overlay_position,
    );
    let hwnd = overlay_window
        .hwnd()
        .map_err(|error| format!("failed to get overlay window handle: {error}"))?;

    unsafe {
        SetWindowPos(
            hwnd,
            None,
            x,
            y,
            width,
            height,
            SWP_NOACTIVATE | SWP_NOZORDER,
        )
        .map_err(|error| format!("failed to set overlay bounds: {error}"))?;
    }

    log::debug!(
        "windows overlay bounds: x={} y={} width={} height={} scale={}",
        x,
        y,
        width,
        height,
        monitor.scale_factor()
    );
    Ok(())
}

/// Creates the recording overlay window and keeps it hidden by default
#[cfg(not(target_os = "macos"))]
pub fn create_recording_overlay(app_handle: &AppHandle) {
    let initial_position = settings::get_settings(app_handle).overlay_position;
    let (initial_width, initial_height) = overlay_dimensions("resting", initial_position);
    // On Linux (Wayland), monitor detection often fails, but we don't need exact coordinates
    // for Layer Shell as we use anchors. On other platforms, we require a monitor.
    #[cfg(not(target_os = "linux"))]
    {
        let position = calculate_overlay_position(app_handle, initial_width, initial_height);
        if position.is_none() {
            debug!("Failed to determine overlay position, not creating overlay window");
            return;
        }
    }

    // Position starts unset — update_overlay_position() sets the correct
    // LogicalPosition before the overlay is shown.
    let mut builder = WebviewWindowBuilder::new(
        app_handle,
        "recording_overlay",
        tauri::WebviewUrl::App("src/overlay/index.html".into()),
    )
    .title("Recording")
    .resizable(false)
    .inner_size(initial_width, initial_height)
    .shadow(false)
    .maximizable(false)
    .minimizable(false)
    .closable(false)
    .accept_first_mouse(true)
    .decorations(false)
    .always_on_top(true)
    .skip_taskbar(true)
    .transparent(true)
    .focusable(false)
    .focused(false)
    .visible(false);

    if let Some(data_dir) = crate::portable::data_dir() {
        builder = builder.data_directory(data_dir.join("webview"));
    }

    #[allow(unused_variables)]
    match builder.build() {
        Ok(window) => {
            #[cfg(target_os = "linux")]
            {
                // Try to initialize GTK layer shell, ignore errors if compositor doesn't support it
                if init_gtk_layer_shell(&window) {
                    debug!("GTK layer shell initialized for overlay window");
                } else {
                    debug!("GTK layer shell not available, falling back to regular window");
                }
            }

            if settings::get_settings(app_handle).overlay_style != OverlayStyle::None {
                let _ = window.show();
                let _ = window.emit("show-overlay", "resting");
            }
            debug!("Persistent Flow Bar window created successfully");
        }
        Err(e) => {
            debug!("Failed to create recording overlay window: {}", e);
        }
    }
}

/// Creates a separate click-through tooltip surface. Keeping it out of the
/// recording panel lets labels extend beyond the tight hit-tested Flow Bar
/// without swallowing clicks intended for the app underneath.
#[cfg(not(target_os = "macos"))]
pub fn create_flowbar_tooltip(app_handle: &AppHandle) {
    match WebviewWindowBuilder::new(
        app_handle,
        "flowbar_tooltip",
        tauri::WebviewUrl::App("src/overlay/tooltip.html".into()),
    )
    .title("Flow Bar Tooltip")
    .inner_size(TOOLTIP_WIDTH, TOOLTIP_HEIGHT)
    .resizable(false)
    .shadow(false)
    .decorations(false)
    .always_on_top(true)
    .skip_taskbar(true)
    .transparent(true)
    .focusable(false)
    .visible(false)
    .build()
    {
        Ok(window) => {
            if let Err(error) = window.set_ignore_cursor_events(true) {
                debug!("Failed to make Flow Bar tooltip click-through: {error}");
            }
        }
        Err(error) => debug!("Failed to create Flow Bar tooltip: {error}"),
    }
}

/// Creates the recording overlay panel and keeps it hidden by default (macOS)
#[cfg(target_os = "macos")]
pub fn create_recording_overlay(app_handle: &AppHandle) {
    let settings = settings::get_settings(app_handle);
    let (initial_width, initial_height) = overlay_dimensions("resting", settings.overlay_position);
    if let Some((x, y)) = calculate_overlay_position(app_handle, initial_width, initial_height) {
        // PanelBuilder creates a Tauri window then converts it to NSPanel.
        // The window remains registered, so get_webview_window() still works.
        match PanelBuilder::<_, RecordingOverlayPanel>::new(app_handle, "recording_overlay")
            .url(WebviewUrl::App("src/overlay/index.html".into()))
            .title("Recording")
            .position(tauri::Position::Logical(tauri::LogicalPosition { x, y }))
            .level(PanelLevel::Status)
            .size(tauri::Size::Logical(tauri::LogicalSize {
                width: initial_width,
                height: initial_height,
            }))
            .has_shadow(false)
            .transparent(true)
            .no_activate(true)
            .corner_radius(0.0)
            .style_mask(StyleMask::empty().borderless().nonactivating_panel())
            // A non-activating NSPanel normally consumes the first mouse press
            // only to establish mouse eligibility. The persistent Flow Bar
            // must start hands-free dictation on that very first click while
            // leaving focus in the user's current text field.
            .with_window(|w| {
                w.decorations(false)
                    .transparent(true)
                    .focusable(false)
                    .accept_first_mouse(true)
            })
            .collection_behavior(
                CollectionBehavior::new()
                    .can_join_all_spaces()
                    .full_screen_auxiliary(),
            )
            .build()
        {
            Ok(panel) => {
                if settings.overlay_style == OverlayStyle::None {
                    panel.hide();
                } else {
                    panel.show();
                    if let Some(window) = app_handle.get_webview_window("recording_overlay") {
                        let _ = window.emit("show-overlay", "resting");
                    }
                }
                start_flowbar_hit_test(app_handle);
            }
            Err(e) => {
                log::error!("Failed to create recording overlay panel: {}", e);
            }
        }
    }
}

#[cfg(target_os = "macos")]
pub fn create_flowbar_tooltip(app_handle: &AppHandle) {
    match PanelBuilder::<_, FlowBarTooltipPanel>::new(app_handle, "flowbar_tooltip")
        .url(WebviewUrl::App("src/overlay/tooltip.html".into()))
        .title("Flow Bar Tooltip")
        .level(PanelLevel::Status)
        .size(tauri::Size::Logical(tauri::LogicalSize {
            width: TOOLTIP_WIDTH,
            height: TOOLTIP_HEIGHT,
        }))
        .has_shadow(false)
        .transparent(true)
        .no_activate(true)
        .ignores_mouse_events(true)
        .corner_radius(0.0)
        .style_mask(StyleMask::empty().borderless().nonactivating_panel())
        .with_window(|window| window.decorations(false).transparent(true).focusable(false))
        .collection_behavior(
            CollectionBehavior::new()
                .can_join_all_spaces()
                .full_screen_auxiliary(),
        )
        .build()
    {
        Ok(panel) => panel.hide(),
        Err(error) => log::error!("Failed to create Flow Bar tooltip panel: {error}"),
    }
}

fn tooltip_position_for_bounds(
    position: tauri::LogicalPosition<f64>,
    size: tauri::LogicalSize<f64>,
    dock: OverlayPosition,
    kind: &str,
) -> (f64, f64) {
    let gap = if kind == "dictate" {
        BUBBLE_TOOLTIP_GAP
    } else {
        ACTION_TOOLTIP_GAP
    };
    match dock {
        OverlayPosition::Bottom => (
            position.x + (size.width - TOOLTIP_WIDTH) / 2.0,
            // The 30 px pill is bottom-pinned inside 6 px hot-area padding.
            position.y + size.height - FLOW_BAR_PADDING - FLOW_BAR_THICKNESS - gap - TOOLTIP_HEIGHT,
        ),
        OverlayPosition::Left => (
            position.x + FLOW_BAR_PADDING + FLOW_BAR_THICKNESS + gap,
            position.y + (size.height - TOOLTIP_HEIGHT) / 2.0,
        ),
        OverlayPosition::Right => (
            position.x + size.width - FLOW_BAR_PADDING - FLOW_BAR_THICKNESS - gap - TOOLTIP_WIDTH,
            position.y + (size.height - TOOLTIP_HEIGHT) / 2.0,
        ),
    }
}

fn flowbar_tooltip_position(app: &AppHandle, kind: &str) -> Result<(f64, f64), String> {
    let bar = app
        .get_webview_window("recording_overlay")
        .ok_or_else(|| "Flow Bar window is unavailable".to_string())?;
    let scale = bar.scale_factor().map_err(|error| error.to_string())?;
    let position = bar
        .outer_position()
        .map_err(|error| error.to_string())?
        .to_logical::<f64>(scale);
    let size = bar
        .outer_size()
        .map_err(|error| error.to_string())?
        .to_logical::<f64>(scale);

    Ok(tooltip_position_for_bounds(
        position,
        size,
        settings::get_settings(app).overlay_position,
        kind,
    ))
}

#[tauri::command]
#[specta::specta]
pub fn show_flowbar_tooltip(app: AppHandle, kind: String) -> Result<(), String> {
    if kind != "dictate" && kind != "cancel" && kind != "finish" {
        return Err("Unknown Flow Bar tooltip".to_string());
    }
    let tooltip = app
        .get_webview_window("flowbar_tooltip")
        .ok_or_else(|| "Flow Bar tooltip is unavailable".to_string())?;
    let (x, y) = flowbar_tooltip_position(&app, &kind)?;
    tooltip
        .set_position(tauri::Position::Logical(tauri::LogicalPosition { x, y }))
        .map_err(|error| error.to_string())?;
    tooltip
        .emit("flowbar-tooltip", kind)
        .map_err(|error| error.to_string())?;
    tooltip.show().map_err(|error| error.to_string())
}

#[tauri::command]
#[specta::specta]
pub fn hide_flowbar_tooltip(app: AppHandle) -> Result<(), String> {
    if let Some(tooltip) = app.get_webview_window("flowbar_tooltip") {
        tooltip.hide().map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn show_overlay_state(app_handle: &AppHandle, state: &str) {
    // Whether the overlay shows at all is governed by overlay_style; position
    // only chooses its dock edge. Checked here (off the main thread)
    // so the common overlay-disabled case never pays for a main-thread hop.
    let settings = settings::get_settings(app_handle);
    if settings.overlay_style == OverlayStyle::None {
        return;
    }

    // The rest queries monitors and the cursor and mutates window geometry. On
    // Linux the monitor/cursor lookups hit GDK/Xlib on the process's shared X11
    // connection, which is only safe from the GTK main thread — running them on
    // a background thread corrupts the connection and hard-crashes the app
    // (issue #227). Hop to the main thread on every platform to keep the
    // geometry path uniform (a no-op cost on Windows, and it also keeps macOS's
    // NSScreen access main-thread-correct). run_on_main_thread runs the closure
    // inline when already on the main thread, so this never deadlocks.
    let handle = app_handle.clone();
    let state = state.to_string();
    let _ = app_handle.run_on_main_thread(move || show_overlay_state_on_main(&handle, &state));
}

fn show_overlay_state_on_main(app_handle: &AppHandle, state: &str) {
    // Keep the previous native bounds for the 100 ms active -> resting morph.
    // Shrinking the window before the CSS transition completes clips the pill.
    OVERLAY_SHOW_GENERATION.fetch_add(1, Ordering::SeqCst);
    OVERLAY_SESSION_SHOWING.store(is_session_state(state), Ordering::SeqCst);
    let next_state_code = overlay_state_code(state);
    let previous_state_code = CURRENT_OVERLAY_STATE.swap(next_state_code, Ordering::Relaxed);
    let should_delay_collapse = next_state_code == 0 && previous_state_code != 0;
    let geometry_state = if should_delay_collapse {
        overlay_state_from_code(previous_state_code)
    } else {
        state
    };
    let position = settings::get_settings(app_handle).overlay_position;
    let (width, height) = overlay_dimensions(geometry_state, position);
    if let Some(overlay_window) = app_handle.get_webview_window("recording_overlay") {
        #[cfg(target_os = "linux")]
        update_gtk_layer_shell_anchors(&overlay_window);

        let size_started = std::time::Instant::now();
        #[cfg(not(target_os = "windows"))]
        let _ = overlay_window.set_size(tauri::Size::Logical(tauri::LogicalSize { width, height }));
        let size_elapsed = size_started.elapsed();

        let pos_started = std::time::Instant::now();
        #[cfg(not(target_os = "windows"))]
        let set_pos_elapsed =
            if let Some((x, y)) = calculate_overlay_position(app_handle, width, height) {
                let set_pos_started = std::time::Instant::now();
                let _ = overlay_window
                    .set_position(tauri::Position::Logical(tauri::LogicalPosition { x, y }));
                set_pos_started.elapsed()
            } else {
                std::time::Duration::ZERO
            };
        #[cfg(target_os = "windows")]
        let set_pos_elapsed = {
            let set_pos_started = std::time::Instant::now();
            if let Err(error) = place_windows_overlay(app_handle, &overlay_window, width, height) {
                log::error!("Failed to place recording overlay: {error}");
            }
            set_pos_started.elapsed()
        };
        let pos_calc_elapsed = pos_started.elapsed() - set_pos_elapsed;

        let show_started = std::time::Instant::now();
        let _ = overlay_window.show();
        let show_elapsed = show_started.elapsed();

        // On Windows, aggressively re-assert "topmost" in the native Z-order after showing
        #[cfg(target_os = "windows")]
        force_overlay_topmost(&overlay_window);

        // Re-assert bounds after show(): the pre-show move crosses the DPI
        // boundary, and tao's WM_DPICHANGED reflow clobbers the first placement.
        #[cfg(target_os = "windows")]
        if let Err(error) = place_windows_overlay(app_handle, &overlay_window, width, height) {
            log::error!("Failed to re-assert recording overlay position: {error}");
        }

        let _ = overlay_window.emit("show-overlay", state);
        log::debug!(
            "overlay '{}': set_size={:?} pos_calc={:?} set_pos={:?} show={:?}",
            state,
            size_elapsed,
            pos_calc_elapsed,
            set_pos_elapsed,
            show_elapsed
        );

        if should_delay_collapse {
            let handle = app_handle.clone();
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(
                    FLOW_BAR_MORPH_COLLAPSE_DELAY_MS,
                ));
                if CURRENT_OVERLAY_STATE.load(Ordering::Relaxed) != 0 {
                    return;
                }
                let main_handle = handle.clone();
                let _ = handle.run_on_main_thread(move || {
                    if CURRENT_OVERLAY_STATE.load(Ordering::Relaxed) == 0 {
                        update_overlay_position_on_main(&main_handle);
                    }
                });
            });
        }
    }
}

/// Shows the recording overlay window with fade-in animation
pub fn show_recording_overlay(app_handle: &AppHandle) {
    let state = if RECORDING_PUSH_TO_TALK.load(Ordering::Relaxed) {
        "recording-ptt"
    } else {
        "recording-toggle"
    };
    show_overlay_state(app_handle, state);
}

// Set by the serial transcription coordinator immediately before an action
// starts, so the renderer receives the actual input mode rather than guessing
// from shortcut names.
static RECORDING_PUSH_TO_TALK: AtomicBool = AtomicBool::new(false);

pub fn set_recording_push_to_talk(push_to_talk: bool) {
    RECORDING_PUSH_TO_TALK.store(push_to_talk, Ordering::Relaxed);
}

/// Shows the larger streaming overlay that displays live transcription text
pub fn show_streaming_overlay(app_handle: &AppHandle) {
    show_overlay_state(app_handle, "streaming");
}

/// Shows the transcribing overlay window
pub fn show_transcribing_overlay(app_handle: &AppHandle) {
    show_overlay_state(app_handle, "transcribing");
}

/// Shows the processing overlay window
pub fn show_processing_overlay(app_handle: &AppHandle) {
    show_overlay_state(app_handle, "processing");
}

/// Returns the persistent Flow Bar to its compact idle state. If overlays are
/// disabled, the native window is hidden instead.
pub fn show_idle_overlay(app_handle: &AppHandle) {
    if settings::get_settings(app_handle).overlay_style == OverlayStyle::None {
        hide_overlay_window(app_handle);
    } else {
        show_overlay_state(app_handle, "resting");
    }
}

/// Updates the overlay window position based on current settings
pub fn update_overlay_position(app_handle: &AppHandle) {
    // Positioning queries monitors/cursor (GDK/Xlib on Linux) and moves the
    // window, so it must run on the main thread — see show_overlay_state.
    let handle = app_handle.clone();
    let _ = app_handle.run_on_main_thread(move || update_overlay_position_on_main(&handle));
}

/// Expands the existing transparent panel to the cursor's monitor while the
/// user drags the Flow Bar. The renderer owns the ghost and drop targets.
#[tauri::command]
#[specta::specta]
pub fn begin_overlay_drag(app: AppHandle) -> Result<(), String> {
    let monitor = get_monitor_with_cursor(&app)
        .ok_or_else(|| "failed to determine the monitor containing the cursor".to_string())?;
    let scale = monitor.scale_factor();
    let x = monitor.position().x as f64 / scale;
    let y = monitor.position().y as f64 / scale;
    let width = monitor.size().width as f64 / scale;
    let height = monitor.size().height as f64 / scale;
    let window = app
        .get_webview_window("recording_overlay")
        .ok_or_else(|| "Flow Bar window is unavailable".to_string())?;

    window
        .set_position(tauri::Position::Logical(tauri::LogicalPosition { x, y }))
        .map_err(|error| error.to_string())?;
    window
        .set_size(tauri::Size::Logical(tauri::LogicalSize { width, height }))
        .map_err(|error| error.to_string())?;
    let _ = window.show();
    Ok(())
}

/// Persists the selected edge and collapses the drag surface back around the
/// painted Flow Bar.
#[tauri::command]
#[specta::specta]
pub fn finish_overlay_drag(app: AppHandle, edge: String) -> Result<(), String> {
    let position = match edge.as_str() {
        "bottom" => OverlayPosition::Bottom,
        "left" => OverlayPosition::Left,
        "right" => OverlayPosition::Right,
        _ => return Err(format!("invalid Flow Bar edge: {edge}")),
    };

    let mut app_settings = settings::get_settings(&app);
    app_settings.overlay_position = position;
    settings::write_settings(&app, app_settings);

    let state = current_overlay_state();
    let (width, height) = overlay_dimensions(state, position);
    let window = app
        .get_webview_window("recording_overlay")
        .ok_or_else(|| "Flow Bar window is unavailable".to_string())?;
    window
        .set_size(tauri::Size::Logical(tauri::LogicalSize { width, height }))
        .map_err(|error| error.to_string())?;
    if let Some((x, y)) = calculate_overlay_position(&app, width, height) {
        window
            .set_position(tauri::Position::Logical(tauri::LogicalPosition { x, y }))
            .map_err(|error| error.to_string())?;
    }
    let _ = window.emit("overlay-position-changed", edge);
    Ok(())
}

/// Mirrors the main dictation shortcut when the expanded idle bar is clicked.
#[tauri::command]
#[specta::specta]
pub fn toggle_overlay_dictation(app: AppHandle) {
    if let Some(coordinator) = app.try_state::<crate::TranscriptionCoordinator>() {
        coordinator.toggle_from_overlay();
    } else {
        log::warn!("TranscriptionCoordinator not initialized");
    }
}

/// Shows a local recovery card when the target application did not consume the
/// transcript. The text is sent only to the overlay renderer and is not logged.
pub fn show_paste_recovery_overlay(app_handle: &AppHandle, text: String) {
    if text.trim().is_empty() {
        show_idle_overlay(app_handle);
        return;
    }

    show_overlay_state(app_handle, "paste-recovery");
    if let Some(window) = app_handle.get_webview_window("recording_overlay") {
        let _ = window.emit("paste-recovery", text);
    }
}

/// How long a notice holds before the bar returns to resting.
const NOTICE_HOLD: Duration = Duration::from_secs(4);

/// Bumped on every state change; a delayed notice fade only fires if nothing
/// newer has shown since.
static OVERLAY_SHOW_GENERATION: AtomicU64 = AtomicU64::new(0);
/// The show generation of the latest notice; it is on screen while this matches.
static OVERLAY_NOTICE_GENERATION: AtomicU64 = AtomicU64::new(u64::MAX);
/// Set while the bar shows a recording or transcribing session.
static OVERLAY_SESSION_SHOWING: AtomicBool = AtomicBool::new(false);

fn is_session_state(state: &str) -> bool {
    matches!(
        state,
        "recording-ptt" | "recording-toggle" | "streaming" | "transcribing" | "processing"
    )
}

pub fn is_session_showing() -> bool {
    OVERLAY_SESSION_SHOWING.load(Ordering::SeqCst)
}

/// Notify the bar that the input stream delivered its first sample chunk.
pub fn emit_recording_ready(app_handle: &AppHandle) {
    if !OVERLAY_ENABLED.load(Ordering::Relaxed) {
        return;
    }
    // Queued on the main thread like show-overlay, so it cannot overtake it.
    let handle = app_handle.clone();
    let _ = app_handle.run_on_main_thread(move || {
        let _ = handle.emit_to("recording_overlay", "recording-ready", ());
    });
}

/// Morphs the bar into `notice` and returns it to resting after [`NOTICE_HOLD`]
/// unless something newer has shown.
pub fn show_overlay_notice(app_handle: &AppHandle, notice: &Notice) {
    if !OVERLAY_ENABLED.load(Ordering::Relaxed) {
        hide_recording_overlay(app_handle);
        return;
    }
    let handle = app_handle.clone();
    let notice = notice.clone();
    let _ = app_handle.run_on_main_thread(move || {
        let _ = handle.emit_to("recording_overlay", "overlay-notice", notice);
        show_overlay_state_on_main(&handle, "notice");
        let generation = OVERLAY_SHOW_GENERATION.load(Ordering::SeqCst);
        OVERLAY_NOTICE_GENERATION.store(generation, Ordering::SeqCst);
        std::thread::spawn(move || {
            std::thread::sleep(NOTICE_HOLD);
            if OVERLAY_SHOW_GENERATION.load(Ordering::SeqCst) == generation {
                show_idle_overlay(&handle);
            }
        });
    });
}

/// Ends the notice early, unless a new session has replaced it.
pub fn dismiss_overlay_notice(app_handle: &AppHandle) {
    let shown = OVERLAY_SHOW_GENERATION.load(Ordering::SeqCst);
    if OVERLAY_NOTICE_GENERATION.load(Ordering::SeqCst) == shown {
        show_idle_overlay(app_handle);
    }
}

#[tauri::command]
#[specta::specta]
pub fn dismiss_overlay_card(app: AppHandle) {
    show_idle_overlay(&app);
}

fn update_overlay_position_on_main(app_handle: &AppHandle) {
    if let Some(overlay_window) = app_handle.get_webview_window("recording_overlay") {
        #[cfg(target_os = "linux")]
        {
            update_gtk_layer_shell_anchors(&overlay_window);
        }

        #[cfg(target_os = "windows")]
        {
            let state = current_overlay_state();
            let position = settings::get_settings(app_handle).overlay_position;
            let (width, height) = overlay_dimensions(state, position);
            if let Err(error) = place_windows_overlay(app_handle, &overlay_window, width, height) {
                log::error!("Failed to update recording overlay position: {error}");
            }
        }

        #[cfg(not(target_os = "windows"))]
        {
            let position = settings::get_settings(app_handle).overlay_position;
            let (width, height) = overlay_dimensions(current_overlay_state(), position);
            let _ =
                overlay_window.set_size(tauri::Size::Logical(tauri::LogicalSize { width, height }));
            if let Some((x, y)) = calculate_overlay_position(app_handle, width, height) {
                let _ = overlay_window
                    .set_position(tauri::Position::Logical(tauri::LogicalPosition { x, y }));
            }
        }
    }
}

fn hide_overlay_window(app_handle: &AppHandle) {
    if let Some(overlay_window) = app_handle.get_webview_window("recording_overlay") {
        let _ = overlay_window.emit("hide-overlay", ());
        let _ = overlay_window.hide();
    }
}

/// Ends a recording/transcription visual by morphing back to the persistent
/// resting bar. `None` remains the only setting that truly hides the window.
pub fn hide_recording_overlay(app_handle: &AppHandle) {
    show_idle_overlay(app_handle);
}

// Cached "overlay is enabled" flag, kept in sync with overlay_style. Avoids
// reading the Tauri store on every audio callback (~24 Hz during recording).
// Defaults to false so the audio path doesn't emit until lib.rs::setup
// populates the cache from initial settings.
static OVERLAY_ENABLED: AtomicBool = AtomicBool::new(false);

/// Update the cached overlay-enabled flag. Called from `lib.rs` at
/// startup after settings load, and from `change_overlay_style_setting`
/// whenever the user changes whether the overlay is shown.
pub fn update_overlay_enabled_cache(enabled: bool) {
    OVERLAY_ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn emit_levels(app_handle: &AppHandle, levels: &[f32]) {
    // Skip emission when the overlay is disabled. The recording_overlay
    // window is created at boot regardless of overlay_style, so without this
    // guard a hidden overlay's WebKit subprocess still
    // processes every event. Each event drives some kind of WebKit
    // C++ allocation that accumulates without bound (mechanism not
    // directly characterized; see issue #1279 for the investigation).
    // For users with `overlay_style: none` (the Linux default) this skip
    // eliminates the upstream driver of that accumulation.
    if !OVERLAY_ENABLED.load(Ordering::Relaxed) {
        return;
    }

    // Throttle to ~30 FPS. Even with the overlay enabled, the raw audio
    // callback fires far faster than the UI needs; capping emission rate
    // cuts the per-frame `eval_script`/IPC volume that drives the wry
    // memory growth in issue #1279 (upstream tauri-apps/wry#1489).
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let last = LAST_MIC_LEVEL_EMIT.load(Ordering::Relaxed);
    if now.saturating_sub(last) < EMIT_THROTTLE_MS {
        return;
    }
    LAST_MIC_LEVEL_EMIT.store(now, Ordering::Relaxed);

    // Target only the overlay window. In Tauri 2 both `AppHandle::emit`
    // and `WebviewWindow::emit` broadcast to all webviews; Tauri's
    // listener filter then skips webviews with no registered listener
    // for the event, so the settings webview never received `mic-level`.
    // But the previous dual-call pattern still produced two `eval_script`
    // calls to the overlay per audio callback (one from each .emit()).
    // `emit_to` with the overlay's window label produces a single
    // eval_script call per callback, cutting the per-callback WebKit
    // dispatch work in half.
    let _ = app_handle.emit_to("recording_overlay", "mic-level", levels);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlay_geometry_tracks_state_and_dock_edge() {
        assert_eq!(
            overlay_dimensions("resting", OverlayPosition::Bottom),
            (BOTTOM_REST_WIDTH, BOTTOM_HEIGHT)
        );
        assert_eq!(
            overlay_dimensions("recording-toggle", OverlayPosition::Left),
            (SIDE_ACTIVE_WIDTH, SIDE_ACTIVE_HEIGHT)
        );
        assert_eq!(
            overlay_dimensions("processing", OverlayPosition::Right),
            (SIDE_WORK_WIDTH, SIDE_WORK_HEIGHT)
        );
        assert_eq!(
            overlay_dimensions("streaming", OverlayPosition::Bottom),
            (OVERLAY_STREAM_WIDTH, OVERLAY_STREAM_HEIGHT)
        );
        assert_eq!(
            overlay_dimensions("paste-recovery", OverlayPosition::Right),
            (RECOVERY_WIDTH, RECOVERY_HEIGHT)
        );
        assert_eq!(
            overlay_dimensions("notice", OverlayPosition::Bottom),
            (NOTICE_WIDTH, NOTICE_HEIGHT)
        );
    }

    #[test]
    fn overlay_state_codes_round_trip_to_geometry_states() {
        for state in [
            "resting",
            "recording-ptt",
            "recording-toggle",
            "streaming",
            "transcribing",
            "processing",
            "error",
            "paste-recovery",
            "notice",
        ] {
            assert_eq!(overlay_state_from_code(overlay_state_code(state)), state);
        }
    }

    #[test]
    fn tooltip_geometry_uses_flow_offsets_from_the_visible_pill() {
        let origin = tauri::LogicalPosition::new(100.0, 200.0);

        assert_eq!(
            tooltip_position_for_bounds(
                origin,
                tauri::LogicalSize::new(BOTTOM_ACTIVE_WIDTH, BOTTOM_HEIGHT),
                OverlayPosition::Bottom,
                "finish",
            ),
            (78.0, 174.0)
        );
        assert_eq!(
            tooltip_position_for_bounds(
                origin,
                tauri::LogicalSize::new(SIDE_ACTIVE_WIDTH, SIDE_ACTIVE_HEIGHT),
                OverlayPosition::Left,
                "cancel",
            ),
            (146.0, 249.0)
        );
        assert_eq!(
            tooltip_position_for_bounds(
                origin,
                tauri::LogicalSize::new(SIDE_REST_WIDTH, SIDE_REST_HEIGHT),
                OverlayPosition::Right,
                "dictate",
            ),
            (-62.0, 221.0)
        );
    }

    #[test]
    fn resting_hit_rect_is_tight_until_hover_then_adds_ready_hysteresis() {
        assert_eq!(
            flowbar_hit_rect(
                "resting",
                OverlayPosition::Right,
                SIDE_REST_WIDTH,
                SIDE_REST_HEIGHT,
                false,
            ),
            FlowBarHitRect {
                x: 34.0,
                y: 12.0,
                width: 20.0,
                height: 52.0,
            }
        );
        assert_eq!(
            flowbar_hit_rect(
                "resting",
                OverlayPosition::Bottom,
                BOTTOM_REST_WIDTH,
                BOTTOM_HEIGHT,
                false,
            ),
            FlowBarHitRect {
                x: 8.0,
                y: 34.0,
                width: 60.0,
                height: 20.0,
            }
        );
        assert_eq!(
            flowbar_hit_rect(
                "resting",
                OverlayPosition::Right,
                SIDE_REST_WIDTH,
                SIDE_REST_HEIGHT,
                true,
            ),
            FlowBarHitRect {
                x: 12.0,
                y: 7.0,
                width: 42.0,
                height: 62.0,
            }
        );
    }

    #[test]
    fn active_hit_rect_tracks_rotated_side_pills() {
        assert_eq!(
            flowbar_hit_rect(
                "recording-toggle",
                OverlayPosition::Right,
                SIDE_ACTIVE_WIDTH,
                SIDE_ACTIVE_HEIGHT,
                true,
            ),
            FlowBarHitRect {
                x: 12.0,
                y: 8.75,
                width: 42.0,
                height: 114.5,
            }
        );
        assert_eq!(
            flowbar_hit_rect(
                "paste-recovery",
                OverlayPosition::Bottom,
                RECOVERY_WIDTH,
                RECOVERY_HEIGHT,
                false,
            ),
            FlowBarHitRect {
                x: 0.0,
                y: 0.0,
                width: RECOVERY_WIDTH,
                height: RECOVERY_HEIGHT,
            }
        );
        assert_eq!(
            flowbar_hit_rect(
                "processing",
                OverlayPosition::Left,
                SIDE_WORK_WIDTH,
                SIDE_WORK_HEIGHT,
                true,
            ),
            FlowBarHitRect {
                x: 0.0,
                y: 11.0,
                width: 42.0,
                height: 110.0,
            }
        );
    }

    #[test]
    fn monitor_hit_test_uses_half_open_physical_bounds() {
        let position = PhysicalPosition::new(-2560, -200);
        let size = PhysicalSize::new(2560, 1440);

        assert!(is_mouse_within_monitor((-2560, -200), &position, &size));
        assert!(is_mouse_within_monitor((-1, 1239), &position, &size));
        assert!(!is_mouse_within_monitor((0, 0), &position, &size));
        assert!(!is_mouse_within_monitor((-1, 1240), &position, &size));
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_cursor_hit_test_does_not_scale_physical_monitor_bounds() {
        let position = PhysicalPosition::new(1920, 0);
        let size = PhysicalSize::new(3840, 2160);
        let cursor = (5000, 1000);

        assert!(is_mouse_within_monitor(cursor, &position, &size));

        // This is the old mixed-coordinate comparison. It excludes a cursor
        // that is visibly inside a secondary display running at 150%.
        let scale = 1.5;
        let logical_position = PhysicalPosition::new(
            (position.x as f64 / scale) as i32,
            (position.y as f64 / scale) as i32,
        );
        let logical_size = PhysicalSize::new(
            (size.width as f64 / scale) as u32,
            (size.height as f64 / scale) as u32,
        );
        assert!(!is_mouse_within_monitor(
            cursor,
            &logical_position,
            &logical_size
        ));
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_overlay_bounds_use_destination_monitor_scale() {
        let monitor_position = PhysicalPosition::new(1920, 0);
        let monitor_size = PhysicalSize::new(3840, 2160);

        assert_eq!(
            windows_overlay_bounds(
                monitor_position,
                monitor_size,
                1.5,
                BOTTOM_REST_WIDTH,
                BOTTOM_HEIGHT,
                OverlayPosition::Bottom,
            ),
            (3783, 2067, 114, 81)
        );
        assert_eq!(
            windows_overlay_bounds(
                monitor_position,
                monitor_size,
                1.5,
                SIDE_REST_WIDTH,
                SIDE_REST_HEIGHT,
                OverlayPosition::Left,
            ),
            (1932, 1023, 81, 114)
        );
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_overlay_bounds_support_negative_monitor_origins() {
        assert_eq!(
            windows_overlay_bounds(
                PhysicalPosition::new(-2560, -200),
                PhysicalSize::new(2560, 1440),
                1.25,
                OVERLAY_STREAM_WIDTH,
                OVERLAY_STREAM_HEIGHT,
                OverlayPosition::Bottom,
            ),
            (-1530, 1080, 500, 150)
        );
    }
}
