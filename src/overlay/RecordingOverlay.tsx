import { listen } from "@tauri-apps/api/event";
import { writeText } from "@tauri-apps/plugin-clipboard-manager";
import React, { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import "./RecordingOverlay.css";
import { commands, events } from "@/bindings";
import type {
  Notice,
  OverlayPosition,
  StreamPhase,
  StreamPhaseEvent,
  StreamTextEvent,
  StreamWorkKind,
} from "@/bindings";
import { syncLanguageFromSettings } from "@/i18n";
import { noticeDisplay } from "@/lib/notices";

type OverlayState =
  | "resting"
  | "recording-ptt"
  | "recording-toggle"
  | "streaming"
  | "transcribing"
  | "processing"
  | "paste-recovery"
  | "notice"
  | "error";

type OverlayNotice = {
  title: string;
  body: string;
};

type DragPhase = "idle" | "arming" | "dragging" | "returning";
type TooltipKind = "dictate" | "cancel" | "finish";

const WAVE_BARS = 10;
const DRAG_THRESHOLD_SQ = 25;
const WAVE_GAIN = 5;
const HOVER_MORPH_LOCK_MS = 100;
const BUBBLE_TOOLTIP_DELAY_MS = 150;
const ACTION_TOOLTIP_DELAY_MS = 50;
const RECOVERY_TIMEOUT_MS = 8_000;
const RECOVERY_COPIED_CLOSE_MS = 1_200;
const NOTICE_TIMEOUT_MS = 6_000;

const WAVE_BAR_GEOMETRY = Array.from({ length: WAVE_BARS }, (_, index) => {
  const center = (WAVE_BARS - 1) / 2;
  const distance = Math.abs(center - index);
  return {
    heightScale: Math.max(0, 1 - (distance * distance) / 48),
    delay: 0.1 * (index < Math.ceil(WAVE_BARS / 2) ? index : index - WAVE_BARS),
  };
});

const normalizePosition = (value: string | undefined): OverlayPosition => {
  if (value === "left" || value === "right") return value;
  return "bottom";
};

const nearestEdge = (x: number, y: number): OverlayPosition => {
  const targets: Array<[OverlayPosition, number, number]> = [
    ["bottom", window.innerWidth / 2, window.innerHeight - 23],
    ["left", 18, window.innerHeight / 2],
    ["right", window.innerWidth - 18, window.innerHeight / 2],
  ];
  return targets.reduce(
    (nearest, target) => {
      const distance = (target[1] - x) ** 2 + (target[2] - y) ** 2;
      return distance < nearest.distance
        ? { edge: target[0], distance }
        : nearest;
    },
    { edge: "bottom" as OverlayPosition, distance: Number.POSITIVE_INFINITY },
  ).edge;
};

const edgeTarget = (edge: OverlayPosition) => {
  if (edge === "left") return { x: 18, y: window.innerHeight / 2 };
  if (edge === "right") {
    return { x: window.innerWidth - 18, y: window.innerHeight / 2 };
  }
  return { x: window.innerWidth / 2, y: window.innerHeight - 23 };
};

const MicGlyph: React.FC<{ className?: string }> = ({ className }) => (
  <svg
    className={className}
    width="16"
    height="16"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <rect x="9" y="2" width="6" height="11" rx="3" />
    <path d="M5 10a7 7 0 0 0 14 0M12 17v5M8 22h8" />
  </svg>
);

const RecordingOverlay: React.FC = () => {
  const { t } = useTranslation();
  const [isVisible, setIsVisible] = useState(true);
  const [state, setState] = useState<OverlayState>("resting");
  const [position, setPosition] = useState<OverlayPosition>("bottom");
  const [isHovered, setIsHovered] = useState(false);
  const [isNativeHovered, setIsNativeHovered] = useState(false);
  const [streamText, setStreamText] = useState<StreamTextEvent>({
    committed: "",
    tentative: "",
  });
  const [phase, setPhase] = useState<StreamPhase>("listening");
  const [, setWorkKind] = useState<StreamWorkKind>("transcribing");
  const [dragPhase, setDragPhase] = useState<DragPhase>("idle");
  const [dragEdge, setDragEdge] = useState<OverlayPosition>("bottom");
  const [ghost, setGhost] = useState({ x: -100, y: -100 });
  const [pasteRecoveryText, setPasteRecoveryText] = useState("");
  const [recoveryCopied, setRecoveryCopied] = useState(false);
  const [notice, setNotice] = useState<OverlayNotice | null>(null);

  const waveformRef = useRef<HTMLSpanElement>(null);
  const audioTargetRef = useRef(0);
  const audioScaleRef = useRef(1);
  const dragOriginRef = useRef<{ x: number; y: number } | null>(null);
  const dragStartedRef = useRef(false);
  const dragArmingRef = useRef(false);
  const dragTokenRef = useRef(0);
  const suppressClickRef = useRef(false);
  const tooltipTimerRef = useRef<number | null>(null);
  const hoverLockedUntilRef = useRef(0);
  const hoverReleaseTimerRef = useRef<number | null>(null);
  const recoveryTimerRef = useRef<number | null>(null);
  const noticeTimerRef = useRef<number | null>(null);

  useEffect(() => {
    let mounted = true;
    const setup = async () => {
      await syncLanguageFromSettings();
      try {
        const settings = await commands.getAppSettings();
        if (mounted && settings.status === "ok") {
          setPosition(normalizePosition(settings.data.overlay_position));
          setIsVisible(settings.data.overlay_style !== "none");
        }
      } catch {
        // The backend state event remains authoritative if the settings read fails.
      }

      const unlistenShow = await listen<OverlayState>(
        "show-overlay",
        (event) => {
          setState(event.payload);
          if (
            event.payload === "resting" ||
            event.payload === "recording-ptt" ||
            event.payload === "recording-toggle" ||
            event.payload === "streaming"
          ) {
            setStreamText({ committed: "", tentative: "" });
            setPhase("listening");
          }
          setIsVisible(true);
        },
      );
      const unlistenHide = await listen("hide-overlay", () =>
        setIsVisible(false),
      );
      const unlistenPosition = await listen<string>(
        "overlay-position-changed",
        (event) => setPosition(normalizePosition(event.payload)),
      );
      const unlistenLevel = await listen<number[]>("mic-level", (event) => {
        // Handy exposes a 16-bucket vocal spectrum; recovered Flow drives its
        // primary waveform from one scalar audio envelope. Collapse the buckets
        // to RMS here, then smooth it in the animation-frame loop below.
        const buckets = event.payload.filter(Number.isFinite);
        const meanSquare = buckets.length
          ? buckets.reduce((sum, level) => sum + level * level, 0) /
            buckets.length
          : 0;
        audioTargetRef.current = Math.min(1, Math.sqrt(meanSquare));
      });
      const unlistenNativeHover = await listen<boolean>(
        "flowbar-native-hover",
        (event) => setIsNativeHovered(event.payload),
      );
      const unlistenPasteRecovery = await listen<string>(
        "paste-recovery",
        (event) => {
          setPasteRecoveryText(event.payload);
          setRecoveryCopied(false);
        },
      );
      const unlistenStream = await events.streamTextEvent.listen((event) => {
        setStreamText(event.payload);
      });
      const unlistenPhase = await events.streamPhaseEvent.listen((event) => {
        const payload: StreamPhaseEvent = event.payload;
        setPhase(payload.phase);
        if (payload.kind) setWorkKind(payload.kind);
      });

      return () => {
        unlistenShow();
        unlistenHide();
        unlistenPosition();
        unlistenLevel();
        unlistenNativeHover();
        unlistenPasteRecovery();
        unlistenStream();
        unlistenPhase();
      };
    };

    let cleanup: (() => void) | undefined;
    void setup().then((value) => {
      if (mounted) cleanup = value;
      else value?.();
    });
    return () => {
      mounted = false;
      cleanup?.();
    };
  }, []);

  // Rust sends notice codes; the text comes from the shared notice table.
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let mounted = true;
    void listen<Notice>("overlay-notice", (event) => {
      const display = noticeDisplay(event.payload.code);
      setNotice({
        title: t(display.title),
        body: display.explanation?.(t) ?? event.payload.detail ?? "",
      });
    }).then((off) => {
      if (mounted) unlisten = off;
      else off();
    });
    return () => {
      mounted = false;
      unlisten?.();
    };
  }, [t]);

  const displayState =
    state === "resting" && (isHovered || isNativeHovered) ? "ready" : state;
  const isWorking =
    state === "transcribing" ||
    state === "processing" ||
    (state === "streaming" && phase === "working");
  const hasStreamText = Boolean(streamText.committed || streamText.tentative);
  const isListening =
    state === "recording-ptt" ||
    state === "recording-toggle" ||
    (state === "streaming" && phase === "listening");
  const hasToggleControls =
    state === "recording-toggle" ||
    (state === "streaming" && phase === "listening");

  useEffect(() => {
    let frame = 0;
    const updateWaveform = () => {
      const target = isListening
        ? Math.max(1, audioTargetRef.current * WAVE_GAIN)
        : 1;
      const current = audioScaleRef.current;
      const smoothing = target > current ? 0.28 : 0.14;
      const next = current + (target - current) * smoothing;
      audioScaleRef.current = Math.abs(next - target) < 0.01 ? target : next;
      waveformRef.current?.style.setProperty(
        "--audio-scale",
        audioScaleRef.current.toFixed(3),
      );
      frame = window.requestAnimationFrame(updateWaveform);
    };
    frame = window.requestAnimationFrame(updateWaveform);
    return () => window.cancelAnimationFrame(frame);
  }, [isListening]);

  const waveform = (
    <span
      ref={waveformRef}
      className={`flow-wave ${isListening ? "is-animated mic-active" : ""}`}
      aria-hidden="true"
    >
      {WAVE_BAR_GEOMETRY.map((bar, index) => (
        <i
          key={index}
          style={
            {
              "--bar-height-scale": bar.heightScale,
              "--delay": `${bar.delay}s`,
            } as React.CSSProperties
          }
        />
      ))}
    </span>
  );

  const loader = (
    <span className="flow-loader" aria-hidden="true">
      {Array.from({ length: 8 }, (_, index) => (
        <i
          key={index}
          style={
            {
              transform: `rotate(${index * 45}deg) translateY(-6px)`,
              animationDelay: `${-0.1375 * (7 - index)}s`,
            } as React.CSSProperties
          }
        />
      ))}
    </span>
  );

  const resetDrag = () => {
    dragTokenRef.current += 1;
    dragOriginRef.current = null;
    dragStartedRef.current = false;
    dragArmingRef.current = false;
  };

  const hideTooltip = useCallback(() => {
    if (tooltipTimerRef.current !== null) {
      window.clearTimeout(tooltipTimerRef.current);
      tooltipTimerRef.current = null;
    }
    void commands.hideFlowbarTooltip();
  }, []);

  const clearRecoveryTimer = useCallback(() => {
    if (recoveryTimerRef.current !== null) {
      window.clearTimeout(recoveryTimerRef.current);
      recoveryTimerRef.current = null;
    }
  }, []);

  const dismissRecovery = useCallback(() => {
    clearRecoveryTimer();
    setPasteRecoveryText("");
    setRecoveryCopied(false);
    void commands.dismissOverlayCard();
  }, [clearRecoveryTimer]);

  const armRecoveryTimer = useCallback(
    (delay = RECOVERY_TIMEOUT_MS) => {
      clearRecoveryTimer();
      recoveryTimerRef.current = window.setTimeout(() => {
        recoveryTimerRef.current = null;
        dismissRecovery();
      }, delay);
    },
    [clearRecoveryTimer, dismissRecovery],
  );

  useEffect(() => {
    if (state === "paste-recovery" && pasteRecoveryText) {
      armRecoveryTimer();
    } else {
      clearRecoveryTimer();
    }
    return clearRecoveryTimer;
  }, [armRecoveryTimer, clearRecoveryTimer, pasteRecoveryText, state]);

  const copyRecoveryText = useCallback(async () => {
    if (!pasteRecoveryText || recoveryCopied) return;
    try {
      await writeText(pasteRecoveryText);
      setRecoveryCopied(true);
      armRecoveryTimer(RECOVERY_COPIED_CLOSE_MS);
    } catch {
      // Keep the card open so the transcript remains recoverable on screen.
    }
  }, [armRecoveryTimer, pasteRecoveryText, recoveryCopied]);

  const clearNoticeTimer = useCallback(() => {
    if (noticeTimerRef.current !== null) {
      window.clearTimeout(noticeTimerRef.current);
      noticeTimerRef.current = null;
    }
  }, []);

  const dismissNotice = useCallback(() => {
    clearNoticeTimer();
    setNotice(null);
    void commands.dismissOverlayNotice();
  }, [clearNoticeTimer]);

  const armNoticeTimer = useCallback(() => {
    clearNoticeTimer();
    noticeTimerRef.current = window.setTimeout(() => {
      noticeTimerRef.current = null;
      dismissNotice();
    }, NOTICE_TIMEOUT_MS);
  }, [clearNoticeTimer, dismissNotice]);

  useEffect(() => {
    if (state === "notice" && notice) armNoticeTimer();
    else clearNoticeTimer();
    return clearNoticeTimer;
  }, [armNoticeTimer, clearNoticeTimer, notice, state]);

  const showTooltip = useCallback(
    (kind: TooltipKind, delay: number) => {
      hideTooltip();
      tooltipTimerRef.current = window.setTimeout(() => {
        tooltipTimerRef.current = null;
        void commands.showFlowbarTooltip(kind);
      }, delay);
    },
    [hideTooltip],
  );

  useEffect(() => hideTooltip, [hideTooltip]);
  useEffect(() => {
    hideTooltip();
  }, [hideTooltip, state, position]);

  const clearHoverRelease = useCallback(() => {
    if (hoverReleaseTimerRef.current !== null) {
      window.clearTimeout(hoverReleaseTimerRef.current);
      hoverReleaseTimerRef.current = null;
    }
  }, []);

  useEffect(
    () => () => {
      clearHoverRelease();
    },
    [clearHoverRelease],
  );

  useEffect(() => {
    if (state !== "resting") {
      clearHoverRelease();
      setIsHovered(false);
    }
  }, [clearHoverRelease, state]);

  useEffect(() => {
    if (state !== "resting") return;

    if (isNativeHovered) {
      clearHoverRelease();
      hoverLockedUntilRef.current = performance.now() + HOVER_MORPH_LOCK_MS;
      showTooltip("dictate", BUBBLE_TOOLTIP_DELAY_MS);
      return;
    }

    // WKWebView can miss pointerleave when the native panel switches back to
    // click-through. Treat the native alpha hit-test as authoritative so the
    // ready pill and tooltip cannot remain latched after the pointer has left.
    clearHoverRelease();
    setIsHovered(false);
    hideTooltip();
  }, [clearHoverRelease, hideTooltip, isNativeHovered, showTooltip, state]);

  const handleBarPointerEnter = () => {
    if (state !== "resting") return;
    clearHoverRelease();
    hoverLockedUntilRef.current = performance.now() + HOVER_MORPH_LOCK_MS;
    setIsHovered(true);
    showTooltip("dictate", BUBBLE_TOOLTIP_DELAY_MS);
  };

  const handleBarPointerLeave = () => {
    if (state !== "resting") return;
    hideTooltip();
    clearHoverRelease();
    const remaining = Math.max(
      0,
      hoverLockedUntilRef.current - performance.now(),
    );
    if (remaining === 0) {
      setIsHovered(false);
      return;
    }
    hoverReleaseTimerRef.current = window.setTimeout(() => {
      hoverReleaseTimerRef.current = null;
      setIsHovered(false);
    }, remaining);
  };

  const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 || state !== "resting") return;
    event.currentTarget.setPointerCapture(event.pointerId);
    dragTokenRef.current += 1;
    dragOriginRef.current = { x: event.clientX, y: event.clientY };
    dragStartedRef.current = false;
    suppressClickRef.current = false;
  };

  const handlePointerMove = async (
    event: React.PointerEvent<HTMLDivElement>,
  ) => {
    const origin = dragOriginRef.current;
    if (!origin) return;

    if (!dragStartedRef.current && !dragArmingRef.current) {
      const distance =
        (event.clientX - origin.x) ** 2 + (event.clientY - origin.y) ** 2;
      if (distance <= DRAG_THRESHOLD_SQ) return;
      dragArmingRef.current = true;
      const dragToken = dragTokenRef.current;
      setDragPhase("arming");
      const result = await commands.beginOverlayDrag();
      if (dragToken !== dragTokenRef.current || !dragOriginRef.current) {
        await commands.finishOverlayDrag(position);
        setDragPhase("idle");
        return;
      }
      dragArmingRef.current = false;
      if (result.status === "error") {
        setDragPhase("idle");
        resetDrag();
        return;
      }
      dragStartedRef.current = true;
      suppressClickRef.current = true;
      setDragPhase("dragging");
      return;
    }

    if (dragStartedRef.current) {
      setGhost({ x: event.clientX, y: event.clientY });
      setDragEdge(nearestEdge(event.clientX, event.clientY));
    }
  };

  const handlePointerUp = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!dragStartedRef.current) {
      resetDrag();
      return;
    }
    const edge = nearestEdge(event.clientX, event.clientY);
    const target = edgeTarget(edge);
    setDragEdge(edge);
    setGhost(target);
    setDragPhase("returning");
    window.setTimeout(async () => {
      await commands.finishOverlayDrag(edge);
      setPosition(edge);
      setDragPhase("idle");
      setIsHovered(false);
      resetDrag();
    }, 300);
  };

  const handleBarClick = () => {
    if (suppressClickRef.current || state !== "resting") {
      suppressClickRef.current = false;
      return;
    }
    void commands.toggleOverlayDictation();
  };

  if (!isVisible) return null;

  return (
    <main
      className={`flowbar-root ${dragPhase !== "idle" ? "is-dragging" : ""}`}
      data-position={position}
      data-state={displayState}
    >
      {dragPhase !== "idle" && (
        <div
          className={`drag-surface ${dragPhase === "returning" ? "exiting" : ""}`}
        >
          {(["bottom", "left", "right"] as OverlayPosition[]).map((edge) => (
            <span
              key={edge}
              className={`drop-zone ${edge} ${dragEdge === edge ? "nearest" : ""}`}
            />
          ))}
          <span
            className={`drag-ghost ${dragPhase === "returning" ? "returning" : ""}`}
            style={
              {
                "--ghost-x": `${ghost.x}px`,
                "--ghost-y": `${ghost.y}px`,
              } as React.CSSProperties
            }
          >
            <MicGlyph className="drag-ghost-icon" />
          </span>
        </div>
      )}

      <section className="flowbar-stage">
        {state === "streaming" && hasStreamText && (
          <div className="live-transcript" aria-live="polite">
            <span>{streamText.committed}</span>{" "}
            <span className="tentative">{streamText.tentative}</span>
          </div>
        )}
        {state === "notice" && notice ? (
          <aside
            className="overlay-notice"
            role="alert"
            onMouseEnter={clearNoticeTimer}
            onMouseLeave={armNoticeTimer}
          >
            <span className="notice-mark" aria-hidden="true">
              <svg width="12" height="12" viewBox="0 0 12 12">
                <circle cx="6" cy="6" r="4.75" />
                <path d="M6 5.25v3M6 3.5v.1" />
              </svg>
            </span>
            <div className="notice-copy">
              <strong>{notice.title}</strong>
              <p>{notice.body}</p>
            </div>
            <button
              className="notice-close"
              type="button"
              aria-label={t("common.close")}
              onClick={dismissNotice}
            >
              <svg
                width="12"
                height="12"
                viewBox="0 0 12 12"
                aria-hidden="true"
              >
                <path d="M2 2 10 10M10 2 2 10" />
              </svg>
            </button>
          </aside>
        ) : state === "paste-recovery" ? (
          <aside
            className="paste-recovery"
            role="alert"
            onMouseEnter={clearRecoveryTimer}
            onMouseLeave={() => armRecoveryTimer()}
          >
            <button
              className="recovery-close"
              type="button"
              aria-label={t("common.close")}
              onClick={dismissRecovery}
            >
              <svg
                width="12"
                height="12"
                viewBox="0 0 12 12"
                aria-hidden="true"
              >
                <path d="M2 2 10 10M10 2 2 10" />
              </svg>
            </button>
            <div className="recovery-hint">
              <svg
                width="16"
                height="16"
                viewBox="0 0 16 16"
                aria-hidden="true"
              >
                <rect x="2" y="3" width="12" height="10" rx="2" />
                <path d="M5 6h6M5 9h4" />
              </svg>
              <span>
                {t("overlay.pasteRecoveryHint", {
                  defaultValue: "Select a text field before dictating",
                })}
              </span>
            </div>
            <p className="recovery-text">{pasteRecoveryText}</p>
            <button
              className={`recovery-copy ${recoveryCopied ? "copied" : ""}`}
              type="button"
              disabled={recoveryCopied}
              onClick={() => void copyRecoveryText()}
            >
              {recoveryCopied
                ? t("overlay.pasteRecoveryCopied", {
                    defaultValue: "Copied",
                  })
                : t("overlay.pasteRecoveryCopy", {
                    defaultValue: "Copy dictation",
                  })}
            </button>
          </aside>
        ) : (
          <div
            className="hot-area"
            onPointerDown={handlePointerDown}
            onPointerMove={(event) => void handlePointerMove(event)}
            onPointerUp={handlePointerUp}
            onPointerCancel={() => {
              if (dragStartedRef.current || dragArmingRef.current) {
                void commands.finishOverlayDrag(position);
              }
              setDragPhase("idle");
              resetDrag();
            }}
            onPointerEnter={handleBarPointerEnter}
            onPointerLeave={handleBarPointerLeave}
            onClick={handleBarClick}
          >
            <div
              className={`thought-bubble ${displayState} ${isWorking ? "working" : ""}`}
              role={state === "resting" ? "button" : "status"}
              tabIndex={state === "resting" ? 0 : -1}
              onFocus={() => showTooltip("dictate", BUBBLE_TOOLTIP_DELAY_MS)}
              onBlur={hideTooltip}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ")
                  handleBarClick();
              }}
            >
              <div className="bubble-row">
                {displayState === "ready" && <MicGlyph className="ready-mic" />}
                {hasToggleControls && (
                  <button
                    className="flow-action cancel visible"
                    aria-label={t("common.close")}
                    onMouseEnter={() =>
                      showTooltip("cancel", ACTION_TOOLTIP_DELAY_MS)
                    }
                    onMouseLeave={hideTooltip}
                    onFocus={() =>
                      showTooltip("cancel", ACTION_TOOLTIP_DELAY_MS)
                    }
                    onBlur={hideTooltip}
                    onPointerDown={(event) => event.stopPropagation()}
                    onClick={(event) => {
                      event.stopPropagation();
                      void commands.cancelOperation();
                    }}
                  >
                    <svg
                      width="12"
                      height="12"
                      viewBox="0 0 12 12"
                      aria-hidden="true"
                    >
                      <path d="M2 2 10 10M10 2 2 10" />
                    </svg>
                  </button>
                )}
                {(isListening || isWorking || state === "error") && waveform}
                {hasToggleControls && (
                  <button
                    className="flow-action finish visible"
                    aria-label={t("overlay.finishAndPaste", {
                      defaultValue: "Finish and paste",
                    })}
                    onMouseEnter={() =>
                      showTooltip("finish", ACTION_TOOLTIP_DELAY_MS)
                    }
                    onMouseLeave={hideTooltip}
                    onFocus={() =>
                      showTooltip("finish", ACTION_TOOLTIP_DELAY_MS)
                    }
                    onBlur={hideTooltip}
                    onPointerDown={(event) => event.stopPropagation()}
                    onClick={(event) => {
                      event.stopPropagation();
                      void commands.toggleOverlayDictation();
                    }}
                  >
                    <svg
                      width="12"
                      height="12"
                      viewBox="0 0 12 12"
                      aria-hidden="true"
                    >
                      <rect x="2" y="2" width="8" height="8" rx="1.25" />
                    </svg>
                  </button>
                )}
                {isWorking && loader}
              </div>
            </div>
          </div>
        )}
      </section>
    </main>
  );
};

export default RecordingOverlay;
