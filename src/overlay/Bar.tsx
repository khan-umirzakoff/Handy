// Shivir's dictation bar. Behaviour, sizes and timings follow Wispr Flow's bar
// as recorded in research/flow-pill-motion.md (section numbers in comments are
// that file's). The code is our own; nothing is copied from Flow.
//
// The native window is a fixed transparent stage (overlay.rs); everything here
// animates in CSS inside it, and we report the interactive rectangles so the
// rest of the stage stays click-through.

import { listen } from "@tauri-apps/api/event";
import { writeText } from "@tauri-apps/plugin-clipboard-manager";
import React, {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { Trans, useTranslation } from "react-i18next";
import { commands, events } from "@/bindings";
import type { DragInsets, Notice } from "@/bindings";
import { syncLanguageFromSettings } from "@/i18n";
import { noticeDisplay } from "@/lib/notices";
import "./bar.css";

type Edge = "bottom" | "left" | "right";
type BarState = "resting" | "ready" | "ptt" | "popo" | "processing" | "error";
type Toast =
  | { kind: "notice"; title: string; body: string }
  | { kind: "paste"; text: string };
type ToastStage =
  | "none"
  | "preparing"
  | "entering"
  | "visible"
  | "exiting"
  | "collapsing";
type TipKind = "dictate" | "finishHint" | "cancel" | "stop";
type DragPhase = "idle" | "dragging" | "returning";

const BARS = 10;
// §5.1: quadratic bulge 1 - d²/48 around the centre; travelling-wave phase 0.1 s × i.
const BAR_SCALE = Array.from({ length: BARS }, (_, i) => {
  const d = Math.abs((BARS - 1) / 2 - i);
  return Math.max(0, 1 - (d * d) / 48);
});
const BAR_DELAY = Array.from(
  { length: BARS },
  (_, i) => 0.1 * (i < Math.ceil(BARS / 2) ? i : i - BARS),
);
const WAVE_GAIN = 5; // §5.3 dictation gain
// shortcut: Flow feeds an adaptive dB-floor level; we lift the RMS of Handy's
// 16 spectrum buckets instead. Tune once real recordings are in.
const LEVEL_LIFT = 1.6;
const HOVER_LOCK_MS = 100; // §4.2
const TIP_AFTER_READY_MS = 150;
const TIP_ON_TARGET_MS = 50;
const DRAG_THRESHOLD_SQ = 25; // §6.1: 5 px
const DOCK_DELAY_MS = 300; // §6.3 step 7
const MORPH_MS = 420; // §7.2
const EXIT_TO_COLLAPSE_MS = 100;
const EXIT_TOTAL_MS = 520;
const REPLACE_MS = 120;
const PASTE_CARD_MS = 8000; // void-dictation card timeout
const COPIED_CLOSE_MS = 1200;
const READY_HIT_PAD = 12;
// Our addition: a click whose dictation never starts (no model, no mic) must
// not leave the local hands-free look on screen.
const CLICK_START_TIMEOUT_MS = 1500; // §1.1 layer 8: ::before inset -12px while ready

const fromNative = (s: string): BarState | null => {
  switch (s) {
    case "resting":
      return "resting";
    case "recording-ptt":
      return "ptt";
    case "recording-toggle":
    case "streaming":
      return "popo";
    case "transcribing":
    case "processing":
      return "processing";
    case "error":
      return "error";
    default:
      return null;
  }
};

const edgeOf = (v: string | undefined): Edge =>
  v === "left" || v === "right" ? v : "bottom";

const prettyHotkey = (binding: string | undefined) =>
  (binding ?? "")
    .split("+")
    .filter(Boolean)
    .map((k) => {
      const key = k.trim().toLowerCase();
      const names: Record<string, string> = {
        option: "⌥",
        alt: "⌥",
        command: "⌘",
        cmd: "⌘",
        control: "⌃",
        ctrl: "⌃",
        shift: "⇧",
        fn: "fn",
        space: "Space",
      };
      return names[key] ?? key.charAt(0).toUpperCase() + key.slice(1);
    })
    .join(" + ");

/** §4.3: the first hover change applies at once, later ones within 100 ms are coalesced. */
function useDebouncedHover(raw: boolean) {
  const [value, setValue] = useState(raw);
  const lockedUntil = useRef(0);
  const timer = useRef<number>();
  useEffect(() => {
    const apply = () => {
      setValue(raw);
      lockedUntil.current = performance.now() + HOVER_LOCK_MS;
    };
    const wait = lockedUntil.current - performance.now();
    window.clearTimeout(timer.current);
    if (wait <= 0) apply();
    else timer.current = window.setTimeout(apply, wait);
    return () => window.clearTimeout(timer.current);
  }, [raw]);
  return value;
}

const Bar: React.FC = () => {
  const { t } = useTranslation();
  const [edge, setEdge] = useState<Edge>("bottom");
  const [enabled, setEnabled] = useState(true);
  const [native, setNativeState] = useState<BarState>("resting");
  const nativeRef = useRef<BarState>("resting");
  const clickStartTimer = useRef<number>();
  const setNative = (next: BarState) => {
    nativeRef.current = next;
    setNativeState(next);
  };
  const [micActive, setMicActive] = useState(false);
  const [hotkey, setHotkey] = useState("");
  const [rawHover, setRawHover] = useState(false);
  const hovered = useDebouncedHover(rawHover);
  const [tip, setTip] = useState<{ kind: TipKind; anchor: HTMLElement } | null>(
    null,
  );
  const tipTimer = useRef<number>();

  const [toast, setToast] = useState<Toast | null>(null);
  const [stage, setStageState] = useState<ToastStage>("none");
  const stageRef = useRef<ToastStage>("none");
  const setStage = (next: ToastStage) => {
    stageRef.current = next;
    setStageState(next);
  };
  const [showContent, setShowContent] = useState(false);
  const [box, setBox] = useState<{ w: number; h: number } | null>(null);
  const [copied, setCopied] = useState(false);
  const toastTimers = useRef<number[]>([]);
  const measureRef = useRef<HTMLDivElement>(null);
  const toastRef = useRef<HTMLDivElement>(null);

  const [drag, setDrag] = useState<DragPhase>("idle");
  const [fullDisplay, setFullDisplay] = useState(false);
  const [insets, setInsets] = useState<DragInsets>({
    top: 0,
    bottom: 0,
    left: 0,
    right: 0,
  });
  const [dropEdge, setDropEdge] = useState<Edge>("bottom");
  const [ghost, setGhost] = useState({ x: -100, y: -100 });
  const dragOrigin = useRef<{ x: number; y: number } | null>(null);
  const [armed, setArmed] = useState(false);
  const suppressClick = useRef(false);

  const hotAreaRef = useRef<HTMLDivElement>(null);
  const waveRef = useRef<HTMLDivElement>(null);
  const levelTarget = useRef(0);
  const levelCur = useRef(0);
  const levelFrame = useRef(0);

  const state: BarState = native === "resting" && hovered ? "ready" : native;
  const toastActive = stage !== "none";

  // ---- native events -------------------------------------------------------
  useEffect(() => {
    let alive = true;
    const offs: Array<() => void> = [];
    const on = <T,>(event: string, fn: (payload: T) => void) =>
      listen<T>(event, (e) => fn(e.payload)).then((off) =>
        alive ? offs.push(off) : off(),
      );

    void (async () => {
      await syncLanguageFromSettings();
      const s = await commands.getAppSettings();
      if (alive && s.status === "ok") {
        setEdge(edgeOf(s.data.overlay_position));
        setEnabled(s.data.overlay_style !== "none");
        setHotkey(prettyHotkey(s.data.bindings?.transcribe?.current_binding));
      }
    })();

    const applyNative = (next: BarState) => {
      const wasLive =
        nativeRef.current === "ptt" || nativeRef.current === "popo";
      if (next === "ptt" || next === "popo") {
        if (!wasLive) setMicActive(false);
      } else {
        setMicActive(false);
        levelTarget.current = 0;
      }
      setNative(next);
      // Every session state replaces a card, so a new take never sits under an old one.
      closeToast();
    };
    void on<string>("show-overlay", (s) => {
      window.clearTimeout(clickStartTimer.current);
      setEnabled(true);
      const next = fromNative(s);
      if (next) applyNative(next);
    });
    // A Live take reports stop and polishing here instead of show-overlay.
    void events.streamPhaseEvent
      .listen((e) => {
        if (e.payload.phase === "working") applyNative("processing");
      })
      .then((off) => (alive ? offs.push(off) : off()));
    void on("hide-overlay", () => setEnabled(false));
    void on("recording-ready", () => setMicActive(true));
    void on<string>("overlay-position-changed", (e) => setEdge(edgeOf(e)));
    void on<boolean>("flowbar-native-hover", (h) => setRawHover(h));
    void on<number[]>("mic-level", (buckets) => {
      const v = buckets.filter(Number.isFinite);
      const rms = v.length
        ? Math.sqrt(v.reduce((a, b) => a + b * b, 0) / v.length)
        : 0;
      levelTarget.current = Math.min(1, rms * LEVEL_LIFT);
      startLevelLoop();
    });
    void on<Notice>("overlay-notice", (n) => {
      const d = noticeDisplay(n.code);
      openToast({
        kind: "notice",
        title: t(d.title),
        body: d.explanation?.(t) ?? n.detail ?? "",
      });
    });
    void on<string>("paste-recovery", (text) => {
      setCopied(false);
      openToast({ kind: "paste", text });
    });

    return () => {
      alive = false;
      offs.forEach((off) => off());
    };
    // Listeners read state through refs/setters only.
  }, []);

  // ---- §5.3 waveform level: rAF low-pass written to --audio-scale -----------
  const startLevelLoop = useCallback(() => {
    if (levelFrame.current) return;
    const tick = () => {
      const cur =
        Math.floor(
          100 * (0.85 * levelCur.current + 0.15 * levelTarget.current),
        ) / 100;
      levelCur.current = cur;
      waveRef.current?.style.setProperty(
        "--audio-scale",
        String(Math.max(1, WAVE_GAIN * cur)),
      );
      if (levelTarget.current === 0 && cur === 0) {
        levelFrame.current = 0;
        return;
      }
      levelFrame.current = requestAnimationFrame(tick);
    };
    levelFrame.current = requestAnimationFrame(tick);
  }, []);
  useEffect(() => () => cancelAnimationFrame(levelFrame.current), []);

  // ---- tooltip (§8) --------------------------------------------------------
  const hideTip = useCallback(() => {
    window.clearTimeout(tipTimer.current);
    setTip(null);
  }, []);
  const showTipSoon = useCallback(
    (kind: TipKind, anchor: HTMLElement | null, delay: number) => {
      window.clearTimeout(tipTimer.current);
      if (!anchor) return;
      tipTimer.current = window.setTimeout(
        () => setTip({ kind, anchor }),
        tip ? 0 : delay,
      );
    },
    [tip],
  );
  const bubbleRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (state === "ready")
      showTipSoon("dictate", bubbleRef.current, TIP_AFTER_READY_MS);
    else if (state === "resting" || state === "processing" || state === "error")
      hideTip();
    // Only the state transition schedules the bubble tooltip.
  }, [state]);

  // ---- clicks --------------------------------------------------------------
  const onBubbleClick = () => {
    if (suppressClick.current) {
      suppressClick.current = false;
      return;
    }
    hideTip();
    if (state === "ready" || state === "resting") {
      setNative("popo"); // §4.5: local jump, the engine follows
      window.clearTimeout(clickStartTimer.current);
      clickStartTimer.current = window.setTimeout(() => {
        if (nativeRef.current === "popo") setNative("resting");
      }, CLICK_START_TIMEOUT_MS);
      void commands.toggleOverlayDictation();
    } else if (state === "ptt" || state === "popo") {
      void commands.toggleOverlayDictation();
    }
  };
  const onCancel = (e: React.MouseEvent) => {
    e.stopPropagation();
    hideTip();
    setNative("resting"); // §4.8: shrink at once, don't wait for the engine
    void commands.cancelOperation();
  };
  const onStop = (e: React.MouseEvent) => {
    e.stopPropagation();
    hideTip();
    void commands.toggleOverlayDictation();
  };

  // ---- drag and dock (§6) --------------------------------------------------
  const anchorFor = useCallback(
    (e: Edge) => {
      const w = window.innerWidth;
      const h = window.innerHeight;
      const midY = (h + insets.top - insets.bottom) / 2;
      if (e === "left") return { x: insets.left + 18, y: midY };
      if (e === "right") return { x: w - insets.right - 18, y: midY };
      return {
        x: w / 2 - (insets.right - insets.left) / 2,
        y: h - insets.bottom - 23,
      };
    },
    [insets],
  );
  const nearestEdge = useCallback(
    (x: number, y: number): Edge =>
      (["bottom", "left", "right"] as Edge[]).reduce((best, e) => {
        const a = anchorFor(e);
        const b = anchorFor(best);
        return (a.x - x) ** 2 + (a.y - y) ** 2 < (b.x - x) ** 2 + (b.y - y) ** 2
          ? e
          : best;
      }, "bottom" as Edge),
    [anchorFor],
  );

  const onWrapperMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0 || (e.target as HTMLElement).closest("button")) return;
    dragOrigin.current = { x: e.screenX, y: e.screenY };
    setArmed(true);
  };
  useEffect(() => {
    const move = (e: MouseEvent) => {
      const o = dragOrigin.current;
      // The release was lost (it happened off the stage): end the press here, or
      // the whole-stage catch rect keeps swallowing clicks.
      if (o && e.buttons === 0) {
        up();
        return;
      }
      if (o && drag === "idle") {
        if (
          (e.screenX - o.x) ** 2 + (e.screenY - o.y) ** 2 <=
          DRAG_THRESHOLD_SQ
        )
          return;
        hideTip();
        suppressClick.current = true;
        setDrag("dragging");
        requestAnimationFrame(() =>
          requestAnimationFrame(() => {
            void commands.beginOverlayDrag().then((r) => {
              if (r.status === "ok") setInsets(r.data);
            });
          }),
        );
        return;
      }
      if (drag === "dragging" && fullDisplay) {
        setGhost({ x: e.clientX, y: e.clientY });
        setDropEdge(nearestEdge(e.clientX, e.clientY));
      }
    };
    const up = () => {
      dragOrigin.current = null;
      setArmed(false);
      if (drag !== "dragging") return;
      const target = fullDisplay ? dropEdge : edge;
      setDrag("returning");
      requestAnimationFrame(() => setGhost(anchorFor(target)));
      window.setTimeout(() => {
        void commands.finishOverlayDrag(target).then(() => {
          setEdge(target);
          setDrag("idle");
          setFullDisplay(false);
        });
      }, DOCK_DELAY_MS);
    };
    document.addEventListener("mousemove", move);
    document.addEventListener("mouseup", up);
    return () => {
      document.removeEventListener("mousemove", move);
      document.removeEventListener("mouseup", up);
    };
  }, [drag, fullDisplay, dropEdge, edge, anchorFor, nearestEdge, hideTip]);
  useEffect(() => {
    const onResize = () =>
      setFullDisplay(
        window.innerWidth >= window.screen.availWidth - 2 &&
          window.innerHeight >= window.screen.availHeight - 2,
      );
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // ---- toast morph (§7) ----------------------------------------------------
  const clearToastTimers = () => {
    toastTimers.current.forEach((id) => window.clearTimeout(id));
    toastTimers.current = [];
  };
  const later = (ms: number, fn: () => void) =>
    toastTimers.current.push(window.setTimeout(fn, ms));

  const openToast = (next: Toast) => {
    clearToastTimers();
    hideTip();
    if (stageRef.current === "visible") {
      // §7.2 step 6: replace in place; the box re-measures and re-morphs.
      setShowContent(false);
      later(REPLACE_MS, () => {
        setToast(next);
        setShowContent(true);
      });
      return;
    }
    setToast(next);
    setShowContent(false);
    setBox(null);
    setStage("preparing");
  };
  const closeToast = () => {
    const cur = stageRef.current;
    if (cur === "none" || cur === "exiting" || cur === "collapsing") return;
    clearToastTimers();
    setShowContent(false);
    setStage("exiting");
    later(EXIT_TO_COLLAPSE_MS, () => setStage("collapsing"));
    later(EXIT_TOTAL_MS, () => {
      setStage("none");
      setToast(null);
      setBox(null);
    });
  };

  // Measure offscreen, then grow from the rest box on the next-but-one frame.
  useLayoutEffect(() => {
    const el = measureRef.current;
    if (!toast || !el) return;
    const r = el.getBoundingClientRect();
    setBox({ w: Math.ceil(r.width), h: Math.ceil(r.height) });
    if (stageRef.current === "preparing") {
      requestAnimationFrame(() =>
        requestAnimationFrame(() => {
          setStage("entering");
          later(MORPH_MS, () => {
            setStage("visible");
            setShowContent(true);
          });
        }),
      );
    }
  }, [toast, copied]);

  // Auto close: the copy card after 8 s unless hovered; notices are timed by Rust.
  const [toastHovered, setToastHovered] = useState(false);
  useEffect(() => {
    if (toast?.kind !== "paste" || stage !== "visible" || toastHovered) return;
    const id = window.setTimeout(() => dismissToast(), PASTE_CARD_MS);
    return () => window.clearTimeout(id);
  }, [toast, stage, toastHovered]);

  const dismissToast = () => {
    closeToast();
    if (toast?.kind === "notice") void commands.dismissOverlayNotice();
    else void commands.dismissOverlayCard();
  };
  const copyPaste = async () => {
    if (toast?.kind !== "paste" || copied) return;
    try {
      await writeText(toast.text);
      setCopied(true);
      // A toast timer, so a newer card or take cancels it.
      later(COPIED_CLOSE_MS, dismissToast);
    } catch {
      // Keep the card open: the text stays readable on screen.
    }
  };

  // ---- catch rects for the native hit test ---------------------------------
  const lastRects = useRef("");
  const reportRects = useCallback(() => {
    const rects: { x: number; y: number; width: number; height: number }[] = [];
    if (drag !== "idle" || armed) {
      rects.push({
        x: 0,
        y: 0,
        width: window.innerWidth,
        height: window.innerHeight,
      });
    } else {
      const hot = hotAreaRef.current?.getBoundingClientRect();
      if (hot && !toastActive) {
        const pad = state === "ready" ? READY_HIT_PAD : 0;
        rects.push({
          x: hot.left - pad,
          y: hot.top - pad,
          width: hot.width + 2 * pad,
          height: hot.height + 2 * pad,
        });
      }
      const card = toastRef.current?.getBoundingClientRect();
      if (card && toastActive && box) {
        rects.push({
          x: card.left,
          y: card.top,
          width: card.width,
          height: card.height,
        });
      }
    }
    const key = JSON.stringify(rects);
    if (key === lastRects.current) return;
    lastRects.current = key;
    void commands.reportBarCatchRects(rects);
  }, [drag, armed, state, toastActive, box]);
  useEffect(() => {
    reportRects();
    // Again once the 100 ms pill and 420 ms toast transitions have settled.
    const a = window.setTimeout(reportRects, 120);
    const b = window.setTimeout(reportRects, MORPH_MS + 40);
    return () => {
      window.clearTimeout(a);
      window.clearTimeout(b);
    };
  }, [reportRects, edge, stage, fullDisplay]);

  // ---- render ----------------------------------------------------------------
  if (!enabled) return null;

  const waveLive = micActive && (state === "ptt" || state === "popo");
  const showMic = state === "ready";
  const tipText =
    tip?.kind === "dictate" ? (
      <>
        {t("overlay.dictate")} <b className="keytip">{hotkey}</b>
      </>
    ) : tip?.kind === "finishHint" ? (
      <Trans
        i18nKey="overlay.pressToFinish"
        values={{ hotkey }}
        components={{ key: <b className="keytip" /> }}
      />
    ) : tip?.kind === "cancel" ? (
      t("overlay.cancel")
    ) : tip?.kind === "stop" ? (
      t("overlay.finishAndPaste")
    ) : null;

  const toastBody =
    toast?.kind === "notice" ? (
      <div className="toast-notice">
        <strong>{toast.title}</strong>
        {toast.body && <p>{toast.body}</p>}
      </div>
    ) : toast?.kind === "paste" ? (
      <div className="toast-paste">
        <div className="paste-head">
          <span className="paste-hint">{t("overlay.pasteRecoveryHint")}</span>
          <button
            className="paste-close"
            aria-label={t("common.close")}
            onClick={dismissToast}
          >
            <svg viewBox="0 0 20 20" aria-hidden="true">
              <circle
                className={`ring ${stage === "visible" && !toastHovered ? "run" : ""}`}
                cx="10"
                cy="10"
                r="8.5"
              />
              <path d="M7 7l6 6M13 7l-6 6" />
            </svg>
          </button>
        </div>
        <p className="paste-text">{toast.text}</p>
        <div className="paste-actions">
          <button className="paste-copy" onClick={copyPaste}>
            {copied
              ? `${t("overlay.pasteRecoveryCopied")} ✓`
              : t("overlay.pasteRecoveryCopy")}
          </button>
        </div>
      </div>
    ) : null;

  return (
    <div className="bar-root" data-edge={edge}>
      {drag !== "idle" && fullDisplay && (
        <>
          <div
            className={`drag-overlay ${drag === "returning" ? "exiting" : ""}`}
          />
          {(["bottom", "left", "right"] as Edge[]).map((z) => (
            <div
              key={z}
              className={`drop-zone ${z} ${dropEdge === z ? "hovered" : ""} ${
                drag === "returning" ? "exiting" : ""
              }`}
              style={
                {
                  "--inset-top": `${insets.top}px`,
                  "--inset-bottom": `${insets.bottom}px`,
                  "--inset-left": `${insets.left}px`,
                  "--inset-right": `${insets.right}px`,
                } as React.CSSProperties
              }
            />
          ))}
        </>
      )}
      {drag !== "idle" && (
        <div
          className={`drag-ghost ${drag === "returning" ? "returning" : ""}`}
          style={{ left: ghost.x, top: ghost.y }}
        >
          <svg viewBox="0 0 16 16" aria-hidden="true">
            <rect x="2" y="6" width="2" height="4" rx="1" />
            <rect x="5.5" y="3" width="2" height="10" rx="1" />
            <rect x="9" y="5" width="2" height="6" rx="1" />
            <rect x="12.5" y="7" width="2" height="2" rx="1" />
          </svg>
        </div>
      )}

      <div className={`island ${drag !== "idle" ? "hidden" : ""}`}>
        <div className="indicator-layer">
          <div className={`hover-area ${toastActive ? "suppressed" : ""}`}>
            <div
              ref={hotAreaRef}
              className="hot-area"
              onMouseEnter={() => setRawHover(true)}
              onMouseLeave={() => {
                setRawHover(false);
                hideTip();
              }}
            >
              <div className="bubble-wrapper" onMouseDown={onWrapperMouseDown}>
                <div
                  ref={bubbleRef}
                  className={`bubble ${state}`}
                  role="button"
                  aria-label={t("overlay.dictate")}
                  onClick={onBubbleClick}
                  onMouseEnter={(e) => {
                    if (state === "ready")
                      showTipSoon("dictate", e.currentTarget, TIP_ON_TARGET_MS);
                    else if (state === "popo")
                      showTipSoon(
                        "finishHint",
                        e.currentTarget,
                        TIP_ON_TARGET_MS,
                      );
                  }}
                >
                  <button
                    className="bar-btn cancel"
                    aria-label={t("overlay.cancel")}
                    tabIndex={state === "popo" ? 0 : -1}
                    onClick={onCancel}
                    onMouseEnter={(e) =>
                      showTipSoon("cancel", e.currentTarget, TIP_ON_TARGET_MS)
                    }
                  >
                    <svg viewBox="0 0 12 12" aria-hidden="true">
                      <path d="M3 3l6 6M9 3l-6 6" />
                    </svg>
                  </button>
                  {showMic ? (
                    <div className="wave mic">
                      <svg
                        className="mic-icon"
                        viewBox="0 0 16 16"
                        aria-hidden="true"
                      >
                        <rect x="5.5" y="1.5" width="5" height="8.5" rx="2.5" />
                        <path d="M3.5 7.5a4.5 4.5 0 0 0 9 0M8 12v2.5" />
                      </svg>
                    </div>
                  ) : (
                    <div ref={waveRef} className={`wave ${state}`}>
                      {BAR_SCALE.map((scale, i) => (
                        <i
                          key={i}
                          className={`wbar ${micActive ? "live" : ""} ${
                            waveLive ? "animate" : ""
                          }`}
                          style={
                            {
                              "--bar-height-scale": scale,
                              animationDelay: `${BAR_DELAY[i]}s`,
                            } as React.CSSProperties
                          }
                        />
                      ))}
                    </div>
                  )}
                  {state === "processing" && (
                    <span className="spinner" aria-hidden="true">
                      {Array.from({ length: 8 }, (_, i) => (
                        <i
                          key={i}
                          style={{
                            transform: `rotate(${45 * i}deg) translate(0, -6px)`,
                            animationDelay: `${-(7 - i) * 0.1375}s`,
                          }}
                        />
                      ))}
                    </span>
                  )}
                  <button
                    className="bar-btn stop"
                    aria-label={t("overlay.finishAndPaste")}
                    tabIndex={state === "popo" ? 0 : -1}
                    onClick={onStop}
                    onMouseEnter={(e) =>
                      showTipSoon("stop", e.currentTarget, TIP_ON_TARGET_MS)
                    }
                  >
                    <svg viewBox="0 0 12 12" aria-hidden="true">
                      <rect x="3" y="3" width="6" height="6" rx="1.2" />
                    </svg>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="notification-layer">
          {toast && (
            <div
              ref={toastRef}
              className={`morph ${stage}`}
              style={
                box && stage !== "preparing" && stage !== "collapsing"
                  ? {
                      width: box.w,
                      height: box.h,
                      borderRadius:
                        toast.kind === "notice" && !toast.body ? 20 : 24,
                    }
                  : undefined
              }
              onMouseEnter={() => setToastHovered(true)}
              onMouseLeave={() => setToastHovered(false)}
              onClick={toast.kind === "notice" ? dismissToast : undefined}
            >
              <div className={`morph-content ${showContent ? "shown" : ""}`}>
                {toastBody}
              </div>
            </div>
          )}
          {toast && (
            <div ref={measureRef} className="morph-measure" aria-hidden="true">
              {toastBody}
            </div>
          )}
        </div>
      </div>

      <Tooltip tip={tip} edge={edge}>
        {tipText}
      </Tooltip>
    </div>
  );
};

/** §8: one tooltip, placed away from the edge, gliding along the bar's axis. */
const Tooltip: React.FC<{
  tip: { kind: TipKind; anchor: HTMLElement } | null;
  edge: Edge;
  children: React.ReactNode;
}> = ({ tip, edge, children }) => {
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  const shownBefore = useRef(false);
  useLayoutEffect(() => {
    if (!tip) {
      shownBefore.current = false;
      return;
    }
    const r = tip.anchor.getBoundingClientRect();
    const gap = tip.kind === "dictate" || tip.kind === "finishHint" ? 4 : 10;
    if (edge === "bottom") setPos({ x: r.left + r.width / 2, y: r.top - gap });
    else if (edge === "left")
      setPos({ x: r.right + gap, y: r.top + r.height / 2 });
    else setPos({ x: r.left - gap, y: r.top + r.height / 2 });
  }, [tip, edge]);
  const animate = shownBefore.current;
  useEffect(() => {
    if (tip) shownBefore.current = true;
  }, [tip]);
  return (
    <div
      className={`tooltip ${tip && pos ? "visible" : ""} ${animate ? "glide" : ""}`}
      data-edge={edge}
      role="tooltip"
      style={pos ? { left: pos.x, top: pos.y } : undefined}
    >
      {children}
    </div>
  );
};

export default Bar;
