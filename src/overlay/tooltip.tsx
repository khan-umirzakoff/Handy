import { listen } from "@tauri-apps/api/event";
import React, { useEffect, useState } from "react";
import ReactDOM from "react-dom/client";
import { useTranslation } from "react-i18next";
import "@/i18n";
import "./Tooltip.css";

type TooltipKind = "dictate" | "cancel" | "finish";

const FlowBarTooltip: React.FC = () => {
  const { t } = useTranslation();
  // The tooltip webview is persistent and can paint once before its first
  // event arrives. Default to the resting-bar action so it never flashes the
  // toggle-only "Close" label during an idle hover.
  const [kind, setKind] = useState<TooltipKind>("dictate");
  const [revision, setRevision] = useState(0);

  useEffect(() => {
    let cleanup: (() => void) | undefined;
    void listen<TooltipKind>("flowbar-tooltip", (event) => {
      setKind(event.payload);
      setRevision((value) => value + 1);
    }).then((unlisten) => {
      cleanup = unlisten;
    });
    return () => cleanup?.();
  }, []);

  return (
    <main key={revision} className="flowbar-tooltip" role="tooltip">
      {kind === "dictate"
        ? t("overlay.clickToDictate", { defaultValue: "Click to dictate" })
        : kind === "cancel"
          ? t("common.close")
          : t("overlay.finishAndPaste", { defaultValue: "Finish and paste" })}
    </main>
  );
};

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <FlowBarTooltip />
  </React.StrictMode>,
);
