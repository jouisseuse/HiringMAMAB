import React, { useEffect, useRef, useState } from "react";

// Compact loading overlay; accepts estimatedMs to show countdown and progress bar.
export function LoadingOverlay({ title = "Loading…", estimatedMs }) {
  const startAt = useRef(performance.now());
  const [now, setNow] = useState(performance.now());

  useEffect(() => {
    let raf;
    const tick = () => {
      setNow(performance.now());
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  const elapsed = now - startAt.current;
  const rem = typeof estimatedMs === "number" ? Math.max(0, estimatedMs - elapsed) : undefined;
  const pct = typeof estimatedMs === "number" && estimatedMs > 0 ? Math.min(1, Math.max(0, 1 - rem / estimatedMs)) : undefined;
  const fmt = (ms) => {
    const s = Math.ceil(ms / 1000);
    const m = Math.floor(s / 60);
    const r = s % 60;
    return m > 0 ? `${m}:${String(r).padStart(2, "0")}` : `${r}s`;
  };

  return (
    <div className="h-full w-full flex items-center justify-center p-4">
      <div className="w-full max-w-md bg-white rounded-2xl shadow p-6">
        <div className="text-lg font-semibold text-center">{title}</div>

        {/* SVG spinner */}
        <div className="mt-4 flex items-center justify-center">
          <svg
            width="44"
            height="44"
            viewBox="0 0 44 44"
            xmlns="http://www.w3.org/2000/svg"
            className="text-empirica-200 stroke-current"
          >
            <g fill="none" fillRule="evenodd" strokeWidth="2">
              <circle cx="22" cy="22" r="1">
                <animate attributeName="r" begin="0s" dur="1.8s" values="1; 20" calcMode="spline" keyTimes="0; 1" keySplines="0.165, 0.84, 0.44, 1" repeatCount="indefinite" />
                <animate attributeName="stroke-opacity" begin="0s" dur="1.8s" values="1; 0" calcMode="spline" keyTimes="0; 1" keySplines="0.3, 0.61, 0.355, 1" repeatCount="indefinite" />
              </circle>
              <circle cx="22" cy="22" r="1">
                <animate attributeName="r" begin="-0.9s" dur="1.8s" values="1; 20" calcMode="spline" keyTimes="0; 1" keySplines="0.165, 0.84, 0.44, 1" repeatCount="indefinite" />
                <animate attributeName="stroke-opacity" begin="-0.9s" dur="1.8s" values="1; 0" calcMode="spline" keyTimes="0; 1" keySplines="0.3, 0.61, 0.355, 1" repeatCount="indefinite" />
              </circle>
            </g>
          </svg>
        </div>

        {/* Optional: countdown and progress bar */}
        {typeof rem === "number" && (
          <>
            <div className="mt-3 text-center text-sm text-gray-600">~{fmt(rem)} left</div>
            <div className="mt-2 h-2 w-full bg-gray-100 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full"
                style={{ width: `${Math.round((pct || 0) * 100)}%`, background: "linear-gradient(to right, #6366f1, #22d3ee)" }}
                aria-label="progress"
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
}
