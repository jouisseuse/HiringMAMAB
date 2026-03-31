// path: src/components/Loading.jsx
import React, { useEffect, useMemo, useRef, useState } from "react";

/* ---------- helpers (self-contained, no external deps) ---------- */
const clamp = (n, lo, hi) => Math.min(hi, Math.max(lo, n));
const fmtTime = (ms) => {
  const s = Math.max(0, Math.ceil(ms / 1000));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return m > 0 ? `${m}:${String(r).padStart(2, "0")}` : `${r}s`;
};
const shuffle = (arr, rng = Math.random) => {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
};

/* ---------- content ---------- */
const SCRAMBLE_WORDS = [
  "algorithm","context","granular","iterator","optimize","reactive","compose","empathy","protocol","horizon"
];
const TYPING_SNIPPETS = [
  "Empirica experiments run in stages and rounds.",
  "Great UX turns waiting into a micro-break.",
  "Typing tests tune both speed and accuracy.",
  "Small games reduce perceived wait time.",
];
const TRIVIA = [
  {
    q: "Which improves perceived wait the most?",
    choices: [
      "A blank spinner",
      "Progress + tiny activities",
      "Random ads",
      "Looping animation",
    ],
    answer: 1,
    explain: "Visible progress + interactive mini-tasks work best.",
  },
  {
    q: "Privacy-friendly YouTube embed domain?",
    choices: ["youtube.com", "youtu.be", "youtube-nocookie.com"],
    answer: 2,
    explain: "Using youtube-nocookie reduces third-party cookies.",
  },
];

/* ---------- sub panes ---------- */
function SpinnerPane() {
  return (
    <div className="h-full w-full flex items-center justify-center py-6">
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
  );
}

function VideoPane({ youtubeId }) {
  if (!youtubeId) return null;
  const src = `https://www.youtube-nocookie.com/embed/${encodeURIComponent(
    youtubeId
  )}?rel=0&modestbranding=1&playsinline=1`;
  return (
    <div className="w-full aspect-video rounded-xl overflow-hidden shadow">
      <iframe
        className="w-full h-full"
        src={src}
        title="Video"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        referrerPolicy="no-referrer"
        allowFullScreen
        // sandbox restricts iframe permissions
        sandbox="allow-same-origin allow-scripts allow-presentation allow-popups"
      />
    </div>
  );
}

function TriviaPane() {
  const [idx, setIdx] = useState(0);
  const [picked, setPicked] = useState(null);
  const item = TRIVIA[idx % TRIVIA.length];
  const correct = picked === item.answer;

  return (
    <div className="w-full max-w-xl">
      <div className="text-lg font-semibold mb-3">{item.q}</div>
      <div className="space-y-2">
        {item.choices.map((c, i) => (
          <button
            key={i}
            onClick={() => setPicked(i)}
            className={`w-full text-left px-4 py-2 rounded-lg border transition
              ${picked === null ? "hover:bg-gray-100" : ""}
              ${
                picked !== null
                  ? i === item.answer
                    ? "bg-green-50 border-green-300"
                    : picked === i
                    ? "bg-red-50 border-red-300"
                    : "opacity-60"
                  : "border-gray-200"
              }`}
          >
            {c}
          </button>
        ))}
      </div>
      {picked !== null && (
        <div className="mt-3 text-sm text-gray-600">
          {correct ? "✅ Correct!" : "❌ Not quite."} {item.explain}
        </div>
      )}
      <div className="mt-4">
        <button
          onClick={() => {
            setIdx((v) => v + 1);
            setPicked(null);
          }}
          className="px-3 py-2 rounded-lg bg-gray-800 text-white hover:bg-black"
        >
          Next
        </button>
      </div>
    </div>
  );
}

function ScramblePane() {
  const [src] = useState(() => SCRAMBLE_WORDS[Math.floor(Math.random() * SCRAMBLE_WORDS.length)]);
  const [scrambled] = useState(() => shuffle(src.split("")).join(""));
  const [guess, setGuess] = useState("");
  const ok = guess.toLowerCase().trim() === src.toLowerCase();

  return (
    <div className="w-full max-w-xl">
      <div className="text-sm text-gray-500 mb-1">Unscramble the word</div>
      <div className="text-2xl font-bold tracking-widest mb-3 select-none">{scrambled}</div>
      <input
        value={guess}
        onChange={(e) => setGuess(e.target.value)}
        placeholder="Type the original word..."
        className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring"
      />
      <div className="mt-2 h-6">
        {ok ? <span className="text-green-600">✨ Nice!</span> : <span className="text-gray-400">Tip: it's a common tech word.</span>}
      </div>
    </div>
  );
}

function TypingPane() {
  const [text] = useState(() => TYPING_SNIPPETS[Math.floor(Math.random() * TYPING_SNIPPETS.length)]);
  const [val, setVal] = useState("");
  const [startAt, setStartAt] = useState(null);
  const [wpm, setWpm] = useState(null);

  useEffect(() => {
    if (val.length === 1 && !startAt) setStartAt(performance.now());
    if (val === text && startAt) {
      const ms = performance.now() - startAt;
      const words = text.trim().split(/\s+/).length;
      setWpm(Math.round((words / ms) * 60000));
    }
  }, [val, text, startAt]);

  return (
    <div className="w-full max-w-xl">
      <div className="text-sm text-gray-500 mb-1">Type this:</div>
      <div className="p-3 rounded-lg bg-gray-50 border mb-3 select-none">{text}</div>
      <textarea
        value={val}
        onChange={(e) => setVal(e.target.value)}
        rows={3}
        className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring"
        placeholder="Start typing..."
      />
      <div className="mt-2 text-sm h-6">
        {wpm ? <span className="text-green-700">WPM: {wpm}</span> : <span className="text-gray-400">Accuracy first. Speed follows.</span>}
      </div>
    </div>
  );
}

function ReadPane() {
  return (
    <div className="w-full max-w-xl space-y-2">
      <div className="text-lg font-semibold">Quick read: Why waiting rooms matter</div>
      <p className="text-gray-700">
        Good waiting UX sets expectations, offers small, optional tasks, and keeps users informed. A visible countdown
        and meaningful progress reduce anxiety and drop-offs.
      </p>
      <ul className="list-disc ml-5 text-gray-700">
        <li>Show time remaining (not just a spinner).</li>
        <li>Offer opt-in micro-activities.</li>
        <li>Keep content short, accessible, and safe for work.</li>
      </ul>
    </div>
  );
}

/* ---------- main component ---------- */
export function SelfLoading({
  estimatedMs = 30000,
  youtubeId,
  allowYouTube = true,
  games = ["trivia", "scramble", "typing", "read"],
  title = "Preparing the next round…",
  onExit, // optional
}) {
  const startAt = useRef(performance.now());
  const [now, setNow] = useState(performance.now());
  const [manualExit, setManualExit] = useState(false);

  // assemble activities; always include spinner
  const activities = useMemo(() => {
    const base = ["spinner"];
    if (allowYouTube && youtubeId) base.push("video");
    for (const g of games) base.push(g);
    return base;
  }, [allowYouTube, youtubeId, games]);

  // restore last tab
  const [mode, setMode] = useState(() => {
    try {
      const saved = localStorage.getItem("loading:last");
      return saved && activities.includes(saved) ? saved : activities[0];
    } catch {
      return activities[0];
    }
  });
  useEffect(() => {
    if (!activities.includes(mode)) setMode(activities[0]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activities]);
  useEffect(() => {
    try { localStorage.setItem("loading:last", mode); } catch {}
  }, [mode]);

  // ticker
  useEffect(() => {
    const id = setInterval(() => setNow(performance.now()), 200);
    return () => clearInterval(id);
  }, []);

  const elapsed = now - startAt.current;
  const rem = clamp(estimatedMs - elapsed, 0, estimatedMs);
  const pct = estimatedMs ? clamp(1 - rem / estimatedMs, 0, 1) : 0;

  // auto exit at 0
  useEffect(() => {
    if (rem <= 0 && !manualExit) {
      setManualExit(true);
      onExit && onExit();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rem]);

  return (
    <div className="min-h-[60vh] w-full flex items-center justify-center p-4">
      <div className="w-full max-w-4xl bg-white rounded-2xl shadow-xl p-6">
        <div className="flex items-center justify-between gap-3">
          <div className="text-xl font-bold">{title}</div>
          <div className="text-sm text-gray-500">~{fmtTime(rem)} left</div>
        </div>

        {/* progress */}
        <div className="mt-3 h-2 w-full bg-gray-100 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full"
            style={{ width: `${pct * 100}%`, background: "linear-gradient(to right, #6366f1, #22d3ee)" }}
            aria-label="progress"
          />
        </div>

        {/* tabs */}
        <div className="mt-4 flex flex-wrap gap-2">
          {activities.map((a) => (
            <button
              key={a}
              onClick={() => setMode(a)}
              className={`px-3 py-1.5 rounded-full border text-sm transition ${
                mode === a ? "bg-black text-white border-black" : "border-gray-300 hover:bg-gray-50"
              }`}
              aria-pressed={mode === a}
            >
              {a === "spinner" ? "Spinner" :
               a === "video" ? "Video" :
               a[0].toUpperCase() + a.slice(1)}
            </button>
          ))}
          <div className="ml-auto">
            <button
              onClick={() => { setManualExit(true); onExit && onExit(); }}
              className="px-3 py-1.5 rounded-full bg-gray-800 text-white hover:bg-black"
            >
              Continue
            </button>
          </div>
        </div>

        {/* content */}
        <div className="mt-5">
          {mode === "spinner" && <SpinnerPane />}
          {mode === "video" && <VideoPane youtubeId={youtubeId} />}
          {mode === "trivia" && <TriviaPane />}
          {mode === "scramble" && <ScramblePane />}
          {mode === "typing" && <TypingPane />}
          {mode === "read" && <ReadPane />}
        </div>

        <div className="mt-6 text-xs text-gray-500">
          Tip: Activities are optional. Your place in the experiment is reserved.
        </div>
      </div>
    </div>
  );
}