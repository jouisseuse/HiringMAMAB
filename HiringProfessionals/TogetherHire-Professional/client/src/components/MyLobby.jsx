// path: src/lobby/MyLobby.jsx
import React, { useMemo, useState } from "react";
import { usePlayer, usePlayers } from "@empirica/core/player/classic/react";
import { LoadingOverlay } from "../components/LoadingOverlay.jsx";

// Joke trivia (answers are 0-based)
const TRIVIA = [
  {
    q: "Why don’t eggs tell jokes?",
    choices: [
      "They are shy",
      "They would crack up",
      "They have no sense of humor",
      "They are afraid of chickens",
    ],
    answer: 1,
    explain: "“Crack up” = laugh hard / crack (pun).",
  },
  {
    q: "Why did the math book look sad?",
    choices: [
      "It failed an exam",
      "It lost its cover",
      "It had too many problems",
      "It was bad at numbers",
    ],
    answer: 2,
    explain: "“Problems” = exercises and troubles (pun).",
  },
  {
    q: "Why don’t skeletons fight each other?",
    choices: [
      "They are too polite",
      "They don’t like violence",
      "They don’t have the guts",
      "They are already dead",
    ],
    answer: 2,
    explain: "“Guts” = courage / internal organs (pun).",
  },
  {
    q: "Why did the computer go to the doctor?",
    choices: [
      "It had a virus",
      "It was too slow",
      "The screen was broken",
      "The keyboard stopped working",
    ],
    answer: 0,
    explain: "Computers can get a “virus”.",
  },
  {
    q: "Why did the scarecrow get a promotion?",
    choices: [
      "He worked overtime",
      "He scared people well",
      "He was outstanding in his field",
      "He dressed professionally",
    ],
    answer: 2,
    explain: "“Outstanding in his field” = excellent / standing in a field.",
  },
  {
    q: "Why don’t we ever tell secrets on a farm?",
    choices: [
      "Animals are noisy",
      "The walls are thin",
      "The potatoes have eyes",
      "Farmers talk too much",
    ],
    answer: 2,
    explain: "Potatoes have “eyes” (sprouts).",
  },
  {
    q: "Why did the student eat his homework?",
    choices: [
      "He was hungry",
      "The teacher told him to",
      "The dog dared him",
      "The teacher said it was a piece of cake",
    ],
    answer: 3,
    explain: "“A piece of cake” = very easy; taken literally.",
  },
  {
    q: "Why did the bicycle fall over?",
    choices: [
      "The road was slippery",
      "It was broken",
      "It was too tired",
      "It was two-tired",
    ],
    answer: 3,
    explain: "“Two-tired” pun on “too tired”.",
  },
  {
    q: "Why did the coffee file a police report?",
    choices: [
      "It was spilled",
      "It was burned",
      "It was mugged",
      "It was too hot",
    ],
    answer: 2,
    explain: "“Mugged” = robbed; “mug” = a cup.",
  },
  {
    q: "Why don’t elevators ever tell jokes?",
    choices: [
      "They’re too serious",
      "They take things to another level",
      "They have bad timing",
      "They’re afraid of letting people down",
    ],
    answer: 3,
    explain: "“Letting people down” = disappointing / going down.",
  },
];

function shuffle(arr, rng = Math.random) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function TriviaCard() {
  const order = useMemo(
    () => shuffle([...Array(TRIVIA.length)].map((_, i) => i)),
    []
  );
  const [idx, setIdx] = useState(0);
  const [picked, setPicked] = useState(null);
  const [score, setScore] = useState(0);

  const total = order.length;
  const done = idx >= total;
  const item = !done ? TRIVIA[order[idx]] : null;
  const isRight = picked !== null && picked === item?.answer;
  const percent = done ? 1 : idx / total;

  return (
    <div className="w-full max-w-2xl mx-auto mt-6">
      <div className="rounded-2xl p-[1px] bg-gradient-to-r from-indigo-300 via-sky-300 to-cyan-300">
        <div className="rounded-2xl bg-white">
          <div className="p-5">
            {/* Header */}
            <div className="flex items-center justify-between">
              <div className="text-sm font-medium text-gray-500">
                Trivia (optional)
              </div>
              <div className="text-xs px-2 py-1 rounded-full bg-gray-100 text-gray-600">
                {done ? "Completed" : `Question ${idx + 1} of ${total}`}
              </div>
            </div>

            {/* Progress bar */}
            <div className="mt-3 h-2 w-full bg-gray-100 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full"
                style={{
                  width: `${Math.round(percent * 100)}%`,
                  background:
                    "linear-gradient(to right, #6366f1, #22d3ee, #06b6d4)",
                }}
                aria-label="progress"
              />
            </div>

            {!done ? (
              <>
                {/* Question */}
                <div className="mt-4 text-xl font-semibold text-gray-900 leading-snug">
                  {item.q}
                </div>

                {/* Choices */}
                <div className="mt-4 space-y-2">
                  {item.choices.map((text, i) => {
                    const selected = picked === i;
                    const correct = picked !== null && i === item.answer;
                    const wrong = picked !== null && selected && i !== item.answer;

                    const base =
                      "w-full text-left px-4 py-3 rounded-xl border transition flex items-start gap-3";
                    const idle =
                      "border-gray-200 hover:bg-gray-50 active:scale-[0.99]";
                    const ok = "bg-green-50 border-green-300";
                    const bad = "bg-red-50 border-red-300";
                    const dim = "opacity-70";

                    return (
                      <button
                        key={i}
                        onClick={() => setPicked(i)}
                        className={[
                          base,
                          picked === null ? idle : "",
                          correct ? ok : "",
                          wrong ? bad : "",
                          picked !== null && !correct && !wrong ? dim : "",
                        ].join(" ")}
                      >
                        <span
                          className={[
                            "inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-full border text-sm font-semibold",
                            selected
                              ? "bg-black text-white border-black"
                              : "bg-white text-gray-700 border-gray-300",
                          ].join(" ")}
                        >
                          {String.fromCharCode(65 + i)}
                        </span>
                        <span className="text-gray-900">{text}</span>
                      </button>
                    );
                  })}
                </div>

                {/* Feedback + Controls */}
                <div className="mt-4 flex items-center justify-between">
                  <div className="min-h-[24px] text-sm text-gray-600">
                    {picked === null
                      ? "Pick one answer."
                      : isRight
                      ? "✅ Correct!"
                      : "❌ Not quite."}{" "}
                    {picked !== null && (
                      <span className="text-gray-500">{item.explain}</span>
                    )}
                  </div>

                  <button
                    onClick={() => {
                      if (picked !== null && isRight) setScore((v) => v + 1);
                      setPicked(null);
                      setIdx((v) => v + 1);
                    }}
                    disabled={picked === null}
                    className={[
                      "px-4 py-2 rounded-xl text-white font-medium shadow",
                      picked === null
                        ? "bg-gray-300 cursor-not-allowed"
                        : "bg-gray-900 hover:bg-black",
                    ].join(" ")}
                  >
                    Next
                  </button>
                </div>
              </>
            ) : (
              <div className="mt-4">
                <div className="text-xl font-semibold">All done!</div>
                <div className="mt-1 text-gray-700">
                  You scored{" "}
                  <span className="font-semibold">
                    {score} / {total}
                  </span>
                  .
                </div>
                <button
                  onClick={() => {
                    setIdx(0);
                    setPicked(null);
                    setScore(0);
                  }}
                  className="mt-4 px-4 py-2 rounded-xl bg-gray-900 text-white font-medium hover:bg-black shadow"
                >
                  Restart
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export function MyLobby() {
  const player = usePlayer();
  const players = usePlayers() || [];

  const treatment = useMemo(() => player?.get("treatment") || {}, [player]);
  const estimatedMs = Number.isFinite(Number(treatment?.lobbyDurationSec))
    ? Math.max(0, Number(treatment.lobbyDurationSec) * 1000)
    : undefined;

  const intro =
    "We’re waiting for a few more participants to join. Thanks for your patience. Let’s do some trivia and see how well you do.";


  return (
    <div className="min-h-[60vh] w-full flex items-center justify-center p-4">
      <div className="w-full max-w-3xl">
        <div className="mb-4 text-center">
          <div className="text-2xl font-bold">Waiting Room</div>
          <div className="text-gray-600">
            We’re waiting for a few more participants to join. Thanks for your patience.
          </div>
        </div>
        <LoadingOverlay
          title={<span>Let’s do some trivia and see how well you do🤓<br/>Preparing the game…</span>}
          estimatedMs={estimatedMs}
        />
        <TriviaCard />

      </div>
    </div>
  );
}