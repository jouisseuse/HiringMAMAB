// path: src/stages/Choice.jsx
import { usePlayer, usePlayers, useRound, useGame, useStageTimer } from "@empirica/core/player/classic/react";
import React, { useEffect, useMemo, useState } from "react";
import { Button } from "../components/Button";
import "@unocss/reset/tailwind-compat.css";

function bernoulliRandom(p) {
  return Math.random() < p ? 1 : 0;
}


// Rotate array by offset
function rotate(arr, offset) {
  const n = arr.length; if (!n) return arr;
  const k = ((offset % n) + n) % n;
  return [...arr.slice(k), ...arr.slice(0, k)];
}

export function Choice() {
  const player = usePlayer();
  const players = usePlayers(); // kept for parity
  const round = useRound();
  const game = useGame();
  const chatEnabled = round.get("chatEnabled");
  const isTutorial = round.get("isTutorial");
  const timer = useStageTimer();

  const [showWarning, setShowWarning] = useState(false);
  const [showWarningModal, setShowWarningModal] = useState(false);

  // 10 universities (stable keys)
  const universities = useMemo(
    () => [
      "Trinity College Dublin",
      "The University of Western Australia",
      "University of Glasgow",
      "Heidelberg University",
      "University of Adelaide",
      "University of Leeds",
      "University of Southampton",
      "University of Sheffield",
      "University of Nottingham",
      "Karlsruhe Institute of Technology",
    ],
    []
  );

  const PROBABILITIES = useMemo(
    () => ({
      "Trinity College Dublin": 0.9,
      "The University of Western Australia": 0.3,
      "University of Glasgow": 0.7,
      "Heidelberg University": 0.1,
      "University of Adelaide": 0.5,
      "University of Leeds": 0.9,
      "University of Southampton": 0.3,
      "University of Sheffield": 0.5,
      "University of Nottingham": 0.7,
      "Karlsruhe Institute of Technology": 0.1,
    }),
    []
  );

  // Participant order index (stable across clients): sort by id
  const participantIndex = useMemo(() => {
    const arr = [...(players || [])].sort((a, b) => String(a.id).localeCompare(String(b.id)));
    const idx = arr.findIndex((p) => p.id === player?.id);
    return idx >= 0 ? idx : 0;
  }, [players, player?.id]);

  const baseOffset = participantIndex % universities.length;
  const baseOptions = useMemo(
    () => rotate(universities, baseOffset).map((name) => ({ name })),
    [universities, baseOffset]
  );

  // Single source of truth for the indicated choice in tutorial
  const [targetName, setTargetName] = useState(null);

  // Initialize/persist target once per tutorial stage
  useEffect(() => {
    if (!isTutorial) {
      setTargetName(null);
      return;
    }
    const first = baseOptions[0]?.name || null;
    setTargetName(first);
    if (first) {
      // Why: keep DB in sync for audit/replay; avoids stale values from older versions.
      player.set("tutorialChoice", first);
    }
  }, [isTutorial, baseOptions, player]);

  // Timer warning
  useEffect(() => {
    const rem = timer?.remaining;
    if (rem === undefined || rem === null) return;
    const remaining = Math.round(rem / 1000);
    setShowWarning(remaining <= 10 && remaining > 0);
  }, [timer]);

  // Team-wide or per-player stats
  const cumulativeResults = useMemo(
    () => (chatEnabled ? game.get("cumulativeResults") || {} : player.get("cumulativeResults") || {}),
    [chatEnabled, game, player]
  );

  function onClick(candidate) {
    // Enforce only when target is defined
    if (isTutorial && targetName && candidate.name !== targetName) {
      setShowWarningModal(true);
      return;
    }
    const p = PROBABILITIES[candidate.name] ?? 0.5;
    const score = bernoulliRandom(p);
    player.round.set("decision", candidate.name);
    player.round.set("score", score);
    player.stage.set("submit", true);
  }

  return (
    <div className="mt-3 sm:mt-5 p-6 bg-gray-100 rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold text-gray-800 mb-4 text-center">
        {isTutorial ? "Tutorial: Learn How to Play" : "Hiring Game: Select Your Candidate"}
      </h2>

      <p className="text-gray-600 mb-4 text-center text-lg min-h-[60px]">
        {isTutorial ? (
          <span>
            Select the <strong>indicated</strong> university to proceed. 😊
          </span>
        ) : (
          <span>
            Select the university you believe will produce the <strong>most productive intern</strong>. 🌟
          </span>
        )}
      </p>

      {showWarning && (
        <div className="text-center text-red-600 font-bold mb-4 min-h-[24px]">
          ⚠️ Hurry up!! Only{" "}
          <span className="tabular-nums">{Math.max(Math.round((timer?.remaining ?? 0) / 1000), 0)}</span> seconds left! 🚨
        </div>
      )}

      {showWarningModal && (
        <div className="absolute inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-lg shadow-lg text-center max-w-sm">
            <div className="text-2xl text-red-600 font-bold mb-4">⚠️ Attention</div>
            <p className="text-gray-800 text-lg mb-4">
              In the tutorial, please select the <strong>indicated</strong> one to proceed.
            </p>
            <button
              onClick={() => setShowWarningModal(false)}
              className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
            >
              Got it!
            </button>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3 items-start justify-center px-4 relative">
        {baseOptions.map((option) => (
          <div key={option.name} className="relative flex flex-col items-center">
            {/* Pointer: tie to the actual target name */}
            {isTutorial && targetName === option.name && (
              <div className="absolute -top-14 flex justify-center w-full">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
                     className="w-14 h-14 fill-gray-700 animate-bounce" aria-hidden="true">
                  <path d="M12 2v20m-5-5l5 5 5-5" />
                </svg>
              </div>
            )}

            <Button
              className="m-1 w-44 h-24 bg-gray-200 hover:bg-gray-300 hover:scale-105 transition-all rounded-lg flex items-center justify-center text-center p-2"
              handleClick={() => onClick(option)}
              title={option.name}
            >
              <div className="flex flex-col items-center gap-1">
                {/* University building icon */}
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
                     className="w-6 h-6" aria-hidden="true">
                  <path d="M12 2 2 7l10 5 10-5-10-5Zm-7 8v7h3v-5h2v5h4v-5h2v5h3v-7l-7 3.5L5 10Z" />
                </svg>
                <span className="text-xs sm:text-sm font-medium leading-snug line-clamp-3 text-black">
                  {option.name}
                </span>
              </div>
            </Button>

            <div className="text-center mt-2">
              <span className="text-green-800 font-bold">
                Success:{" "}
                <span className="text-2xl font-extrabold">
                  {cumulativeResults[option.name]?.success || 1}
                </span>
              </span>
              <br />
              <span className="text-red-600 font-bold">
                Failure:{" "}
                <span className="text-2xl font-extrabold">
                  {cumulativeResults[option.name]?.failures || 1}
                </span>
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}