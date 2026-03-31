// path: src/stages/GroupAllocation.jsx
import React, { useState, useMemo, useEffect } from "react";
import { usePlayer, useRound, useGame, usePlayers } from "@empirica/core/player/classic/react";
import { Button } from "../components/Button";
import "@unocss/reset/tailwind-compat.css";


// Rotate array by offset
function rotate(arr, offset) {
  const n = arr.length;
  if (!n) return arr;
  const k = ((offset % n) + n) % n;
  return [...arr.slice(k), ...arr.slice(0, k)];
}

export function GroupAllocation() {
  const player = usePlayer();
  const players = usePlayers();
  const game = useGame();
  const round = useRound();
  const chatEnabled = round.get("chatEnabled");

  const totalSlots = 100;
  const [allocations, setAllocations] = useState({});
  const [remainingSlots, setRemainingSlots] = useState(totalSlots);

  // Exact labels used everywhere (buttons, results keys, saved data)
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

  // Team-wide or per-player stats depending on chatEnabled
  const cumulativeResults = useMemo(() => {
    return chatEnabled ? game.get("cumulativeResults") || {} : player.get("cumulativeResults") || {};
  }, [chatEnabled, game, player]);

  // Keep remaining slots in sync when allocations change elsewhere
  useEffect(() => {
    const totalAllocated = Object.values(allocations).reduce((sum, v) => sum + (Number(v) || 0), 0);
    setRemainingSlots(Math.max(0, totalSlots - totalAllocated));
  }, [allocations]);

  const handleInputChange = (groupName, value) => {
    const num = Math.max(0, parseInt(value, 10) || 0); // clamp to [0, ...]
    // Provisional total with the edited group
    const totalAllocated = Object.entries(allocations).reduce(
      (sum, [key, val]) => sum + (key === groupName ? num : (Number(val) || 0)),
      0
    );
    if (totalAllocated <= totalSlots) {
      setAllocations((prev) => ({ ...prev, [groupName]: num }));
      setRemainingSlots(totalSlots - totalAllocated);
    }
  };

  const handleSubmit = () => {
    player.set("groupAllocations", allocations);
    player.stage.set("submit", true);
  };

  return (
    <div className="mt-3 sm:mt-5 p-6 bg-gray-100 rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold text-gray-800 mb-4 text-center">Group Allocation</h2>

      <p className="text-gray-600 mb-4 text-center text-lg">
        🎉 Congratulations! This is the <strong>final task</strong> of the game. The closer your answers are to the true productivity, the higher your <strong>bonus</strong> will be! 💰
      </p>
      <p className="text-gray-600 mb-4 text-center text-lg">
        Based on what you observed about intern performance during the task, please allocate <strong>{totalSlots}</strong> intern positions across the universities.
      </p>
      <p className="text-center text-gray-800 font-bold mb-4">Remaining Slots: {remainingSlots}</p>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3 items-start justify-center px-4 relative">
        {baseOptions.map((option) => (
          <div key={option.name} className="relative flex flex-col items-center">
            <Button
              className="m-1 w-44 h-24 bg-gray-200 hover:bg-gray-300 rounded-lg flex items-center justify-center text-center p-2"
            >
              {/* University building icon + name (black) */}
              <div className="flex flex-col items-center gap-1">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
                     className="w-6 h-6" aria-hidden="true">
                  <path d="M12 2 2 7l10 5 10-5-10-5Zm-7 8v7h3v-5h2v5h4v-5h2v5h3v-7l-7 3.5L5 10Z" />
                </svg>
                <span className="text-xs sm:text-sm font-medium leading-snug line-clamp-3 text-black">
                  {option.name}
                </span>
              </div>
            </Button>

            <div className="text-sm mt-2 text-center">
              <span className="text-green-800 font-bold">
                Success: {cumulativeResults[option.name]?.success ?? 0}
              </span>
              <br />
              <span className="text-red-600 font-bold">
                Failures: {cumulativeResults[option.name]?.failures ?? 0}
              </span>
            </div>

            <input
              type="number"
              min="0"
              max={totalSlots}
              className="mt-2 border border-gray-300 rounded p-1 text-center w-16"
              value={allocations[option.name] ?? 0}
              onChange={(e) => handleInputChange(option.name, e.target.value)}
            />
          </div>
        ))}
      </div>

      <div className="text-center mt-6">
        <button
          onClick={handleSubmit}
          disabled={remainingSlots > 0}
          className={`px-6 py-2 rounded text-white font-bold ${
            remainingSlots === 0 ? "bg-blue-500 hover:bg-blue-600" : "bg-gray-400 cursor-not-allowed"
          }`}
        >
          Submit Allocation
        </button>
      </div>
    </div>
  );
}