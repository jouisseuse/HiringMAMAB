import React, { useState, useMemo } from "react";
import { usePlayer, usePlayers, useRound, useGame } from "@empirica/core/player/classic/react";
import { Button } from "../components/Button";
import "@unocss/reset/tailwind-compat.css";

export function GroupAllocation() {
  const player = usePlayer();
  const game = useGame();
  const round = useRound();
  const chatEnabled = round.get("chatEnabled");

  // 初始名额
  const totalSlots = 100;
  const [allocations, setAllocations] = useState({});
  const [remainingSlots, setRemainingSlots] = useState(totalSlots);
  const tutorialChoice = player.get("tutorialChoice");

  const options = [
    { name: "Crimson", color: "fill-red-700 hover:fill-red-900 bg-gray-200" },
    { name: "Bright Green", color: "fill-lime-400 hover:fill-lime-500 bg-gray-200" },
    { name: "Amber", color: "fill-amber-500 hover:fill-amber-600 bg-gray-200" },
    { name: "Purple", color: "fill-purple-700 hover:fill-purple-900 bg-gray-200" },
    { name: "Sky Blue", color: "fill-sky-500 hover:fill-sky-700 bg-gray-200" },
    { name: "Pink", color: "fill-pink-500 hover:fill-pink-700 bg-gray-200" },
    { name: "Indigo", color: "fill-indigo-500 hover:fill-indigo-700 bg-gray-200" },
    { name: "Slate", color: "fill-slate-400 hover:fill-slate-600 bg-gray-200" },
    { name: "Orange", color: "fill-orange-600 hover:fill-orange-700 bg-gray-200" },
    { name: "Black", color: "fill-black hover:fill-gray-800 bg-gray-200" },
  ];

  // 根据 tutorialChoice 动态调整顺序
  const sortedOptions = useMemo(() => {
    if (!tutorialChoice) return options;
    const startIndex = options.findIndex((option) => option.name === tutorialChoice);
    return [
      ...options.slice(startIndex),
      ...options.slice(0, startIndex),
    ];
  }, [tutorialChoice, options]);

  // 获取 cumulativeResults，根据是否有交流展示不同的数据
  const cumulativeResults = useMemo(() => {
    if (chatEnabled) {
      return game.get("cumulativeResults") || {}; // 全局结果
    }
    return player.get("cumulativeResults") || {}; // 玩家个人结果
  }, [chatEnabled, game, player]);

  const handleInputChange = (groupName, value) => {
    const num = Math.max(0, parseInt(value, 10) || 0); // 将非数字处理为 0
    const totalAllocated = Object.entries(allocations).reduce(
      (sum, [key, val]) => (key === groupName ? sum + num : sum + val),
      0
    );
  
    if (totalAllocated <= totalSlots) {
      setAllocations((prev) => ({ ...prev, [groupName]: num }));
      setRemainingSlots(totalSlots - totalAllocated);
    }
  };

  const handleSubmit = () => {
    console.log("Allocations submitted:", allocations);
    player.set("groupAllocations", allocations); // 保存分配到玩家数据
    player.stage.set("submit", true); // 提交阶段
  };

  return (
    <div className="mt-3 sm:mt-5 p-6 bg-gray-100 rounded-lg shadow-lg">
        <h2 className="text-2xl font-bold text-gray-800 mb-4 text-center">Group Allocation</h2>
        <p className="text-gray-600 mb-4 text-center text-lg">
        🎉 Congratulations! This is the <strong>final task</strong> of the game. The closer your answers are to the true productivity, the higher your <strong>bonus</strong> will be! 💰
        </p>
        <p className="text-gray-600 mb-4 text-center text-lg">
        After observing the groups' performance during the game, allocate <strong>{totalSlots}</strong> hires across the groups based on their productivity.
        </p>
        <p className="text-center text-gray-800 font-bold mb-4">Remaining Slots: {remainingSlots}</p>

      <div className="grid grid-cols-5 gap-3 items-center justify-center px-4 relative">
        {sortedOptions.map((option, index) => (
          <div key={index} className="relative flex flex-col items-center">
            <Button
              className={`m-1 ${option.color} w-24 h-24 rounded-lg flex items-center justify-center`}
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" className="w-full h-full">
                <path d="M12 12c-2.21 0-4-1.79-4-4s1.79-4 4-4 4 1.79 4 4-1.79 4-4 4zm0 2c2.67 0 8 1.34 8 4v2H4v-2c0-2.66 5.33-4 8-4z" />
              </svg>
            </Button>
            <div className="text-sm mt-2 text-center">
              <span className="text-green-800 font-bold">
                Success: {cumulativeResults[option.name]?.success || 1}
              </span>
              <br />
              <span className="text-red-600 font-bold">
                Failures: {cumulativeResults[option.name]?.failures || 1}
              </span>
            </div>
            <input
                type="number"
                min="0"
                max={totalSlots}
                className="mt-2 border border-gray-300 rounded p-1 text-center w-16"
                value={allocations[option.name] || 0} // 始终显示数字，默认为 0
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