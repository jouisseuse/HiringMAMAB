import { usePlayer, usePlayers, useRound, useGame, useStageTimer } from "@empirica/core/player/classic/react";
import React, { useEffect, useMemo, useState } from "react";
import { Button } from "../components/Button";
import "@unocss/reset/tailwind-compat.css";

function bernoulliRandom(p) {
  return Math.random() < p ? 1 : 0;
}

export function Choice() {
  const player = usePlayer();
  const players = usePlayers();
  const round = useRound();
  const game = useGame();
  const chatEnabled = round.get("chatEnabled");
  const isTutorial = round.get("isTutorial");
  const tutorialChoice = player.get("tutorialChoice");
  const timer = useStageTimer();

  const [showWarning, setShowWarning] = useState(false);
  const [showWarningModal, setShowWarningModal] = useState(false);

  const options = [
    { name: "Crimson", color: "fill-red-700 hover:fill-red-900 bg-gray-200" }, // 深红色
    { name: "Bright Green", color: "fill-lime-400 hover:fill-lime-500 bg-gray-200" }, // 明亮荧光绿
    { name: "Amber", color: "fill-amber-500 hover:fill-amber-600 bg-gray-200" }, // 金黄色
    { name: "Purple", color: "fill-purple-700 hover:fill-purple-900 bg-gray-200" }, // 紫色
    { name: "Sky Blue", color: "fill-sky-500 hover:fill-sky-700 bg-gray-200" }, // 天蓝色
    { name: "Pink", color: "fill-pink-500 hover:fill-pink-700 bg-gray-200" }, // 粉色
    { name: "Indigo", color: "fill-indigo-500 hover:fill-indigo-700 bg-gray-200" }, // 靛蓝色
    { name: "Slate", color: "fill-slate-400 hover:fill-slate-600 bg-gray-200" }, // 浅灰蓝
    { name: "Orange", color: "fill-orange-600 hover:fill-orange-700 bg-gray-200" }, // 橙色
    { name: "Black", color: "fill-black hover:fill-gray-800 bg-gray-200" }, // 黑色
  ];

  const probabilities = {
    "Crimson": 0.53,
    "Bright Green": 0.95,
    "Amber": 0.3,
    "Purple": 0.1,
    "Sky Blue": 0.68,
    "Pink": 0.4,
    "Indigo": 0.16,
    "Slate": 0.8,
    "Orange": 0.22,
    "Black": 0.12
  };

  useEffect(() => {
    if (timer?.remaining || timer?.remaining === 0) {
      const remaining = Math.round(timer.remaining / 1000);
      if (remaining <= 10 && remaining > 0) {
        setShowWarning(true);
      } else {
        setShowWarning(false);
      }
    }
  }, [timer]);

  // 根据 tutorialChoice 动态调整顺序
  const sortedOptions = useMemo(() => {
    if (!tutorialChoice) return options;
    const startIndex = options.findIndex((option) => option.name === tutorialChoice);
    return [
      ...options.slice(startIndex),
      ...options.slice(0, startIndex),
    ];
  }, [tutorialChoice]);

  // 获取 cumulativeResults，根据是否有交流展示不同的数据
  const cumulativeResults = useMemo(() => {
    if (chatEnabled) {
      return game.get("cumulativeResults") || {}; // 全局结果
    }
    return player.get("cumulativeResults") || {}; // 玩家个人结果
  }, [chatEnabled, game, player]);

  function onClick(candidate) {
    if (isTutorial && candidate.name !== tutorialChoice) {
      setShowWarningModal(true); // 显示自定义弹窗
      return;
    }

    const score = bernoulliRandom(0.9); // Reward calculation
    player.round.set("decision", candidate.name);
    // const score = bernoulliRandom(probabilities[candidate.name]);
    // console.log(probabilities[candidate.name])
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
            Follow the instructions in this tutorial round. Please select the{" "}
            <strong>indicated</strong> one. 😊 
          </span>
        ) : (
          <span>
            Select the candidate you believe is <strong>most productive</strong>. 🌟
          </span>
        )}
      </p>
  
      {/* 警告消息 */}
      {showWarning && (
        <div className="text-center text-red-600 font-bold mb-4 min-h-[24px]">
          ⚠️ Hurry up!! Only{" "}
          <span className="tabular-nums">{Math.max(Math.round(timer?.remaining / 1000), 0)}</span> seconds left to make your choice! 🚨
        </div>
      )}

      {/* 自定义弹窗 */}
      {showWarningModal && (
        <div className="absolute top-0 left-0 w-full h-full bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-lg shadow-lg text-center max-w-sm">
            <div className="text-2xl text-red-600 font-bold mb-4">⚠️ Attention</div>
            <p className="text-gray-800 text-lg mb-4">
              In the tutorial, you must select the <strong>indicated</strong> one to proceed.
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
  
      <div className="grid grid-cols-5 gap-3 items-center justify-center px-4 relative">
        {sortedOptions.map((option, index) => (
          <div key={index} className="relative flex flex-col items-center">
            {/* 跳动箭头 */}
            {isTutorial && index === 0 && (
              <div className="absolute -top-14 flex justify-center w-full">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  className="w-14 h-14 fill-gray-700 animate-bounce"
                >
                  <path d="M12 2v20m-5-5l5 5 5-5" />
                </svg>
              </div>
            )}
            <Button
              className={`m-1 ${option.color} w-24 h-24 hover:scale-105 transition-all rounded-lg flex items-center justify-center`}
              handleClick={() => onClick(option)}
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" className="w-full h-full">
                <path d="M12 12c-2.21 0-4-1.79-4-4s1.79-4 4-4 4 1.79 4 4-1.79 4-4 4zm0 2c2.67 0 8 1.34 8 4v2H4v-2c0-2.66 5.33-4 8-4z" />
              </svg>
            </Button>
            {/* Success and Failure Counts */}
            {/* <div className="text-center mt-2">
              <span className="text-green-800 font-bold">
                Success:{" "}
                <span
                  className="text-2xl font-extrabold"
                  style={{ textShadow: "1px 1px 2px rgba(0, 0, 0, 0.3)" }}
                >
                  {cumulativeResults[option.name]?.success || 1}
                </span>
              </span>
              <br />
              <span className="text-red-600 font-bold">
                Failure:{" "}
                <span
                  className="text-2xl font-extrabold"
                  style={{ textShadow: "1px 1px 2px rgba(0, 0, 0, 0.3)" }}
                >
                  {cumulativeResults[option.name]?.failures || 1}
                </span>
              </span>
            </div> */}
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