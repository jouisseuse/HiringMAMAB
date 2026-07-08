import React from "react";
import { Button } from "../components/Button";
import { useGame, useRound } from "@empirica/core/player/classic/react";

export function Introduction({ next }) {
  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100">
      <div className="w-full max-w-3xl p-10 bg-white rounded-lg shadow-md text-left">
        <h3 className="text-3xl font-extrabold text-gray-800 mb-6">
          🏢 Welcome to the Together Hire! 💼
        </h3>
        <p className="text-gray-700 text-lg leading-8 mb-6">
          Congratulations! You've been appointed as a key member of the <strong>hiring committee</strong>. 
          Your task is to evaluate and select candidates from <strong>10 different groups</strong>. 
        </p>
        <p className="text-gray-700 text-lg leading-8 mb-6">
          Every decision you make will provide valuable insights into the <strong>productivity</strong> of each group. 
          Your goal is to identify the <strong>most capable groups of workers</strong> and maximize your <strong>bonus</strong>! 💰
        </p>
        <p className="text-gray-700 text-lg leading-8 mb-6">
          ⚖️ In some rounds, you will see others' choices and results for reference. In others, you'll rely solely on your 
          own judgment. Stay observant and strategic to uncover the strengths of each group and make the best decisions. 🌟
        </p>
        <p className="text-red-600 text-lg font-semibold leading-8 mb-6">
          ⏳ The waiting time may be long. Please be patient.  
          Once the game starts, please <strong>DO NOT Exit</strong> as it will affect the entire game for all players.
        </p>
        <Button
          className="bg-blue-500 text-white py-3 px-8 rounded-lg hover:bg-blue-600"
          handleClick={next}
          autoFocus
        >
          🚀 Begin the Hiring Challenge
        </Button>
      </div>
    </div>
  );
}