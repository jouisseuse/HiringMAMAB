import React from "react";
import { Button } from "../components/Button";
import { useGame, useRound } from "@empirica/core/player/classic/react";

export function Introduction({ next }) {
  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100">
      <div className="w-full max-w-3xl p-10 bg-white rounded-lg shadow-md text-left">
        <h3 className="text-3xl font-extrabold text-gray-800 mb-6">
          🏢 Welcome to the Hiring Boardroom! 💼
        </h3>
        <p className="text-gray-700 text-lg leading-8 mb-6">
          Congratulations! You've been appointed as a key member of the hiring committee. 
          You are responsible for selecting interns from students at <strong>10 different universities</strong>.
        </p>  
        <p className="text-gray-700 text-lg leading-8 mb-6">
          Across the task, you will make a series of hiring decisions. As you gain experience, you will learn about the <strong>typical performance</strong> of students from each university. 
          Your goal is to use this information to identify which universities provide <strong>strong student interns</strong>, and maximize your total <strong>bonus</strong>! 💰
        </p>
        <p className="text-gray-700 text-lg leading-8 mb-6">
          It is completely fine if you are unfamiliar with these universities. We are interested in how people learn from experience, 
          so please <strong>DO NOT</strong> search the internet or use any AI tools during the task.
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