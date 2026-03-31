import React from "react";
import { Button } from "../components/Button.jsx";
import { useRound, usePlayer } from "@empirica/core/player/classic/react";

export function GameIntroduction() {
  const round = useRound();
  const player = usePlayer();

  if (!round) {
    console.error("GameIntroduction: No round available");
    return <div>Loading...</div>;
  }

  // console.log(`[gameintro] ${round}`);
  // console.log(round.get("chatEnabled"));
  const chatEnabled = round.get("chatEnabled")

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100">
      <div className="w-full max-w-3xl p-10 bg-white rounded-lg shadow-md text-left">
        <h3 className="text-3xl font-extrabold text-gray-800 mb-6">
          🎯 Task: Hiring the Best Candidate
        </h3>
        <p className="text-gray-700 text-lg leading-8 mb-6">
          You will be asked to make 50 hiring decisions. In each round, you can select one university from which to hire an intern.
        </p>
        <p className="text-gray-700 text-lg leading-8 mb-6">
          Just as in real life, intern performance might be relevant to their university but not necessarily a perfect indicator. Your task is to learn which universities tend to <strong>produce more productive interns</strong>.
        </p>
        <p className="text-gray-700 text-lg leading-8 mb-6">
          💰 For each successful hire, you will earn <strong>a bonus of 1 cent</strong> in addition to your base payment. The more successful hires you make, the greater the bonus you will earn, up to a maximum of 50 cents.
        </p>
        {chatEnabled ? (
          <p className="text-gray-700 text-lg leading-8 mb-6">
            🗣️  This round includes <strong>information sharing</strong>. After making your decision, you will see the choices and results of other committee members. 
          </p>  
        ) : (
          <p className="text-gray-700 text-lg leading-8 mb-6">
            🗣️ This round <strong>does not include information sharing</strong>. After making your decision, you will not see the choices and results of other committee members.
          </p>
        )}
        <Button
          className="bg-blue-500 text-white py-3 px-8 rounded-lg hover:bg-blue-600"
          handleClick={() => {
            player.stage.set("submit", true);
          }}
          autoFocus
        >
          ✅ Continue to the Next Round
        </Button>
      </div>
    </div>
  );
}

