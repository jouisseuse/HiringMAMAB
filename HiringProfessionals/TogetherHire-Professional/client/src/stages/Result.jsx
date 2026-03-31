import React, { useEffect, useMemo, useState } from "react";
import { usePlayer, usePlayers, useRound, useGame } from "@empirica/core/player/classic/react";
import { Button } from "../components/Button";

function bernoulliRandom(p) {
  return Math.random() < p ? 1 : 0;
}

export function Result() {
  const player = usePlayer();
  const players = usePlayers();
  const round = useRound();
  const chatEnabled = round.get("chatEnabled"); // Check if communication mode is enabled

  const aggregatedResults = round.get("aggregatedResults");
  const personalResult = {
    choice: player.round.get("decision"),
    success: player.round.get("score") === 1,
  };
  
  // Handle undefined personalResult for non-communication mode
  if (!chatEnabled && personalResult === null) {
    return (
      <div className="mt-10 sm:mt-12 p-10 bg-gray-100 rounded-lg shadow-md">
        <h3 className="text-2xl font-bold text-gray-800 mb-4">Hiring Round Results</h3>
        <p className="text-gray-700 leading-6 mb-6">Loading your results...</p>
      </div>
    );
  }

  return (
    <div className="mt-10 sm:mt-12 p-10 bg-gray-100 rounded-lg shadow-md">
      <h3 className="text-2xl font-bold text-gray-800 mb-4">Hiring Round Results</h3>
      <p className="text-gray-700 leading-6 mb-6">
        {chatEnabled
          ? "The results of this round are summarized across all participants."
          : "The results of this round are based on your personal decision."}
      </p>

      {chatEnabled ? (
        <ul className="text-gray-700 mb-6">
          {Object.entries(aggregatedResults).map(([choice, result]) => (
            <li key={choice} className="mb-2">
              <p>
                Candidate: <strong>{choice}</strong>, Selected by: <strong>{result.count}</strong> people, Success: <strong>{result.success}</strong>, Failures: <strong>{result.failures}</strong>
                {/* <strong>Candidate:</strong> {choice} <strong>Selected by:</strong> {result.count} people <strong>Success:</strong> {result.success} <strong>Failures:</strong> {result.failures} */}
              </p>
            </li>
          ))}
        </ul>
      ) : (
        personalResult && (
          <div className="text-gray-700 mb-6">
            <p>
              <strong>You selected:</strong> {personalResult.choice} <br />
              <strong>Result:</strong> {personalResult.success ? "Success" : "Failure"}
            </p>
          </div>
        )
      )}

      <Button
        className="bg-blue-500 text-white py-2 px-6 rounded hover:bg-blue-600"
        handleClick={() => player.stage.set("submit", true)}
      >
        Continue
      </Button>
    </div>
  );
}