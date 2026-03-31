import { ClassicListenersCollector } from "@empirica/core/admin/classic";

export const Empirica = new ClassicListenersCollector();

Empirica.onGameStart(({ game }) => {
  const treatment = game.get("treatment");
  const { numRounds, chatEnabled, type } = treatment;

  console.log(`Game started with treatment: ${type}, chatEnabled: ${chatEnabled}`);

  // Initialize cumulativeResults for all candidates
  const initialResults = {};
  const options = [
    "Trinity College Dublin",
    "The University of Western Australia",
    "University of Glasgow",
    "Heidelberg University",
    "University of Adelaide",
    "University of Leeds",
    "University of Southampton",
    "University of Sheffield",
    "University of Nottingham",
    "Karlsruhe Institute of Technology"
  ];
  options.forEach((option) => {
    initialResults[option] = { success: 1, failures: 1};
  });

  game.set("cumulativeResults", initialResults); // shared across all players

  if (Array.isArray(game.players)) {
    game.players.forEach((player, index) => {
      // Initialize per-player cumulativeResults
      player.set("cumulativeResults", initialResults);

      // Assign tutorial choice
      const assignedChoice = options[index % options.length];
      player.set("tutorialChoice", assignedChoice);

      console.log(`Initialized Player ${player.id}: tutorialChoice=${assignedChoice}`);
    });
  } else {
    console.error("Players is not defined or not an array:", game.players);
  }

  // Set up rounds based on treatment type
  if (type === "communication" || type === "non-communication") {
    // Single-mode treatment
    addIntroductionRound(game, chatEnabled);
    addTutorialRound(game, chatEnabled);
    addRounds(game, numRounds, chatEnabled);
    addPostGameSurvey(game, chatEnabled);

  } else if (type === "combined") {
    // Combined treatment
    console.log("Setting up Combined-Treatment...");

    // Non-communication phase
    addIntroductionRound(game, false);
    addTutorialRound(game, false);
    addRounds(game, numRounds, false);

    // Communication phase
    addIntroductionRound(game, true);
    addRounds(game, numRounds, true);
  } else {
    console.error(`Unknown treatment: ${type}`);
  }
});

// Add introduction round
function addIntroductionRound(game, chatEnabled) {
  const introRound = game.addRound({
    name: chatEnabled ? "Communication Mode Introduction" : "Non-Communication Introduction",
    "chatEnabled": chatEnabled,
    "flag": false,
  });
  introRound.addStage({ name: "introduction", duration: 60 });
}

function addTutorialRound(game, chatEnabled) {
  const tutorialRound = game.addRound({
    name: "Tutorial Round",
    "isTutorial": true,
    "flag": true,
    "chatEnabled": chatEnabled,
  });

  tutorialRound.addStage({ name: "choice", duration: 60 });
}

// Add regular game rounds
function addRounds(game, numRounds, chatEnabled) {
  for (let i = 0; i < numRounds; i++) {
    const round = game.addRound({
      name: `${chatEnabled ? "Communication" : "Non-Communication"} Round ${i + 1}`,
      "chatEnabled": chatEnabled,
      "flag": true,
      "isTutorial": false,
    });
    round.addStage({ name: "choice", duration: 30 });
  }
}

function addPostGameSurvey(game, chatEnabled) {
  const postSurveyRound = game.addRound({ name: "Post-Survey", "chatEnabled": chatEnabled });
  postSurveyRound.addStage({ name: "Group-Allocation", duration: 200 });
}

Empirica.onRoundStart(({ round }) => {
  console.log(`Round started: ${round.get("name")}, Chat Enabled: ${round.get("chatEnabled")}, tutorial: ${round.get("isTutorial")}`);
});

Empirica.onStageStart(({ stage }) => {
  console.log(`Stage started: ${stage.get("name")}, Chat Enabled: ${stage.round.get("chatEnabled")}`);
});

Empirica.onStageEnded(({ stage }) => {
  console.log(`Stage ended: ${stage.get("name")}, Chat Enabled: ${stage.round.get("chatEnabled")}`);
});

Empirica.onRoundEnded(({ round }) => {
  if (round.get("flag") === true) {
    const game = round.currentGame;
    const players = round.currentGame.players;
    const cumulativeResultsGame = game.get("cumulativeResults") || {};
    const chatEnabled = !!round.get("chatEnabled");
    const isTutorial = !!round.get("isTutorial");

    // Ensure all candidate options are initialized
    const options = [
      "Trinity College Dublin",
      "The University of Western Australia",
      "University of Glasgow",
      "Heidelberg University",
      "University of Adelaide",
      "University of Leeds",
      "University of Southampton",
      "University of Sheffield",
      "University of Nottingham",
      "Karlsruhe Institute of Technology"
    ];

    players.forEach((player) => {
      let playerChoice = player.round.get("decision");
      let score = player.round.get("score");
      const cumulativeResultsPlayer = player.get("cumulativeResults") || {};

      if (!playerChoice) {
        console.error(`Player ${player.id} has no choice for this round.`);
      }

      // Initialize if first occurrence (global)
      if (!cumulativeResultsGame[playerChoice]) {
        cumulativeResultsGame[playerChoice] = { success: 1, failures: 1 };
      }

      // Update global results
      if (score === 1) {
        cumulativeResultsGame[playerChoice].success += 1;
      } else {
        cumulativeResultsGame[playerChoice].failures += 1;
      }

      // Initialize if first occurrence (per-player)
      if (!cumulativeResultsPlayer[playerChoice]) {
        cumulativeResultsPlayer[playerChoice] = { success: 1, failures: 1 };
      }

      // Update per-player results
      if (score === 1) {
        cumulativeResultsPlayer[playerChoice].success += 1;
      } else {
        cumulativeResultsPlayer[playerChoice].failures += 1;
      }
      player.set("cumulativeResults", cumulativeResultsPlayer);

      const totalscore = player.get("score") || 0;
      const newscore = totalscore + score;
      player.set("score", newscore);
      console.log(`initial score: ${totalscore}, new score: ${newscore}`);
    });

    // Persist updated cumulative results
    game.set("cumulativeResults", cumulativeResultsGame);
  }
});

Empirica.onGameEnded(({ game }) => {
  console.log(`Game ended. Treatment: ${game.get("treatment").type}`);
});

function bernoulliRandom(p) {
  return Math.random() < p ? 1 : 0;
}
