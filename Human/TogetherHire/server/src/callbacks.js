import { ClassicListenersCollector } from "@empirica/core/admin/classic";

export const Empirica = new ClassicListenersCollector();

Empirica.onGameStart(({ game }) => {
  const treatment = game.get("treatment");
  const { numRounds, chatEnabled, type } = treatment;

  console.log(`Game started with treatment: ${type}, chatEnabled: ${chatEnabled}`);

  // 初始化 cumulativeResults
  const initialResults = {};
  // const options = ["Red", "Blue", "Green", "Yellow", "Purple", "Orange", "Pink", "Brown", "Gray", "Black"];
  const options = [
    "Crimson",       // 深红色
    "Bright Green",  // 明亮荧光绿
    "Amber",         // 金黄色
    "Purple",        // 紫色
    "Sky Blue",      // 天蓝色
    "Pink",          // 粉色
    "Indigo",        // 靛蓝色
    "Slate",         // 浅灰蓝
    "Orange",        // 橙色
    "Black"          // 黑色
  ];
  options.forEach((option) => {
    initialResults[option] = { success: 1, failures: 1};
  });

  game.set("cumulativeResults", initialResults); // 全局共享

  if (Array.isArray(game.players)) {
    game.players.forEach((player, index) => {
      // 初始化 cumulativeResults 为每位玩家
      player.set("cumulativeResults", initialResults);

      // 为教程回合指定选择
      const assignedChoice = options[index % options.length];
      player.set("tutorialChoice", assignedChoice);

      console.log(`Initialized Player ${player.id}: tutorialChoice=${assignedChoice}`);
    });
  } else {
    console.error("Players is not defined or not an array:", game.players);
  }

  // 根据不同 treatment 动态生成回合
  if (type === "communication" || type === "non-communication") {
    // 单一模式
    addIntroductionRound(game, chatEnabled);
    addTutorialRound(game, chatEnabled);
    addRounds(game, numRounds, chatEnabled);
    addPostGameSurvey(game, chatEnabled);

  } else if (type === "combined") {
    // 组合模式
    console.log("Setting up Combined-Treatment...");

    // 非通信模式
    addIntroductionRound(game, false);
    addTutorialRound(game,false);
    addRounds(game, numRounds, false);

    // 通信模式
    addIntroductionRound(game, true);
    addRounds(game, numRounds, true);
  } else {
    console.error(`Unknown treatment: ${type}`);
  }
});

// 添加介绍回合
function addIntroductionRound(game, chatEnabled) {
  const introRound = game.addRound({
    name: chatEnabled ? "Communication Mode Introduction" : "Non-Communication Introduction",
    "chatEnabled": chatEnabled,
    "flag": false,
  });
  // introRound.set("chatEnabled", chatEnabled);
  introRound.addStage({ name: "introduction", duration: 60 });
}

function addTutorialRound(game, chatEnabled) {

  const tutorialRound = game.addRound({
    name: "Tutorial Round",
    "isTutorial": true, // 标记为教程回合
    "flag": true,
    "chatEnabled": chatEnabled,
  });

  tutorialRound.addStage({ name: "choice", duration: 60 });
  // tutorialRound.addStage({ name: "result", duration: 30 });
}

// 添加普通回合
function addRounds(game, numRounds, chatEnabled) {
  for (let i = 0; i < numRounds; i++) {
    const round = game.addRound({
      name: `${chatEnabled ? "Communication" : "Non-Communication"} Round ${i + 1}`,
      "chatEnabled": chatEnabled,
      "flag": true,
      "isTutorial": false, 
    });
    // round.set("chatEnabled", chatEnabled);
    round.addStage({ name: "choice", duration: 30 });

    // round.addStage({ name: "result", duration: 30 });
  }
}

function addPostGameSurvey(game, chatEnabled) {
  const postSurveyRound = game.addRound({ name: "Post-Survey", "chatEnabled": chatEnabled,});
  postSurveyRound.addStage({ name: "Group-Allocation", duration: 200 }); // 100 秒
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

    // 初始化候选人的默认值
    const options = [
      "Crimson",       // 深红色
      "Bright Green",  // 明亮荧光绿
      "Amber",         // 金黄色
      "Purple",        // 紫色
      "Sky Blue",      // 天蓝色
      "Pink",          // 粉色
      "Indigo",        // 靛蓝色
      "Slate",         // 浅灰蓝
      "Orange",        // 橙色
      "Black"          // 黑色
    ];


    players.forEach((player) => {
      const playerChoice = player.round.get("decision");
      const score = player.round.get("score");
      const cumulativeResultsPlayer = player.get("cumulativeResults") || {};

      if (!playerChoice) {
        console.error(`Player ${player.id} has no choice for this round.`);
        return;
      }

      // 初始化全局 cumulativeResults
      if (!cumulativeResultsGame[playerChoice]) {
        cumulativeResultsGame[playerChoice] = { success: 1, failures: 1 };
      }

      // 更新全局结果
      if (score === 1) {
        cumulativeResultsGame[playerChoice].success += 1;
      } else {
        cumulativeResultsGame[playerChoice].failures += 1;
      }

      // 初始化全局 cumulativeResults
      if (!cumulativeResultsPlayer[playerChoice]) {
        cumulativeResultsPlayer[playerChoice] = { success: 1, failures: 1 };
      }

      // 更新全局结果
      if (score === 1) {
        cumulativeResultsPlayer[playerChoice].success += 1;
      } else {
        cumulativeResultsPlayer[playerChoice].failures += 1;
      }
      player.set("cumulativeResults",cumulativeResultsPlayer);
      

      // 聚合结果
      // aggregatedResults[playerChoice].count += 1;
      // if (score === 1) {
      //   aggregatedResults[playerChoice].success += 1;
      // } else {
      //   aggregatedResults[playerChoice].failures += 1;
      // }

      const totalscore = player.get("score") || 0;
      const newscore = totalscore + score;
      player.set("score", newscore);
      console.log(`initial score: ${totalscore}, new score: ${newscore}`);

    });

    // 保存全局和本回合的聚合结果
    game.set("cumulativeResults", cumulativeResultsGame);
    // stage.round.set("aggregatedResults", aggregatedResults);
  }
});

Empirica.onGameEnded(({ game }) => {
  console.log(`Game ended. Treatment: ${game.get("treatment").type}`);
});

function bernoulliRandom(p) {
  return Math.random() < p ? 1 : 0;
}