import { EmpiricaClassic } from "@empirica/core/player/classic";
import { EmpiricaContext } from "@empirica/core/player/classic/react";
import { EmpiricaMenu, EmpiricaParticipant } from "@empirica/core/player/react";
import React from "react";
import { Game } from "./Game";
import { ExitSurvey } from "./intro-exit/ExitSurvey";
import { Introduction } from "./intro-exit/Introduction";
import { MyConsent } from "./intro-exit/MyConsent";
import { MyLobby } from "./components/MyLobby.jsx";
import { LoadingOverlay } from "./components/LoadingOverlay.jsx";


export default function App() {
  const urlParams = new URLSearchParams(window.location.search);
  const playerKey = urlParams.get("participantKey") || "";

  const { protocol, host } = window.location;
  const url = `${protocol}//${host}/query`;

  function introSteps({ game, player }) {
    return [
      (props) => <Introduction {...props}/>, // The global introduction page
    ];
  }

  function exitSteps({ game, player }) {
    return [ExitSurvey];
  }


  return (
    <EmpiricaParticipant url={url} ns={playerKey} modeFunc={EmpiricaClassic}>
      <div className="h-screen relative">
        <EmpiricaMenu position="bottom-left" />
        <div className="h-full overflow-auto">
          <EmpiricaContext
            consent={MyConsent}
            introSteps={introSteps}
            exitSteps={exitSteps}
            lobby={MyLobby}                        // Custom lobby component
            connecting={() => <LoadingOverlay />}  // Connecting to server
            loading={() => <LoadingOverlay />}     // Loading game data
          >  
            <Game />
          </EmpiricaContext>
        </div>
      </div>
    </EmpiricaParticipant>
  );
}