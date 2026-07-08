import { Consent } from "@empirica/core/player/react";

export function MyConsent({ onConsent }) {
  console.log("test");
  return (
    <Consent
      title="Do you consent to participate in this experiment?"
      text="This experiment is part of a scientific project, approved by the Institutional Review Board at the University of Chicago, NO. 24-1184."
      buttonText="I consent."
      onConsent={onConsent}
    />
  );
}
