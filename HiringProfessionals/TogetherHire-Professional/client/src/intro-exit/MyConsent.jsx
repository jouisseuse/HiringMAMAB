import { Consent } from "@empirica/core/player/react";

export function MyConsent({ onConsent }) {
  return (
    <div className="consent-shell">
      <div className="consent-card" role="dialog" aria-labelledby="consent-title">
        <header className="consent-header">
          <h2 id="consent-title">Study Title: Computational Social Cognition</h2>
          <p className="muted">Study Number: IRB24-1184</p>
        </header>

        <div className="consent-body" aria-describedby="consent-body">
          <div id="consent-body" className="content">
            <p>
              Welcome! You are invited to participate in a research study about how people learn and
              make decisions. This study will collect your responses by asking you to make decisions
              and answering surveys. This consent form provides information about the study’s purpose,
              procedures, risks, and benefits so that you can make an informed decision about whether
              to participate. Please take time to read the information below before beginning the study.
            </p>

            <p>
              If and when you are ready, click <strong>“Accept and Continue”</strong> to indicate that you
              voluntarily agree to participate in the study.
            </p>

            <p>
              If you have questions about the experiment in general, you may contact the Bai Lab in the
              University of Chicago Department of Psychology at <strong>uchicagobailab@gmail.com</strong>.
              If you have questions about your rights as a research participant, you may contact the
              Social &amp; Behavioral Sciences Institutional Review Board at University of Chicago at{" "}
              <strong>sbs-irb@uchicago.edu</strong>.
            </p>

            <div className="qa-grid">
              <section>
                <h4>Q: How will I be compensated?</h4>
                <p>
                  <strong>A:</strong> You will receive a minimum of $5.00 for completing the study and up to
                  $1.00 in total depending on your performance during the task.
                </p>
              </section>

              <section>
                <h4>Q: What is the purpose of this study?</h4>
                <p>
                  <strong>A:</strong> This study aims to help us understand how people learn and make decisions.
                </p>
              </section>

              <section>
                <h4>Q: What will I be doing today, and how long will it take?</h4>
                <p>
                  <strong>A:</strong> During this study, you will see visual stimuli (e.g., images, words, shapes)
                  presented on a computer screen. You will respond according to the task instructions. You will
                  also be asked to complete a series of sequential decisions and questionnaires. The length of
                  your participation will be as advertised by the experimenter, and you will be compensated at
                  the rate specified for this online study. If you decide to withdraw, data collected up until
                  the point of withdrawal may still be included in analysis.
                </p>
              </section>

              <section>
                <h4>Q: What are the potential benefits and risks?</h4>
                <p>
                  <strong>A:</strong> Taking part in this research study may not benefit you personally beyond
                  compensation, but we may learn new things that could help others. Your participation in this
                  study does not involve any risk to you beyond that of everyday interactions with computer
                  reading (e.g. potential eye strain, wrist strain).
                </p>
              </section>

              <section className="full">
                <h4>Q: Will my personal information remain confidential?</h4>
                <p>
                  <strong>A:</strong> Your CloudResearch ID will be used to distribute payment to you but will not
                  be stored with the research data we collect from you. Please be aware that your CloudResearch ID
                  can potentially be linked to information about you. We will not be accessing any personally
                  identifying information about you. De-identified data from this study may be used for future
                  research studies or shared with other researchers for future research without your additional
                  informed consent.
                </p>
              </section>
            </div>

            <p className="final-note">
              To continue, click <strong>“Accept and Continue”</strong>. By continuing, you are acknowledging that you
              have read this form and agree to participate in the research study.
            </p>
          </div>
        </div>

        <div className="consent-actions">
          {/* Why: Primary action is explicit; add a secondary if you want to handle declines. */}
          {/* <button type="button" className="ghost-btn" onClick={() => {/* your decline flow */ /*}}>Decline and Exit</button> */}
          <button type="button" className="accept-btn" onClick={onConsent}>
            Accept and Continue
          </button>
        </div>
      </div>

      {/* Scoped styles for the consent card */}
      <style>{`
        .consent-shell {
          min-height: 100vh;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 24px;
          background: #f6f7fb;
        }
        .consent-card {
          width: min(100%, 1000px);
          border-radius: 16px;
          background: #fff;
          box-shadow: 0 12px 32px rgba(0,0,0,0.12);
          display: flex;
          flex-direction: column;
          max-height: 90vh; /* keeps the whole card from becoming too tall */
        }
        .consent-header {
          padding: 24px 28px 8px;
          border-bottom: 1px solid #eef0f3;
        }
        .consent-header h2 {
          margin: 0 0 4px 0;
          font-size: 22px;
        }
        .muted { margin: 0; color: #6b7280; }
        .consent-body {
          padding: 16px 28px;
          overflow: auto;           /* scrolls only the body */
          max-height: 60vh;         /* makes the window feel shorter */
        }
        .content { line-height: 1.55; }
        .content p { margin: 10px 0; }
        .qa-grid {
          display: grid;
          grid-template-columns: 1fr;
          gap: 16px 24px;
          margin-top: 12px;
        }
        .qa-grid section h4 { margin: 0 0 6px; }
        .qa-grid section p { margin: 0; }
        @media (min-width: 900px) {
          .qa-grid { grid-template-columns: 1fr 1fr; }
          .qa-grid .full { grid-column: 1 / -1; }
        }
        .final-note { margin-top: 16px; }
        .consent-actions {
          position: sticky;         /* keeps buttons visible while scrolling body */
          bottom: 0;
          display: flex;
          justify-content: flex-end;
          gap: 12px;
          padding: 14px 28px 20px;
          border-top: 1px solid #eef0f3;
          background: #fff;
          border-bottom-left-radius: 16px;
          border-bottom-right-radius: 16px;
        }
        .accept-btn {
          padding: 10px 16px;
          border: 0;
          border-radius: 9999px;
          font-weight: 600;
          background: #111827;
          color: #fff;
          cursor: pointer;
        }
        .accept-btn:hover { filter: brightness(0.95); }
        .accept-btn:focus-visible { outline: 3px solid #93c5fd; outline-offset: 2px; }
        .ghost-btn {
          padding: 10px 14px;
          background: transparent;
          border: 1px solid #d1d5db;
          border-radius: 9999px;
          color: #374151;
          cursor: pointer;
        }
        .ghost-btn:hover { background: #f3f4f6; }
      `}</style>
    </div>
  );
}