import React from "react";
import { API_BASE } from "../api/client";

export default function Dashboard() {
  return (
    <>
      <div className="pageHeader">
        <div>
          <div className="hTitle">Dashboard</div>
          <div className="hSub">Unified clinical workflow for registration, validation, and case review</div>
        </div>
        <span className="badge badgeSafe">System Ready</span>
      </div>

      <div className="grid3">
        <div className="card">
          <div className="cardTitle">Workflow</div>
          <div className="hSub">Register {"->"} Validate {"->"} Review History</div>
        </div>
        <div className="card">
          <div className="cardTitle">Input Channels</div>
          <div className="hSub">Manual Entry, OCR, Voice Transcription</div>
        </div>
        <div className="card">
          <div className="cardTitle">Platform</div>
          <div className="hSub">FastAPI + MongoDB + Chroma</div>
        </div>
      </div>

      <div className="sectionGap" />

      <div className="grid2">
        <div className="card">
          <div className="cardTitle">Quick Start</div>
          <div className="row">
            <div className="listItem">
              <span className="listIndex">1</span>
              <div>
                <div className="listTitle">Register Profiles</div>
                <div className="listSub">Add patient and doctor records in the Register page.</div>
              </div>
            </div>
            <div className="listItem">
              <span className="listIndex">2</span>
              <div>
                <div className="listTitle">Validate Prescription</div>
                <div className="listSub">Run manual, OCR, or voice validation from Prescription.</div>
              </div>
            </div>
            <div className="listItem">
              <span className="listIndex">3</span>
              <div>
                <div className="listTitle">Review Clinical Memory</div>
                <div className="listSub">Inspect prior patient and doctor cases in History.</div>
              </div>
            </div>
          </div>
          <hr className="sep" />
          <div className="hSub">
            Active API endpoint: <b>{API_BASE}</b>
          </div>
        </div>

        <div className="card">
          <div className="cardTitle">Validation Output Preview</div>
          <div className="hSub">Typical structured response format</div>
          <hr className="sep" />
          <div className="resultBox">
{`Overall_Status: NOT SAFE
Key_Reason: ...
Interactions:
- ...
Action: ...`}
          </div>
        </div>
      </div>
    </>
  );
}
