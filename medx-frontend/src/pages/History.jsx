import React, { useState } from "react";
import axios from "axios";

const API = "http://127.0.0.1:8000";

export default function History() {
  const [patientId, setPatientId] = useState("PAT005");
  const [doctorId, setDoctorId] = useState("DOC005");
  const [patientHistory, setPatientHistory] = useState([]);
  const [doctorHistory, setDoctorHistory] = useState([]);
  const [msg, setMsg] = useState("");

  const loadPatient = async () => {
    setMsg("");
    try {
      const res = await axios.get(`${API}/patients/${patientId}/history?limit=20`);
      setPatientHistory(res.data?.history || []);
      setMsg("Patient history loaded successfully.");
    } catch {
      setMsg("Failed to load patient history.");
    }
  };

  const loadDoctor = async () => {
    setMsg("");
    try {
      const res = await axios.get(`${API}/doctors/${doctorId}/history?limit=20`);
      setDoctorHistory(res.data?.history || []);
      setMsg("Doctor history loaded successfully.");
    } catch {
      setMsg("Failed to load doctor history.");
    }
  };

  return (
    <>
      <div className="pageHeader">
        <div>
          <div className="hTitle">History</div>
          <div className="hSub">Search patient and doctor records from previous consultations</div>
        </div>
        <span className="badge">Clinical Memory</span>
      </div>

      <div className="grid2">
        <div className="card">
          <div className="cardTitle">Patient History</div>
          <div className="hSub">Retrieve patient-specific visits and validation logs</div>
          <hr className="sep" />
          <div className="row">
            <div>
              <div className="fieldLabel">Patient ID</div>
              <input className="input" value={patientId} onChange={(e) => setPatientId(e.target.value)} />
            </div>
            <div className="btnRow">
              <button className="btn btnPrimary" onClick={loadPatient}>Load</button>
            </div>
            <div className="resultBox">
              {patientHistory.length ? JSON.stringify(patientHistory, null, 2) : "-"}
            </div>
          </div>
        </div>

        <div className="card">
          <div className="cardTitle">Doctor History</div>
          <div className="hSub">Retrieve consultations and solved cases by doctor</div>
          <hr className="sep" />
          <div className="row">
            <div>
              <div className="fieldLabel">Doctor ID</div>
              <input className="input" value={doctorId} onChange={(e) => setDoctorId(e.target.value)} />
            </div>
            <div className="btnRow">
              <button className="btn btnPrimary" onClick={loadDoctor}>Load</button>
            </div>
            <div className="resultBox">
              {doctorHistory.length ? JSON.stringify(doctorHistory, null, 2) : "-"}
            </div>
          </div>
        </div>
      </div>

      {msg && <div className="toast">{msg}</div>}
    </>
  );
}
