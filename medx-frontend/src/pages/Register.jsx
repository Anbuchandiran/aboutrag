import React, { useState } from "react";
import axios from "axios";
import { API_BASE as API } from "../api/client";

export default function Register() {
  const [patient, setPatient] = useState({
    patient_id: "",
    name: "",
    age: 0,
    gender: "Male",
    phone: "",
    chronic_conditions: "",
    allergies: "",
    notes: "",
  });

  const [doctor, setDoctor] = useState({
    doctor_id: "",
    name: "",
    department: "",
    phone: "",
    email: "",
  });

  const [msg, setMsg] = useState("");

  const savePatient = async () => {
    setMsg("");
    const payload = {
      patient_id: patient.patient_id.trim(),
      name: patient.name.trim(),
      age: Number(patient.age || 0),
      gender: patient.gender,
      phone: patient.phone.trim(),
      chronic_conditions: patient.chronic_conditions
        ? patient.chronic_conditions.split(",").map((s) => s.trim()).filter(Boolean)
        : [],
      allergies: patient.allergies
        ? patient.allergies.split(",").map((s) => s.trim()).filter(Boolean)
        : [],
      notes: patient.notes || "",
    };

    await axios.post(`${API}/patients/upsert`, payload);
    setMsg(`Patient saved: ${payload.patient_id}`);
  };

  const saveDoctor = async () => {
    setMsg("");
    const payload = {
      doctor_id: doctor.doctor_id.trim(),
      name: doctor.name.trim(),
      department: doctor.department.trim(),
      phone: doctor.phone.trim(),
      email: doctor.email.trim(),
    };

    await axios.post(`${API}/doctors/upsert`, payload);
    setMsg(`Doctor saved: ${payload.doctor_id}`);
  };

  return (
    <>
      <div className="pageHeader">
        <div>
          <div className="hTitle">Register</div>
          <div className="hSub">Create and maintain master data for patients and doctors</div>
        </div>
        <span className="badge">MongoDB Required</span>
      </div>

      <div className="grid2">
        <div className="card">
          <div className="cardTitle">Patient</div>
          <div className="hSub">Demographics, clinical background, and notes</div>
          <hr className="sep" />
          <div className="row">
            <div>
              <div className="fieldLabel">Patient ID</div>
              <input className="input" value={patient.patient_id}
                onChange={(e) => setPatient({ ...patient, patient_id: e.target.value })} placeholder="PAT005" />
            </div>
            <div>
              <div className="fieldLabel">Name</div>
              <input className="input" value={patient.name}
                onChange={(e) => setPatient({ ...patient, name: e.target.value })} placeholder="Anbu Kumar" />
            </div>
            <div className="grid2">
              <div>
                <div className="fieldLabel">Age</div>
                <input className="input" type="number" value={patient.age}
                  onChange={(e) => setPatient({ ...patient, age: e.target.value })} />
              </div>
              <div>
                <div className="fieldLabel">Gender</div>
                <select className="input" value={patient.gender}
                  onChange={(e) => setPatient({ ...patient, gender: e.target.value })}>
                  <option>Male</option>
                  <option>Female</option>
                  <option>Other</option>
                </select>
              </div>
            </div>

            <div>
              <div className="fieldLabel">Phone</div>
              <input className="input" value={patient.phone}
                onChange={(e) => setPatient({ ...patient, phone: e.target.value })} placeholder="9876543210" />
            </div>

            <div>
              <div className="fieldLabel">Chronic Conditions (comma separated)</div>
              <input className="input" value={patient.chronic_conditions}
                onChange={(e) => setPatient({ ...patient, chronic_conditions: e.target.value })} placeholder="diabetes, hypertension" />
            </div>

            <div>
              <div className="fieldLabel">Allergies (comma separated)</div>
              <input className="input" value={patient.allergies}
                onChange={(e) => setPatient({ ...patient, allergies: e.target.value })} placeholder="aspirin" />
            </div>

            <div>
              <div className="fieldLabel">Notes</div>
              <textarea className="textarea" value={patient.notes}
                onChange={(e) => setPatient({ ...patient, notes: e.target.value })} placeholder="Optional notes" />
            </div>

            <div className="btnRow">
              <button className="btn btnPrimary" onClick={savePatient}>Save Patient</button>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="cardTitle">Doctor</div>
          <div className="hSub">Provider identity and department details</div>
          <hr className="sep" />
          <div className="row">
            <div>
              <div className="fieldLabel">Doctor ID</div>
              <input className="input" value={doctor.doctor_id}
                onChange={(e) => setDoctor({ ...doctor, doctor_id: e.target.value })} placeholder="DOC005" />
            </div>
            <div>
              <div className="fieldLabel">Name</div>
              <input className="input" value={doctor.name}
                onChange={(e) => setDoctor({ ...doctor, name: e.target.value })} placeholder="Dr. Raj" />
            </div>
            <div>
              <div className="fieldLabel">Department</div>
              <input className="input" value={doctor.department}
                onChange={(e) => setDoctor({ ...doctor, department: e.target.value })} placeholder="Cardiology" />
            </div>
            <div className="grid2">
              <div>
                <div className="fieldLabel">Phone</div>
                <input className="input" value={doctor.phone}
                  onChange={(e) => setDoctor({ ...doctor, phone: e.target.value })} placeholder="9876543210" />
              </div>
              <div>
                <div className="fieldLabel">Email</div>
                <input className="input" value={doctor.email}
                  onChange={(e) => setDoctor({ ...doctor, email: e.target.value })} placeholder="doctor@example.com" />
              </div>
            </div>

            <div className="btnRow">
              <button className="btn btnPrimary" onClick={saveDoctor}>Save Doctor</button>
            </div>
          </div>
        </div>
      </div>

      {msg && <div className="toast">{msg}</div>}
    </>
  );
}
