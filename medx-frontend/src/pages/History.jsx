import React, { useMemo, useState } from "react";
import axios from "axios";
import { API_BASE as API } from "../api/client";

function getErrorMessage(error, fallback) {
  return error?.response?.data?.detail || error?.response?.data?.error || error?.message || fallback;
}

function formatDate(value) {
  if (!value) return "Unknown time";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("en-IN", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function toTitleCase(value) {
  return String(value || "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function summarizeRecord(record) {
  return (
    record.complaint ||
    record.diagnosis ||
    record.prescription_text ||
    record.notes ||
    "Consultation record"
  );
}

function extractMetaEntries(record) {
  return Object.entries(record || {}).filter(([key, value]) => {
    if (
      [
        "_id",
        "created_at",
        "complaint",
        "diagnosis",
        "prescription_text",
        "notes",
        "doctor",
        "patient",
        "doctor_id",
        "patient_id",
      ].includes(key)
    ) {
      return false;
    }

    if (value == null || value === "") return false;
    if (typeof value === "object" && !Array.isArray(value) && Object.keys(value).length === 0) return false;
    if (Array.isArray(value) && value.length === 0) return false;
    return true;
  });
}

function renderValue(value) {
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  return String(value);
}

function HistoryCard({ item, mode }) {
  const counterpart = mode === "patient" ? item.doctor : item.patient;
  const counterpartLabel = mode === "patient" ? "Doctor" : "Patient";
  const metaEntries = extractMetaEntries(item);

  return (
    <article className="historyCard">
      <div className="historyCardTop">
        <div>
          <div className="historyCardTitle">{summarizeRecord(item)}</div>
          <div className="historyCardSub">{formatDate(item.created_at)}</div>
        </div>
        <span className="badge badgeGhost">{counterpartLabel} record</span>
      </div>

      <div className="historyIdentity">
        <div className="identityAvatar">{mode === "patient" ? "DR" : "PT"}</div>
        <div>
          <div className="identityName">
            {counterpart?.name || counterpart?.doctor_id || counterpart?.patient_id || "Linked profile unavailable"}
          </div>
          <div className="identityMeta">
            {mode === "patient"
              ? `${counterpart?.doctor_id || item.doctor_id || "Unknown ID"}${counterpart?.department ? ` • ${counterpart.department}` : ""}`
              : `${counterpart?.patient_id || item.patient_id || "Unknown ID"}${counterpart?.age ? ` • ${counterpart.age} yrs` : ""}${counterpart?.gender ? ` • ${counterpart.gender}` : ""}`}
          </div>
        </div>
      </div>

      <div className="historySections">
        {item.complaint && (
          <div className="historySection">
            <span className="historySectionLabel">Complaint</span>
            <p>{item.complaint}</p>
          </div>
        )}

        {item.diagnosis && (
          <div className="historySection">
            <span className="historySectionLabel">Diagnosis</span>
            <p>{item.diagnosis}</p>
          </div>
        )}

        {item.prescription_text && (
          <div className="historySection">
            <span className="historySectionLabel">Prescription</span>
            <p>{item.prescription_text}</p>
          </div>
        )}

        {item.notes && (
          <div className="historySection">
            <span className="historySectionLabel">Notes</span>
            <p>{item.notes}</p>
          </div>
        )}
      </div>

      {metaEntries.length > 0 && (
        <div className="historyMetaGrid">
          {metaEntries.map(([key, value]) => (
            <div key={key} className="historyMetaItem">
              <span>{toTitleCase(key)}</span>
              <strong>{renderValue(value)}</strong>
            </div>
          ))}
        </div>
      )}
    </article>
  );
}

function HistoryPanel({
  title,
  subtitle,
  value,
  onChange,
  onLoad,
  loading,
  records,
  emptyText,
  mode,
}) {
  return (
    <section className="card historyPanel">
      <div className="historyPanelHead">
        <div>
          <div className="cardTitle cardTitleTight">{title}</div>
          <div className="hSub">{subtitle}</div>
        </div>
        <span className="badge badgeGhost">{records.length} items</span>
      </div>

      <div className="historySearchRow">
        <input
          className="input historySearchInput"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") onLoad();
          }}
          placeholder={mode === "patient" ? "Enter patient ID" : "Enter doctor ID"}
        />
        <button className="btn btnPrimary" onClick={onLoad} disabled={loading}>
          {loading ? "Loading..." : "Load history"}
        </button>
      </div>

      <div className="historyTimeline">
        {records.length ? (
          records.map((item) => <HistoryCard key={item._id || `${mode}-${item.created_at}-${summarizeRecord(item)}`} item={item} mode={mode} />)
        ) : (
          <div className="historyEmpty">{emptyText}</div>
        )}
      </div>
    </section>
  );
}

export default function History() {
  const [patientId, setPatientId] = useState("PAT005");
  const [doctorId, setDoctorId] = useState("DOC005");
  const [patientHistory, setPatientHistory] = useState([]);
  const [doctorHistory, setDoctorHistory] = useState([]);
  const [patientLoading, setPatientLoading] = useState(false);
  const [doctorLoading, setDoctorLoading] = useState(false);
  const [msg, setMsg] = useState("");

  const patientTotal = useMemo(() => patientHistory.length, [patientHistory]);
  const doctorTotal = useMemo(() => doctorHistory.length, [doctorHistory]);

  const loadPatient = async () => {
    if (!patientId.trim()) {
      setMsg("Enter a patient ID before loading history.");
      return;
    }

    setPatientLoading(true);
    setMsg("");
    try {
      const res = await axios.get(`${API}/patients/${patientId.trim()}/history?limit=20`);
      const history = Array.isArray(res.data?.history) ? res.data.history : [];
      setPatientHistory(history);
      setMsg(history.length ? "Patient history loaded successfully." : "No patient visits found for that ID yet.");
    } catch (error) {
      setPatientHistory([]);
      setMsg(getErrorMessage(error, "Failed to load patient history."));
    } finally {
      setPatientLoading(false);
    }
  };

  const loadDoctor = async () => {
    if (!doctorId.trim()) {
      setMsg("Enter a doctor ID before loading history.");
      return;
    }

    setDoctorLoading(true);
    setMsg("");
    try {
      const res = await axios.get(`${API}/doctors/${doctorId.trim()}/history?limit=20`);
      const history = Array.isArray(res.data?.history) ? res.data.history : [];
      setDoctorHistory(history);
      setMsg(history.length ? "Doctor history loaded successfully." : "No doctor consultations found for that ID yet.");
    } catch (error) {
      setDoctorHistory([]);
      setMsg(getErrorMessage(error, "Failed to load doctor history."));
    } finally {
      setDoctorLoading(false);
    }
  };

  return (
    <>
      <div className="pageHeader pageHeaderSplit">
        <div>
          <div className="eyebrow">Clinical Memory</div>
          <div className="hTitle">History</div>
          <div className="hSub">Track previous visits, consultation notes, and linked doctor or patient context in a cleaner timeline view.</div>
        </div>
        <div className="headerStats">
          <div className="statChip">
            <strong>{patientTotal}</strong>
            <span>Patient records</span>
          </div>
          <div className="statChip">
            <strong>{doctorTotal}</strong>
            <span>Doctor records</span>
          </div>
        </div>
      </div>

      <div className="heroCard">
        <div className="heroCardContent">
          <div>
            <div className="heroTitle">Search memory without the clutter</div>
            <div className="heroSub">
              The history page now surfaces visits as readable cards instead of raw JSON, while keeping all metadata available when the backend returns extra fields.
            </div>
          </div>
          <div className="heroPills">
            <span className="miniPill">Timeline cards</span>
            <span className="miniPill">Better errors</span>
            <span className="miniPill">Enter-to-search</span>
          </div>
        </div>
      </div>

      <div className="grid2">
        <HistoryPanel
          title="Patient history"
          subtitle="Review prior visits and the doctors involved in each case."
          value={patientId}
          onChange={setPatientId}
          onLoad={loadPatient}
          loading={patientLoading}
          records={patientHistory}
          emptyText="Patient visits will appear here after you search."
          mode="patient"
        />

        <HistoryPanel
          title="Doctor history"
          subtitle="See consultation activity and linked patient context for the selected doctor."
          value={doctorId}
          onChange={setDoctorId}
          onLoad={loadDoctor}
          loading={doctorLoading}
          records={doctorHistory}
          emptyText="Doctor consultations will appear here after you search."
          mode="doctor"
        />
      </div>

      {msg && <div className="toast">{msg}</div>}
    </>
  );
}
