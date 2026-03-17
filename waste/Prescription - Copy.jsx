import React, { useMemo, useState } from "react";
import axios from "axios";

const API = import.meta.env.VITE_API_BASE || "http://localhost:8000";

function statusFromText(answer) {
  const a = (answer || "").toLowerCase();
  if (a.includes("overall_status") && a.includes("not safe")) return "NOT SAFE";
  if (a.includes("not safe")) return "NOT SAFE";
  if (a.includes("safe")) return "SAFE";
  return "RESULT";
}

function StatusBadge({ answer }) {
  const s = statusFromText(answer);
  if (s === "SAFE") return <span className="badge badgeSafe">SAFE</span>;
  if (s === "NOT SAFE") return <span className="badge badgeDanger">NOT SAFE</span>;
  return <span className="badge badgeWarn">RESULT</span>;
}

export default function Prescription() {
  const [tab, setTab] = useState("manual"); // manual | ocr | voice

  const [patientId, setPatientId] = useState("PAT005");
  const [doctorId, setDoctorId] = useState("DOC005");

  const [manualText, setManualText] = useState("warfarin + aspirin");

  const [ocrFile, setOcrFile] = useState(null);
  const [voiceFile, setVoiceFile] = useState(null);

  const [extracted, setExtracted] = useState("");
  const [answer, setAnswer] = useState("");
  const [meta, setMeta] = useState(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  const queryText = useMemo(() => {
    if (tab === "manual") return manualText;
    return extracted;
  }, [tab, manualText, extracted]);

  const extractOCR = async () => {
    if (!ocrFile) return setMsg("Select an image first.");
    setBusy(true);
    setMsg("");
    setExtracted("");
    setAnswer("");
    setMeta(null);

    try {
      const fd = new FormData();
      fd.append("file", ocrFile);
      const res = await axios.post(`${API}/ocr/image`, fd);
      setExtracted(res.data?.text || "");
      setMeta(res.data);
      if (res.data?.error) {
        setMsg(`OCR error: ${res.data.error}`);
      } else {
        setMsg(res.data?.text ? "OCR extracted text." : "OCR returned empty.");
      }
    } catch (e) {
      setMsg(`OCR failed: ${e?.response?.data?.error || e.message}`);
    } finally {
      setBusy(false);
    }
  };

  const extractVoice = async () => {
    if (!voiceFile) return setMsg("Select an audio file first.");
    setBusy(true);
    setMsg("");
    setExtracted("");
    setAnswer("");
    setMeta(null);

    try {
      const fd = new FormData();
      fd.append("file", voiceFile);
      const res = await axios.post(`${API}/stt/audio`, fd);
      setExtracted(res.data?.text || "");
      setMeta(res.data);
      if (res.data?.error) {
        setMsg(`STT error: ${res.data.error}`);
      } else {
        setMsg(res.data?.text ? "Speech converted to text." : "STT returned empty.");
      }
    } catch (e) {
      setMsg(`Speech-to-text failed: ${e?.response?.data?.error || e.message}`);
    } finally {
      setBusy(false);
    }
  };

  const validate = async () => {
    if (!patientId.trim() || !doctorId.trim()) return setMsg("Enter Patient ID and Doctor ID.");
    if (!queryText || !queryText.trim()) return setMsg("Provide input text (manual/OCR/voice).");

    setBusy(true);
    setMsg("");
    setAnswer("");
    setMeta(null);

    try {
      const payload = {
        patient_id: patientId.trim(),
        doctor_id: doctorId.trim(),
        query: queryText.trim(),
      };
      const res = await axios.post(`${API}/ask`, payload);
      setAnswer(res.data?.answer || "");
      setMeta(res.data);
      setMsg("Validation complete.");
    } catch (e) {
      setMsg(`Validation failed: ${e?.response?.data?.detail || e.message}`);
    } finally {
      setBusy(false);
    }
  };

  const copyResult = async () => {
    try {
      await navigator.clipboard.writeText(answer || "");
      setMsg("Copied result to clipboard.");
    } catch {
      setMsg("Copy failed.");
    }
  };

  return (
    <>
      <div className="pageHeader">
        <div>
          <div className="hTitle">Prescription</div>
          <div className="hSub">Manual / OCR / Voice -> Validate with RAG + Patient Context</div>
        </div>
        <StatusBadge answer={answer} />
      </div>

      <div className="card">
        <div className="grid2">
          <div>
            <div className="fieldLabel">Patient ID</div>
            <input className="input" value={patientId} onChange={(e) => setPatientId(e.target.value)} />
          </div>
          <div>
            <div className="fieldLabel">Doctor ID</div>
            <input className="input" value={doctorId} onChange={(e) => setDoctorId(e.target.value)} />
          </div>
        </div>

        <hr className="sep" />

        <div className="tabs">
          <button className={"tabBtn " + (tab === "manual" ? "tabBtnActive" : "")} onClick={() => setTab("manual")}>
            Manual
          </button>
          <button className={"tabBtn " + (tab === "ocr" ? "tabBtnActive" : "")} onClick={() => setTab("ocr")}>
            OCR Image
          </button>
          <button className={"tabBtn " + (tab === "voice" ? "tabBtnActive" : "")} onClick={() => setTab("voice")}>
            Voice
          </button>
        </div>

        {tab === "manual" && (
          <div className="row">
            <div>
              <div className="fieldLabel">Enter drugs</div>
              <textarea
                className="textarea"
                value={manualText}
                onChange={(e) => setManualText(e.target.value)}
                placeholder="Example: warfarin + aspirin"
              />
            </div>
          </div>
        )}

        {tab === "ocr" && (
          <div className="row">
            <div>
              <div className="fieldLabel">Upload prescription image</div>
              <input className="input" type="file" accept="image/*" onChange={(e) => setOcrFile(e.target.files?.[0] || null)} />
            </div>
            <div className="btnRow">
              <button className="btn" onClick={extractOCR} disabled={busy}>
                Extract Text (OCR)
              </button>
            </div>
          </div>
        )}

        {tab === "voice" && (
          <div className="row">
            <div>
              <div className="fieldLabel">Upload audio (webm/wav/mp3)</div>
              <input className="input" type="file" accept="audio/*" onChange={(e) => setVoiceFile(e.target.files?.[0] || null)} />
            </div>
            <div className="btnRow">
              <button className="btn" onClick={extractVoice} disabled={busy}>
                Convert Speech -> Text
              </button>
            </div>
          </div>
        )}

        {(tab === "ocr" || tab === "voice") && (
          <>
            <hr className="sep" />
            <div className="cardTitle">Extracted Text</div>
            <div className="resultBox">{extracted || "-"}</div>
          </>
        )}

        <div className="btnRow">
          <button className="btn btnPrimary" onClick={validate} disabled={busy}>
            {busy ? "Working..." : "Validate"}
          </button>
          <button className="btn" onClick={() => { setAnswer(""); setMeta(null); setMsg(""); }}>
            Clear Result
          </button>
          <button className="btn" onClick={copyResult} disabled={!answer}>
            Copy Result
          </button>
        </div>

        {msg && <div className="toast">{msg}</div>}
      </div>

      <div style={{ height: 14 }} />

      <div className="card">
        <div className="pageHeader" style={{ marginBottom: 10 }}>
          <div>
            <div className="cardTitle" style={{ marginBottom: 0 }}>Validation Output</div>
            <div className="hSub">Scrollable panel (large outputs will not disappear)</div>
          </div>
          <StatusBadge answer={answer} />
        </div>

        <div className="resultBox">{answer || "Run validation to see output here..."}</div>

        {meta && (
          <>
            <hr className="sep" />
            <div className="cardTitle">Debug / Response Meta (optional)</div>
            <div className="resultBox">{JSON.stringify(meta, null, 2)}</div>
          </>
        )}
      </div>
    </>
  );
}
