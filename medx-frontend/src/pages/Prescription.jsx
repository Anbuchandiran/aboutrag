import React, { useMemo, useRef, useState } from "react";
import axios from "axios";
import { API_BASE as API } from "../api/client";

function statusFromText(answer) {
  const a = (answer || "").toLowerCase();
  const match = a.match(/overall[\s_-]*status\s*:\s*([a-z\s_-]+)/i);
  const statusText = (match?.[1] || a).toLowerCase();

  if (statusText.includes("not safe")) return "NOT SAFE";
  if (statusText.includes("caution")) return "CAUTION";
  if (statusText.includes("insufficient")) return "INSUFFICIENT";
  if (statusText.includes("safe")) return "SAFE";
  return "RESULT";
}

function StatusBadge({ answer }) {
  const s = statusFromText(answer);
  if (s === "SAFE") return <span className="badge badgeSafe">SAFE</span>;
  if (s === "NOT SAFE") return <span className="badge badgeDanger">NOT SAFE</span>;
  if (s === "CAUTION") return <span className="badge badgeWarn">CAUTION</span>;
  if (s === "INSUFFICIENT") return <span className="badge badgeWarn">INSUFFICIENT</span>;
  return <span className="badge badgeWarn">RESULT</span>;
}

export default function Prescription() {
  const [tab, setTab] = useState("manual"); // manual | ocr | voice

  const [patientId, setPatientId] = useState("PAT005");
  const [doctorId, setDoctorId] = useState("DOC005");

  const [manualText, setManualText] = useState("");

  const [ocrFile, setOcrFile] = useState(null);
  const [voiceFile, setVoiceFile] = useState(null);

  const [extracted, setExtracted] = useState("");
  const [answer, setAnswer] = useState("");
  const [meta, setMeta] = useState(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  const [isRecording, setIsRecording] = useState(false);

  const mediaRecorderRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const chunksRef = useRef([]);

  const queryText = useMemo(() => {
    if (tab === "manual") return manualText;
    return extracted;
  }, [tab, manualText, extracted]);

  const validateWithQuery = async (query) => {
    if (!patientId.trim() || !doctorId.trim()) {
      setMsg("Please enter both Patient ID and Doctor ID.");
      return;
    }
    if (!query || !query.trim()) {
      setMsg("Please provide input text (manual, OCR, or voice).");
      return;
    }

    const payload = {
      patient_id: patientId.trim(),
      doctor_id: doctorId.trim(),
      query: query.trim(),
    };

    const res = await axios.post(`${API}/ask`, payload);
    setAnswer(res.data?.answer || "");
    setMeta(res.data);
  };

  const extractOCR = async () => {
    if (!ocrFile) return setMsg("Please select a prescription image first.");
    setBusy(true);
    setMsg("");
    setExtracted("");
    setAnswer("");
    setMeta(null);

    try {
      const fd = new FormData();
      fd.append("file", ocrFile);
      const res = await axios.post(`${API}/ocr/image`, fd);
      const text = (res.data?.text || "").trim();
      const ocrValidation = res.data?.validation || "";
      setExtracted(text);
      setMeta(res.data);

      if (res.data?.error) {
        setMsg(`OCR error: ${res.data.error}`);
      } else {
        setMsg(text ? "OCR text extracted." : "No text was detected from the image.");
        if (text) {
          setAnswer(text);
          setMsg(ocrValidation ? "OCR text extracted and validation completed." : "OCR text extracted.");
        }
      }
    } catch (e) {
      setMsg(`OCR failed: ${e?.response?.data?.error || e.message}`);
    } finally {
      setBusy(false);
    }
  };

  const transcribeAudioFile = async (file, autoValidate = true) => {
    setBusy(true);
    setMsg("");
    setExtracted("");
    setAnswer("");
    setMeta(null);

    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await axios.post(`${API}/stt/audio`, fd);
      const text = res.data?.text || "";
      setExtracted(text);
      setMeta(res.data);

      if (res.data?.error) {
        setMsg(`STT error: ${res.data.error}`);
      } else {
        setMsg(text ? "Speech converted to text. Sending for validation..." : "No text was detected from the audio.");
        if (text && autoValidate) {
          await validateWithQuery(text);
          setMsg("Voice transcription and validation completed.");
        }
      }
    } catch (e) {
      setMsg(`Speech-to-text failed: ${e?.response?.data?.error || e.message}`);
    } finally {
      setBusy(false);
    }
  };

  const extractVoice = async () => {
    if (!voiceFile) return setMsg("Please select an audio file first.");
    await transcribeAudioFile(voiceFile, true);
  };

  const startRecording = async () => {
    if (!navigator.mediaDevices || !window.MediaRecorder) {
      setMsg("Audio recording is not supported by this browser.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      chunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) chunksRef.current.push(event.data);
      };

      recorder.onstop = async () => {
        try {
          const blob = new Blob(chunksRef.current, { type: "audio/webm" });
          const file = new File([blob], "voice-recording.webm", { type: "audio/webm" });
          setVoiceFile(file);
          await transcribeAudioFile(file, true);
        } catch (e) {
          setMsg(`Recording processing failed: ${e.message}`);
        }
      };

      recorder.start();
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
      setMsg("Recording in progress...");
    } catch (e) {
      setMsg(`Microphone access failed: ${e.message}`);
    }
  };

  const stopRecording = () => {
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }

    setIsRecording(false);
    setMsg("Recording stopped. Processing audio...");
  };

  const validate = async () => {
    setBusy(true);
    setMsg("");
    setAnswer("");
    setMeta(null);

    try {
      await validateWithQuery(queryText);
      setMsg("Validation completed successfully.");
    } catch (e) {
      setMsg(`Validation failed: ${e?.response?.data?.detail || e.message}`);
    } finally {
      setBusy(false);
    }
  };

  const copyResult = async () => {
    try {
      await navigator.clipboard.writeText(answer || "");
      setMsg("Validation output copied to clipboard.");
    } catch {
      setMsg("Unable to copy output.");
    }
  };

  return (
    <>
      <div className="pageHeader">
        <div>
          <div className="hTitle">Prescription</div>
          <div className="hSub">Validate medication combinations using manual, OCR, or voice input</div>
        </div>
        <StatusBadge answer={answer} />
      </div>

      <div className="card">
        <div className="cardTitle cardTitleTight">Input and Validation</div>
        <div className="hSub">Provide prescription input and run clinical safety checks</div>
        <hr className="sep" />
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
          <button className={"tabBtn " + (tab === "manual" ? "tabBtnActive" : "")} onClick={() => setTab("manual")}>Manual</button>
          <button className={"tabBtn " + (tab === "ocr" ? "tabBtnActive" : "")} onClick={() => setTab("ocr")}>OCR Image</button>
          <button className={"tabBtn " + (tab === "voice" ? "tabBtnActive" : "")} onClick={() => setTab("voice")}>Voice</button>
        </div>

        {tab === "manual" && (
          <div className="row">
            <div>
              <div className="fieldLabel">Enter drugs</div>
              <textarea className="textarea" value={manualText} onChange={(e) => setManualText(e.target.value)} placeholder="Example: warfarin + aspirin" />
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
              <button className="btn" onClick={extractOCR} disabled={busy}>Extract Text (OCR) + Validate</button>
            </div>
          </div>
        )}

        {tab === "voice" && (
          <div className="row">
            <div>
              <div className="fieldLabel">Record voice</div>
              <div className="btnRow">
                <button className="btn" onClick={startRecording} disabled={busy || isRecording}>Start Recording</button>
                <button className="btn" onClick={stopRecording} disabled={busy || !isRecording}>Stop Recording</button>
              </div>
            </div>
            <div>
              <div className="fieldLabel">Or upload audio (webm/wav/mp3)</div>
              <input className="input" type="file" accept="audio/*" onChange={(e) => setVoiceFile(e.target.files?.[0] || null)} />
              <div className="btnRow">
                <button className="btn" onClick={extractVoice} disabled={busy}>Convert Speech + Validate</button>
              </div>
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
          <button className="btn btnPrimary" onClick={validate} disabled={busy}>{busy ? "Processing..." : "Validate"}</button>
          <button className="btn" onClick={() => { setAnswer(""); setMeta(null); setMsg(""); }}>Clear Result</button>
          <button className="btn" onClick={copyResult} disabled={!answer}>Copy Result</button>
        </div>

        {msg && <div className="toast">{msg}</div>}
      </div>

      <div className="sectionGap" />

      <div className="card">
        <div className="pageHeader compactHeader">
          <div>
            <div className="cardTitle cardTitleTight">Validation Output</div>
            <div className="hSub">Large validation responses remain visible in this scrollable panel</div>
          </div>
          <StatusBadge answer={answer} />
        </div>

        <div className="resultBox">{answer || "Run validation to view output here."}</div>

        {meta && (
          <>
            <hr className="sep" />
            <div className="cardTitle">Response Metadata (Optional)</div>
            <div className="resultBox">{JSON.stringify(meta, null, 2)}</div>
          </>
        )}
      </div>
    </>
  );
}
