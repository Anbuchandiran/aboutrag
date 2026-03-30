const patientIdInput = document.getElementById("patientId");
const doctorIdInput = document.getElementById("doctorId");
const imageInput = document.getElementById("imageInput");
const queryInput = document.getElementById("query");
const queryMeta = document.getElementById("queryMeta");
const extractBtn = document.getElementById("extractBtn");
const validateBtn = document.getElementById("validateBtn");
const statusDiv = document.getElementById("status");
const resultCard = document.getElementById("resultCard");
const resultBadge = document.getElementById("resultBadge");
const resultSummary = document.getElementById("resultSummary");
const resultText = document.getElementById("resultText");

const API_BASE = "https://pursuant-pearl-semiopenly.ngrok-free.dev";

function setStatus(message, type = "idle") {
  statusDiv.textContent = message;
  statusDiv.className = `status status-${type}`;
}

function setBusy(button, label, busyLabel, isBusy) {
  button.disabled = isBusy;
  button.textContent = isBusy ? busyLabel : label;
}

function syncQueryMeta() {
  queryMeta.textContent = `${queryInput.value.trim().length} chars`;
}

function getAnswerStatus(answer) {
  const text = String(answer || "").toUpperCase();
  if (text.includes("NOT SAFE")) return "NOT SAFE";
  if (text.includes("CAUTION")) return "CAUTION";
  if (text.includes("SAFE")) return "SAFE";
  return "INSUFFICIENT";
}

function setBadge(answer) {
  const status = getAnswerStatus(answer);
  resultBadge.className = "badge";

  if (status === "NOT SAFE") {
    resultBadge.textContent = status;
    resultBadge.classList.add("notsafe");
  } else if (status === "CAUTION") {
    resultBadge.textContent = status;
    resultBadge.classList.add("caution");
  } else if (status === "SAFE") {
    resultBadge.textContent = status;
    resultBadge.classList.add("safe");
  } else {
    resultBadge.textContent = status;
    resultBadge.classList.add("insufficient");
  }
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function parseAnswer(answer) {
  const text = String(answer || "").trim();
  const lines = text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  const sections = {
    overallStatus: "",
    keyReason: "",
    interactions: [],
    action: "",
    other: [],
  };

  let activeSection = "";

  for (const line of lines) {
    const lower = line.toLowerCase();

    if (lower.startsWith("overall_status:") || lower.startsWith("overall status:")) {
      sections.overallStatus = line.split(":").slice(1).join(":").trim();
      activeSection = "";
      continue;
    }

    if (lower.startsWith("key_reason:") || lower.startsWith("key reason:")) {
      sections.keyReason = line.split(":").slice(1).join(":").trim();
      activeSection = "";
      continue;
    }

    if (lower.startsWith("interactions:")) {
      const value = line.split(":").slice(1).join(":").trim();
      if (value) sections.interactions.push(value);
      activeSection = "interactions";
      continue;
    }

    if (lower.startsWith("action:")) {
      sections.action = line.split(":").slice(1).join(":").trim();
      activeSection = "action";
      continue;
    }

    if (line.startsWith("-") || line.startsWith("*")) {
      const item = line.slice(1).trim();
      if (activeSection === "interactions") {
        sections.interactions.push(item);
      } else if (activeSection === "action") {
        sections.action = sections.action ? `${sections.action} ${item}` : item;
      } else {
        sections.other.push(item);
      }
      continue;
    }

    if (activeSection === "interactions") {
      sections.interactions.push(line);
    } else if (activeSection === "action") {
      sections.action = sections.action ? `${sections.action} ${line}` : line;
    } else {
      sections.other.push(line);
    }
  }

  return sections;
}

function renderStructuredAnswer(answer) {
  const parsed = parseAnswer(answer);
  const cards = [];

  if (parsed.overallStatus || getAnswerStatus(answer) !== "INSUFFICIENT") {
    cards.push(`
      <div class="summary-card">
        <span class="summary-label">Overall Status</span>
        <strong>${escapeHtml(parsed.overallStatus || getAnswerStatus(answer))}</strong>
      </div>
    `);
  }

  if (parsed.keyReason) {
    cards.push(`
      <div class="summary-card">
        <span class="summary-label">Key Reason</span>
        <strong>${escapeHtml(parsed.keyReason)}</strong>
      </div>
    `);
  }

  if (parsed.action) {
    cards.push(`
      <div class="summary-card">
        <span class="summary-label">Recommended Action</span>
        <strong>${escapeHtml(parsed.action)}</strong>
      </div>
    `);
  }

  if (cards.length) {
    resultSummary.innerHTML = cards.join("");
    resultSummary.classList.remove("hidden");
  } else {
    resultSummary.innerHTML = "";
    resultSummary.classList.add("hidden");
  }

  const interactionHtml = parsed.interactions.length
    ? `<div class="result-section">
         <div class="result-section-title">Interactions</div>
         <ul class="result-list">${parsed.interactions.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
       </div>`
    : "";

  const otherHtml = parsed.other.length
    ? `<div class="result-section">
         <div class="result-section-title">Clinical Notes</div>
         <div class="result-copy">${parsed.other.map((item) => `<p>${escapeHtml(item)}</p>`).join("")}</div>
       </div>`
    : "";

  if (interactionHtml || otherHtml) {
    resultText.innerHTML = `${interactionHtml}${otherHtml}`;
  } else {
    resultText.innerHTML = `<div class="result-copy"><p>${escapeHtml(answer || "No response received.")}</p></div>`;
  }
}

function showResult(answer, structured = true) {
  setBadge(answer);
  if (structured) {
    renderStructuredAnswer(answer);
  } else {
    resultSummary.innerHTML = "";
    resultSummary.classList.add("hidden");
    resultText.innerHTML = `<div class="result-copy"><p>${escapeHtml(answer)}</p></div>`;
  }
  resultCard.classList.remove("hidden");
}

function hideResult() {
  resultSummary.innerHTML = "";
  resultSummary.classList.add("hidden");
  resultText.innerHTML = "";
  resultCard.classList.add("hidden");
}

queryInput.addEventListener("input", syncQueryMeta);

extractBtn.addEventListener("click", async () => {
  const file = imageInput.files[0];
  if (!file) {
    setStatus("Choose a prescription image before running OCR.", "error");
    return;
  }

  setBusy(extractBtn, "Extract medicine names", "Extracting...", true);
  setStatus("Extracting text from the selected prescription image...", "loading");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch(`${API_BASE}/ocr/image`, {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (!response.ok) {
      setStatus(data?.error || "OCR request failed.", "error");
      return;
    }

    queryInput.value = data.text || "";
    syncQueryMeta();
    setStatus(queryInput.value ? "OCR complete. Review the extracted medicines and validate when ready." : "OCR completed, but no readable medicine text was detected.", "success");
  } catch (error) {
    setStatus("Cannot connect to the OCR backend right now.", "error");
  } finally {
    setBusy(extractBtn, "Extract medicine names", "Extracting...", false);
  }
});

validateBtn.addEventListener("click", async () => {
  const patient_id = patientIdInput.value.trim();
  const doctor_id = doctorIdInput.value.trim();
  const query = queryInput.value.trim();

  if (!patient_id || !doctor_id || !query) {
    setStatus("Fill in Patient ID, Doctor ID, and the medicines list before validation.", "error");
    return;
  }

  setBusy(validateBtn, "Validate prescription", "Validating...", true);
  setStatus("Running clinical validation for this prescription...", "loading");
  hideResult();

  try {
    const response = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ patient_id, doctor_id, query })
    });

    const data = await response.json();

    if (!response.ok) {
      showResult(JSON.stringify(data, null, 2), false);
      setStatus(data?.detail || data?.error || "Validation failed.", "error");
      return;
    }

    const answer = data.answer || "No response received.";
    showResult(answer, true);
    setStatus("Validation complete. Review the structured summary below.", "success");
  } catch (error) {
    setStatus("Cannot connect to the validation backend right now.", "error");
  } finally {
    setBusy(validateBtn, "Validate prescription", "Validating...", false);
  }
});

syncQueryMeta();
setStatus("Waiting for input.", "idle");
