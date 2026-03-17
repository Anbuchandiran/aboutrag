const patientIdInput = document.getElementById("patientId");
const doctorIdInput = document.getElementById("doctorId");
const imageInput = document.getElementById("imageInput");
const queryInput = document.getElementById("query");
const extractBtn = document.getElementById("extractBtn");
const validateBtn = document.getElementById("validateBtn");
const statusDiv = document.getElementById("status");
const resultCard = document.getElementById("resultCard");
const resultBadge = document.getElementById("resultBadge");
const resultText = document.getElementById("resultText");

function setBadge(answer) {
  const text = answer.toUpperCase();
  resultBadge.className = "badge";

  if (text.includes("NOT SAFE")) {
    resultBadge.textContent = "NOT SAFE";
    resultBadge.classList.add("notsafe");
  } else if (text.includes("CAUTION")) {
    resultBadge.textContent = "CAUTION";
    resultBadge.classList.add("caution");
  } else if (text.includes("SAFE")) {
    resultBadge.textContent = "SAFE";
    resultBadge.classList.add("safe");
  } else {
    resultBadge.textContent = "INSUFFICIENT";
    resultBadge.classList.add("insufficient");
  }
}

extractBtn.addEventListener("click", async () => {
  const file = imageInput.files[0];
  if (!file) {
    statusDiv.textContent = "Please choose an image first.";
    return;
  }

  statusDiv.textContent = "Extracting text from image...";
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("http://127.0.0.1:8000/ocr/image", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (!response.ok) {
      statusDiv.textContent = "OCR request failed.";
      return;
    }

    queryInput.value = data.text || "";
    statusDiv.textContent = "OCR extraction complete.";
  } catch (error) {
    statusDiv.textContent = "Cannot connect to OCR backend.";
  }
});

validateBtn.addEventListener("click", async () => {
  const patient_id = patientIdInput.value.trim();
  const doctor_id = doctorIdInput.value.trim();
  const query = queryInput.value.trim();

  if (!patient_id || !doctor_id || !query) {
    statusDiv.textContent = "Please fill Patient ID, Doctor ID, and medicines.";
    return;
  }

  statusDiv.textContent = "Validating prescription...";
  resultCard.classList.add("hidden");
  resultText.textContent = "";

  try {
    const response = await fetch("http://127.0.0.1:8000/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ patient_id, doctor_id, query })
    });

    const data = await response.json();

    if (!response.ok) {
      statusDiv.textContent = "Validation failed.";
      resultCard.classList.remove("hidden");
      resultText.textContent = JSON.stringify(data, null, 2);
      return;
    }

    const answer = data.answer || "No response received.";
    setBadge(answer);
    resultText.textContent = answer;
    resultCard.classList.remove("hidden");
    statusDiv.textContent = "Validation complete.";
  } catch (error) {
    statusDiv.textContent = "Cannot connect to validation backend.";
  }
});