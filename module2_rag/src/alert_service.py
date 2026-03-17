# module2_rag/src/05_sms.py
import os
import re
import json
import smtplib
from datetime import datetime
from email.message import EmailMessage
from typing import Dict, Any, Optional
from dotenv import load_dotenv

import requests
from pymongo import MongoClient

load_dotenv("module2_rag/.env")

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors

# ---------------- PATHS ----------------
MODULE2_RAG_BASE = os.getenv("MODULE2_RAG_BASE", "module2_rag")

TEXT_PATH = os.path.join(MODULE2_RAG_BASE, "data", "last_validation.txt")
CONTEXT_JSON_PATH = os.path.join(MODULE2_RAG_BASE, "data", "last_context.json")
PDF_PATH = os.path.join(MODULE2_RAG_BASE, "data", "MedX_Alert_Report.pdf")

# ---------------- EMAIL CONFIG ----------------
SENDER_EMAIL = os.getenv("ALERT_SENDER_EMAIL", "").strip()
APP_PASSWORD = os.getenv("ALERT_APP_PASSWORD", "").strip()
FALLBACK_RECEIVER_EMAIL = os.getenv("ALERT_RECEIVER_EMAIL", "").strip()

ALERT_STATUSES = {"NOT SAFE"}

# ---------------- SMS CONFIG (FAST2SMS) ----------------
FAST2SMS_API_KEY = os.getenv("FAST2SMS_API_KEY", "").strip()
SMS_MAX_CHARS = int(os.getenv("ALERT_SMS_MAX_CHARS", "300"))

# ---------------- MONGO (SYNC) ----------------
# Use whichever exists in your .env
MONGO_URI = (
    os.getenv("MONGO_URI", "").strip()
    or os.getenv("MONGODB_URI", "").strip()
    or os.getenv("MONGO_URL", "").strip()
)
MONGO_DB_NAME = (
    os.getenv("MONGO_DB_NAME", "").strip()
    or os.getenv("DB_NAME", "").strip()
    or os.getenv("MONGO_DB", "").strip()
    or "medx"
)

def _get_db():
    if not MONGO_URI:
        raise RuntimeError(
            "Mongo URI missing. Add MONGO_URI (or MONGODB_URI) in module2_rag/.env"
        )
    client = MongoClient(MONGO_URI)
    return client[MONGO_DB_NAME]

# ---------------- HELPERS ----------------
def read_text(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}

def extract_status(text: str) -> str:
    m = re.search(
        r"^\s*(Overall_Status|Status)\s*:\s*(.+)\s*$",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    return m.group(2).strip().upper() if m else "UNKNOWN"

def safe_str(x: Any, default: str = "N/A") -> str:
    if x is None:
        return default
    if isinstance(x, (list, tuple)):
        return ", ".join(map(str, x)) if x else default
    s = str(x).strip()
    return s if s else default

def build_patient_info_from_context(ctx: Dict[str, Any]) -> Dict[str, Any]:
    patient = ctx.get("patient", {}) or {}
    doctor = ctx.get("doctor", {}) or {}
    return {
        "patient_id": ctx.get("patient_id"),
        "name": patient.get("name"),
        "age": patient.get("age"),
        "gender": patient.get("gender"),
        "phone": patient.get("phone"),
        "chronic_conditions": patient.get("chronic_conditions"),
        "allergies": patient.get("allergies"),
        "doctor_id": ctx.get("doctor_id"),
        "doctor_name": doctor.get("name"),
        "department": doctor.get("department"),
        "generated_at": ctx.get("generated_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

def strip_patient_details_block(text: str) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    out = []
    skipping = False

    for line in lines:
        s = line.strip()

        if s.lower() == "patient details":
            skipping = True
            continue

        if skipping and (set(s) <= set("-") and len(s) >= 5):
            skipping = False
            continue

        if skipping and s.lower() == "validation result":
            skipping = False
            out.append(line)
            continue

        if not skipping:
            out.append(line)

    return "\n".join(out).strip()

# ---------------- DOCTOR CONTACT (SYNC MONGO) ----------------
def get_doctor_contact_by_id(doctor_id: str) -> Dict[str, str]:
    doctor_id = (doctor_id or "").strip()
    if not doctor_id:
        return {"email": "", "phone": "", "name": ""}

    db = _get_db()
    doctors_col = db["doctors"]

    doc = doctors_col.find_one({"_id": doctor_id}) or doctors_col.find_one({"doctor_id": doctor_id})
    if not doc:
        print(f"❌ No doctor found in Mongo for doctor_id={doctor_id}")
        return {"email": "", "phone": "", "name": ""}

    return {
        "email": str(doc.get("email", "")).strip(),
        "phone": str(doc.get("phone", "")).strip(),
        "name": str(doc.get("name", "")).strip(),
    }

# ---------------- SMTP SEND ----------------
def smtp_send(msg: EmailMessage) -> None:
    if not (SENDER_EMAIL and APP_PASSWORD):
        raise RuntimeError("Missing ALERT_SENDER_EMAIL / ALERT_APP_PASSWORD in .env")

    with smtplib.SMTP("smtp.gmail.com", 587, timeout=25) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login(SENDER_EMAIL, APP_PASSWORD)
        smtp.send_message(msg)

# ---------------- PDF ----------------
def create_pdf_report(validation_text: str, patient_info: Dict[str, Any], pdf_path: str) -> None:
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("MedX Clinical Validation Report", styles["Heading1"]))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.black))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("Patient Details", styles["Heading2"]))
    elements.append(Spacer(1, 0.12 * inch))

    details_order = [
        ("Patient ID", patient_info.get("patient_id")),
        ("Patient Name", patient_info.get("name")),
        ("Age", patient_info.get("age")),
        ("Gender", patient_info.get("gender")),
        ("Phone", patient_info.get("phone")),
        ("Chronic Conditions", patient_info.get("chronic_conditions")),
        ("Allergies", patient_info.get("allergies")),
        ("Doctor ID", patient_info.get("doctor_id")),
        ("Doctor Name", patient_info.get("doctor_name")),
        ("Department", patient_info.get("department")),
        ("Generated At", patient_info.get("generated_at")),
    ]

    for k, v in details_order:
        elements.append(Paragraph(f"<b>{k}:</b> {safe_str(v)}", styles["Normal"]))
        elements.append(Spacer(1, 0.06 * inch))

    elements.append(Spacer(1, 0.2 * inch))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("Validation Result", styles["Heading2"]))
    elements.append(Spacer(1, 0.12 * inch))

    validation_text = strip_patient_details_block(validation_text)

    for line in (validation_text or "").splitlines():
        line = line.strip()
        if not line:
            elements.append(Spacer(1, 0.06 * inch))
            continue
        safe_line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        elements.append(Paragraph(safe_line, styles["Normal"]))
        elements.append(Spacer(1, 0.06 * inch))

    doc.build(elements)

# ---------------- EMAIL ----------------
def send_email_with_pdf(subject: str, body: str, pdf_path: str, to_email: str) -> None:
    to_email = (to_email or "").strip()
    if not to_email:
        raise RuntimeError("Receiver email missing (doctor email empty and no fallback).")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = to_email
    msg.set_content(body + "\n\n(Full PDF report attached.)")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    msg.add_attachment(
        pdf_bytes,
        maintype="application",
        subtype="pdf",
        filename=os.path.basename(pdf_path),
    )

    smtp_send(msg)

# ---------------- SMS (FAST2SMS) ----------------
def send_sms_fast2sms(phone: str, message: str) -> bool:
    if not FAST2SMS_API_KEY:
        print("❌ FAST2SMS_API_KEY missing in .env")
        return False

    digits = re.sub(r"\D", "", str(phone))
    if len(digits) < 10:
        print(f"❌ Invalid doctor phone: {phone}")
        return False

    url = "https://www.fast2sms.com/dev/bulkV2"
    payload = {
        "message": (message or "")[:SMS_MAX_CHARS],
        "language": "english",
        "route": "q",
        "numbers": digits[-10:],
    }
    headers = {
        "authorization": FAST2SMS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    r = requests.post(url, json=payload, headers=headers, timeout=20)
    print("Fast2SMS status_code:", r.status_code)
    print("Fast2SMS raw response:", r.text)

    try:
        data = r.json()
        return bool(data.get("return")) is True
    except Exception:
        return False

# ---------------- MAIN LOGIC ----------------
def generate_and_alert(
    validation_text: str,
    patient_info: Dict[str, Any],
    ctx: Dict[str, Any],
    pdf_path: str = PDF_PATH,
    always_send: bool = False,
) -> Dict[str, Any]:
    status = extract_status(validation_text)

    create_pdf_report(validation_text, patient_info, pdf_path)

    emailed = False
    sms_sent = False
    doctor_email = ""
    doctor_phone = ""

    if always_send or (status in ALERT_STATUSES):
        doctor_id = (ctx.get("doctor_id") or patient_info.get("doctor_id") or "").strip()
        ctx_doctor = ctx.get("doctor", {}) or {}
        contact = get_doctor_contact_by_id(doctor_id)

        doctor_email = (
            str(ctx_doctor.get("email", "")).strip()
            or str(contact.get("email", "")).strip()
            or FALLBACK_RECEIVER_EMAIL
        )

        doctor_phone = (
            str(ctx_doctor.get("phone", "")).strip()
            or str(contact.get("phone", "")).strip()
        )

        print("DEBUG doctor_id:", doctor_id)
        print("DEBUG ctx doctor email:", ctx_doctor.get("email"))
        print("DEBUG mongo doctor email:", contact.get("email"))
        print("DEBUG final email used:", doctor_email)

        if not doctor_email:
            raise RuntimeError(f"No email found for doctor_id={doctor_id}")

        subject = f"MedX Clinical Alert: {status}"
        body = (
            f"Status: {status}\n"
            f"Patient ID: {safe_str(patient_info.get('patient_id'))}\n"
            f"Patient Name: {safe_str(patient_info.get('name'))}\n"
            f"Doctor ID: {safe_str(doctor_id)}\n"
            f"Doctor Name: {safe_str(ctx_doctor.get('name') or patient_info.get('doctor_name'))}\n\n"
            f"{validation_text}"
        )

        send_email_with_pdf(subject, body, pdf_path, to_email=doctor_email)
        emailed = True
        print("✅ Sending email to:", doctor_email)

        if doctor_phone:
            sms_text = (
                f"MedX ALERT: {status}. "
                f"PID:{safe_str(patient_info.get('patient_id'))} "
                f"Name:{safe_str(patient_info.get('name'))}. "
                f"Check email for PDF."
            )
            sms_sent = send_sms_fast2sms(doctor_phone, sms_text)

    return {
        "status": status,
        "pdf_path": pdf_path,
        "emailed": emailed,
        "sms_sent": sms_sent,
        "doctor_email": doctor_email,
        "doctor_phone": doctor_phone,
    }

def main():
    validation_text = read_text(TEXT_PATH)
    ctx = read_json(CONTEXT_JSON_PATH)

    patient_info = build_patient_info_from_context(ctx) if ctx else {}
    patient_info = patient_info or {}
    patient_info.setdefault("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    result = generate_and_alert(validation_text, patient_info, ctx, PDF_PATH, always_send=False)
    print("Detected status:", result["status"])
    print("PDF generated:", result["pdf_path"])
    print("Email sent:", result["emailed"])
    print("SMS sent:", result["sms_sent"])

if __name__ == "__main__":
    main()