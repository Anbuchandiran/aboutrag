# module2_rag/src/ocr_service.py

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Any

import cv2
import easyocr
import numpy as np


# Load once and reuse
_reader = easyocr.Reader(["en"], gpu=False)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert uploaded image bytes into an OCR-friendly OpenCV image.
    """
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Unable to read image. Please upload a valid image file.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Slight denoise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Improve contrast
    gray = cv2.equalizeHist(gray)

    # Threshold for handwritten/printed prescription visibility
    processed = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    return processed


def extract_text_from_image_bytes(image_bytes: bytes) -> Dict[str, Any]:
    """
    Run OCR and return raw text + line-wise output + confidence.
    """
    img = preprocess_image(image_bytes)

    results = _reader.readtext(img, detail=1, paragraph=False)

    lines: List[str] = []
    confidences: List[float] = []

    for item in results:
        # easyocr result => [bbox, text, confidence]
        text = item[1].strip()
        conf = float(item[2])

        if text:
            lines.append(text)
            confidences.append(conf)

    raw_text = "\n".join(lines)
    avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

    return {
        "raw_text": raw_text,
        "lines": lines,
        "avg_confidence": avg_conf,
    }


def normalize_prescription_text(text: str) -> str:
    """
    Clean OCR noise a bit.
    """
    text = text.replace("\r", "\n")
    text = re.sub(r"[|]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def extract_possible_drugs(text: str) -> List[str]:
    """
    Very simple drug extraction heuristic.
    Later you can replace this with DB/chroma matching.
    """
    if not text:
        return []

    # split by commas, newlines, slash, plus
    chunks = re.split(r"[,/\n+]", text)

    cleaned: List[str] = []
    for chunk in chunks:
        item = chunk.strip()

        # remove dose patterns like 500mg, 1-0-1, OD, BD etc.
        item = re.sub(r"\b\d+(\.\d+)?\s?(mg|ml|mcg|g|tab|tabs|cap|caps)\b", "", item, flags=re.I)
        item = re.sub(r"\b(od|bd|tid|qid|hs|sos)\b", "", item, flags=re.I)
        item = re.sub(r"\b\d-\d-\d\b", "", item)

        # keep alphabetic names only
        item = re.sub(r"[^a-zA-Z\s-]", "", item).strip()

        # filter small junk tokens
        if len(item) >= 3:
            cleaned.append(item.lower())

    # unique preserve order
    seen = set()
    final = []
    for d in cleaned:
        if d not in seen:
            seen.add(d)
            final.append(d)

    return final