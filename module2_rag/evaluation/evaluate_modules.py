import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    for ch in [",", ";", "/", "|", "+", "\n", "\r", "\t"]:
        text = text.replace(ch, " ")
    return " ".join(text.split())


def tokenize_medicines(value: Any) -> List[str]:
    text = normalize_text(value)
    if not text:
        return []
    return sorted(set(token for token in text.split(" ") if token))


def percent(part: float, total: float) -> float:
    if not total:
        return 0.0
    return round((part / total) * 100, 2)


def extract_status(answer: str) -> str:
    text = str(answer or "").upper()
    if "NOT SAFE" in text:
      return "NOT SAFE"
    if "CAUTION" in text:
      return "CAUTION"
    if "SAFE" in text:
      return "SAFE"
    if "INSUFFICIENT" in text:
      return "INSUFFICIENT"
    return "UNKNOWN"


def contains_keywords(text: str, keywords: Sequence[str]) -> bool:
    haystack = normalize_text(text)
    return all(normalize_text(keyword) in haystack for keyword in keywords)


def compare_token_sets(predicted: Sequence[str], expected: Sequence[str]) -> Dict[str, float]:
    predicted_set = set(predicted)
    expected_set = set(expected)
    true_positive = len(predicted_set & expected_set)
    precision = percent(true_positive, len(predicted_set)) if predicted_set else 0.0
    recall = percent(true_positive, len(expected_set)) if expected_set else 0.0

    if precision == 0.0 and recall == 0.0:
        f1 = 0.0
    else:
        precision_ratio = precision / 100.0
        recall_ratio = recall / 100.0
        f1 = round((2 * precision_ratio * recall_ratio) / (precision_ratio + recall_ratio) * 100, 2)

    exact_match = predicted_set == expected_set
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": 1.0 if exact_match else 0.0,
    }


def read_cases(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    modules = ["ocr", "voice", "manual", "validation", "history"]
    return {module: list(data.get(module, [])) for module in modules}


def post_json(session: requests.Session, base_url: str, route: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = session.post(f"{base_url}{route}", json=payload, timeout=180)
    response.raise_for_status()
    return response.json()


def post_file(session: requests.Session, base_url: str, route: str, file_path: Path, field_name: str = "file") -> Dict[str, Any]:
    with file_path.open("rb") as handle:
        response = session.post(
            f"{base_url}{route}",
            files={field_name: (file_path.name, handle)},
            timeout=180,
        )
    response.raise_for_status()
    return response.json()


def get_json(session: requests.Session, base_url: str, route: str) -> Dict[str, Any]:
    response = session.get(f"{base_url}{route}", timeout=180)
    response.raise_for_status()
    return response.json()


def evaluate_ocr(session: requests.Session, base_url: str, root_dir: Path, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    results = []
    for case in cases:
        file_path = (root_dir / case["file"]).resolve()
        response = post_file(session, base_url, "/ocr/image", file_path)
        predicted_tokens = tokenize_medicines(response.get("normalized_text") or response.get("text"))
        expected_tokens = tokenize_medicines(case.get("expected_text", ""))
        token_metrics = compare_token_sets(predicted_tokens, expected_tokens)
        results.append(
            {
                "id": case.get("id", file_path.name),
                "predicted_text": response.get("text", ""),
                "expected_text": case.get("expected_text", ""),
                "avg_confidence": response.get("avg_confidence", 0.0),
                **token_metrics,
            }
        )

    return summarize_token_module("ocr", results)


def evaluate_voice(session: requests.Session, base_url: str, root_dir: Path, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    results = []
    for case in cases:
        file_path = (root_dir / case["file"]).resolve()
        response = post_file(session, base_url, "/stt/audio", file_path)
        predicted_tokens = tokenize_medicines(response.get("text", ""))
        expected_tokens = tokenize_medicines(case.get("expected_text", ""))
        token_metrics = compare_token_sets(predicted_tokens, expected_tokens)
        predicted_status = extract_status(response.get("validation", ""))
        expected_status = case.get("expected_status", "").upper().strip() or None
        status_match = 1.0 if expected_status and predicted_status == expected_status else 0.0
        results.append(
            {
                "id": case.get("id", file_path.name),
                "predicted_text": response.get("text", ""),
                "expected_text": case.get("expected_text", ""),
                "predicted_status": predicted_status,
                "expected_status": expected_status,
                "status_match": status_match,
                **token_metrics,
            }
        )

    return summarize_token_module("voice", results, include_status=True)


def evaluate_manual(session: requests.Session, base_url: str, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    results = []
    for case in cases:
        payload = {
            "patient_id": case["patient_id"],
            "doctor_id": case["doctor_id"],
            "query": case["query"],
        }
        response = post_json(session, base_url, "/ask", payload)
        predicted_status = extract_status(response.get("answer", ""))
        expected_status = case.get("expected_status", "").upper().strip()
        required_keywords = case.get("required_keywords", [])
        keyword_match = 1.0 if contains_keywords(response.get("answer", ""), required_keywords) else 0.0
        results.append(
            {
                "id": case.get("id", case["query"]),
                "predicted_status": predicted_status,
                "expected_status": expected_status,
                "status_match": 1.0 if predicted_status == expected_status else 0.0,
                "keyword_match": keyword_match,
                "used_previous_case": bool(response.get("used_previous_case")),
            }
        )

    return summarize_status_module("manual", results)


def evaluate_validation(session: requests.Session, base_url: str, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    results = []
    for case in cases:
        payload = {
            "patient_id": case["patient_id"],
            "doctor_id": case["doctor_id"],
            "query": case["query"],
        }
        response = post_json(session, base_url, "/ask", payload)
        answer = response.get("answer", "")
        predicted_status = extract_status(answer)
        expected_status = case.get("expected_status", "").upper().strip()
        required_keywords = case.get("required_keywords", [])
        forbidden_keywords = case.get("forbidden_keywords", [])
        required_match = 1.0 if contains_keywords(answer, required_keywords) else 0.0
        forbidden_clear = 1.0 if not any(normalize_text(keyword) in normalize_text(answer) for keyword in forbidden_keywords) else 0.0
        results.append(
            {
                "id": case.get("id", case["query"]),
                "predicted_status": predicted_status,
                "expected_status": expected_status,
                "status_match": 1.0 if predicted_status == expected_status else 0.0,
                "required_match": required_match,
                "forbidden_clear": forbidden_clear,
            }
        )

    return summarize_status_module("validation", results, include_forbidden=True)


def find_history_match(records: Sequence[Dict[str, Any]], expected: Dict[str, Any]) -> bool:
    expected_text = normalize_text(expected.get("contains_text", ""))
    expected_counterpart = normalize_text(expected.get("counterpart_id", ""))

    for record in records:
        joined = normalize_text(
            " ".join(
                str(record.get(key, ""))
                for key in ["complaint", "diagnosis", "prescription_text", "notes", "patient_id", "doctor_id"]
            )
        )
        counterpart_values = [
            record.get("doctor_id", ""),
            record.get("patient_id", ""),
            (record.get("doctor") or {}).get("doctor_id", ""),
            (record.get("patient") or {}).get("patient_id", ""),
        ]
        counterpart_text = normalize_text(" ".join(str(value) for value in counterpart_values))

        text_ok = not expected_text or expected_text in joined
        counterpart_ok = not expected_counterpart or expected_counterpart in counterpart_text
        if text_ok and counterpart_ok:
            return True

    return False


def evaluate_history(session: requests.Session, base_url: str, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    results = []
    for case in cases:
        route = f"/patients/{case['patient_id']}/history?limit={case.get('limit', 20)}" if case.get("patient_id") else f"/doctors/{case['doctor_id']}/history?limit={case.get('limit', 20)}"
        response = get_json(session, base_url, route)
        records = response.get("history", [])
        matched = find_history_match(records, case)
        results.append(
            {
                "id": case.get("id", case.get("patient_id") or case.get("doctor_id")),
                "matched": 1.0 if matched else 0.0,
                "returned_records": len(records),
            }
        )

    total = len(results)
    matched = sum(item["matched"] for item in results)
    avg_returned = round(sum(item["returned_records"] for item in results) / total, 2) if total else 0.0
    return {
        "module": "history",
        "cases": total,
        "success_rate": percent(matched, total),
        "average_records_returned": avg_returned,
        "details": results,
    }


def summarize_token_module(module: str, results: List[Dict[str, Any]], include_status: bool = False) -> Dict[str, Any]:
    total = len(results)
    exact_matches = sum(item["exact_match"] for item in results)
    summary = {
        "module": module,
        "cases": total,
        "exact_match_accuracy": percent(exact_matches, total),
        "average_precision": round(sum(item["precision"] for item in results) / total, 2) if total else 0.0,
        "average_recall": round(sum(item["recall"] for item in results) / total, 2) if total else 0.0,
        "average_f1": round(sum(item["f1"] for item in results) / total, 2) if total else 0.0,
        "details": results,
    }
    if include_status:
        expected_status_cases = [item for item in results if item.get("expected_status")]
        status_matches = sum(item["status_match"] for item in expected_status_cases)
        summary["status_accuracy"] = percent(status_matches, len(expected_status_cases))
    return summary


def summarize_status_module(module: str, results: List[Dict[str, Any]], include_forbidden: bool = False) -> Dict[str, Any]:
    total = len(results)
    status_matches = sum(item["status_match"] for item in results)
    summary = {
        "module": module,
        "cases": total,
        "status_accuracy": percent(status_matches, total),
        "keyword_pass_rate": percent(sum(item.get("keyword_match", item.get("required_match", 0.0)) for item in results), total),
        "details": results,
    }
    if include_forbidden:
        summary["forbidden_keyword_clear_rate"] = percent(sum(item["forbidden_clear"] for item in results), total)
    return summary


def print_summary(report: Dict[str, Any]) -> None:
    print("\nModule Accuracy Summary")
    print("=======================")
    for module_name, module_report in report["modules"].items():
        if module_report["cases"] == 0:
            print(f"{module_name}: no cases")
            continue

        print(f"{module_name}:")
        for key, value in module_report.items():
            if key in {"module", "cases", "details"}:
                continue
            print(f"  {key}: {value}")


def write_output(report: Dict[str, Any], output_path: Optional[Path]) -> None:
    if not output_path:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MedX OCR, voice, manual, validation, and history modules.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Backend base URL")
    parser.add_argument("--cases", required=True, help="Path to labeled evaluation cases JSON")
    parser.add_argument("--root-dir", default=".", help="Base directory used to resolve relative file paths in cases")
    parser.add_argument("--output", help="Optional path to save JSON report")
    args = parser.parse_args()

    cases_path = Path(args.cases).resolve()
    root_dir = Path(args.root_dir).resolve()
    output_path = Path(args.output).resolve() if args.output else None

    case_groups = read_cases(cases_path)
    session = requests.Session()
    base_url = args.base_url.rstrip("/")

    report = {
        "base_url": base_url,
        "case_file": str(cases_path),
        "modules": {
            "ocr": evaluate_ocr(session, base_url, root_dir, case_groups["ocr"]),
            "voice": evaluate_voice(session, base_url, root_dir, case_groups["voice"]),
            "manual": evaluate_manual(session, base_url, case_groups["manual"]),
            "validation": evaluate_validation(session, base_url, case_groups["validation"]),
            "history": evaluate_history(session, base_url, case_groups["history"]),
        },
    }

    print_summary(report)
    write_output(report, output_path)


if __name__ == "__main__":
    main()
