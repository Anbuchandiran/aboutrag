# Module Evaluation

Use `evaluate_modules.py` to calculate percentage-based results for:

- OCR upload
- Voice transcription
- Manual query handling
- Validation output
- History retrieval

## What it measures

- `ocr`: exact-match accuracy, average precision, average recall, average F1
- `voice`: exact-match accuracy, average precision, average recall, average F1, status accuracy
- `manual`: status accuracy and keyword pass rate
- `validation`: status accuracy, required keyword pass rate, forbidden keyword clear rate
- `history`: success rate for finding expected records

## Run

```powershell
python module2_rag/evaluation/evaluate_modules.py `
  --base-url http://127.0.0.1:8000 `
  --cases module2_rag/evaluation/sample_cases.json `
  --root-dir . `
  --output module2_rag/evaluation/report.json
```

## Notes

- You need the backend running for the evaluator to work.
- Replace placeholder file paths in `sample_cases.json` with your real audio and image samples.
- Accuracy percentages are only meaningful when the cases file contains labeled ground truth.
