# 9. Machine Learning Methodology

## 9.1 Problem Formulation

The MedX project is formulated as an applied machine learning problem for intelligent prescription understanding and safety prediction. The objective is to analyze medicine names extracted from manual text, prescription images, or voice input and determine whether the prescribed drug combination is clinically safe for a specific patient.

From an ML perspective, the system solves a multi-stage prediction problem:

1. Medicine name extraction from noisy multimodal input
2. Normalization of extracted tokens into valid drug names
3. Retrieval of relevant medical evidence from structured and vector knowledge sources
4. Safety classification of the prescription into a clinically meaningful category

### Input Features

The effective input to the ML pipeline includes:

- Raw prescription text entered manually
- OCR-extracted text from uploaded prescription images
- Speech-to-text transcription from recorded audio
- Patient context:
  - age
  - allergies
  - chronic conditions
  - current medications
  - prior clinical history

### Output Target

The output is a structured clinical decision:

- `SAFE`
- `CAUTION`
- `NOT SAFE`
- `INSUFFICIENT`

In addition to the predicted class, the system produces supporting explanations, drug-wise notes, and recommended doctor actions. Therefore, the project can be viewed as a hybrid ML-based clinical decision-support system combining classification, information retrieval, and language generation.

## 9.2 Models Used

The project uses multiple machine learning models working together in a pipeline.

### 1. OCR Model for Prescription Understanding

- `EasyOCR` is used to read prescription text from images.
- It performs learned visual text recognition rather than rule-based image parsing.
- This model is important because prescriptions often contain:
  - handwriting
  - non-uniform spacing
  - low contrast
  - noisy backgrounds

To improve OCR quality, the input image is preprocessed using grayscale conversion, contrast enhancement, adaptive thresholding, and resized variants. Region-based extraction is also used so the OCR model focuses on likely text areas.

### 2. Speech-to-Text Model

- `faster-whisper` is used for voice prescription transcription.
- This is a deep learning speech recognition model that converts spoken medicine names into text.
- A medical prompt is supplied during inference so the model gives higher priority to likely drug names.

This module allows the system to support multimodal medical input and improves usability for faster prescription entry.

### 3. Fuzzy Matching and Token Normalization

After OCR or speech transcription, the extracted tokens may still contain spelling noise. To improve recognition accuracy:

- `RapidFuzz`
- Levenshtein similarity
- synonym mapping
- typo normalization

are used to map noisy tokens to valid drug names.

Although this stage is not a standalone deep learning model, it functions as a learned similarity-based post-processing layer that improves the quality of downstream prediction.

### 4. Sentence Embedding Model for Retrieval

The project uses `SentenceTransformer` embeddings through ChromaDB with the `all-MiniLM-L6-v2` embedding model.

This model converts drug-related text into dense vector representations so semantically relevant information can be retrieved even when exact wording differs. The embedding model is used for:

- drug evidence retrieval
- interaction evidence retrieval
- semantic similarity search across local medical knowledge collections

### 5. Large Language Model for Clinical Reasoning

The final prediction stage uses `Gemini` as a language model for medical reasoning.

The LLM receives:

- normalized medicine names
- patient context
- structured drug-interaction evidence
- vector-retrieved supporting documents

Based on these signals, the model performs the final safety classification and explanation generation. In ML terms, this stage acts as a context-aware decision model for classification and reasoning.

## 9.3 Model Selection

The MedX system does not rely on a single conventional model such as Logistic Regression, SVM, or Random Forest. Instead, a hybrid machine learning design was selected because prescription validation requires multimodal understanding, semantic retrieval, and clinical reasoning.

The selected architecture includes:

1. OCR model for image-based text recognition
2. Speech recognition model for audio-based input
3. Embedding model for semantic retrieval
4. Fuzzy similarity layer for token correction
5. LLM for final safety classification

This model combination was chosen because:

- prescriptions may be provided in text, image, or speech form
- exact keyword matching is insufficient for noisy medical data
- semantic retrieval improves coverage over purely structured lookup
- reasoning with patient context requires more than simple classification

Thus, the final design is a hybrid ML pipeline that combines perception models, retrieval models, and reasoning models.

## 9.4 Training and Inference Pipeline

Although the project mainly uses pre-trained models, it still follows a machine learning pipeline structure during inference.

### 1. Data Ingestion

- Input is received as text, image, or audio.

### 2. Feature Extraction

- OCR converts image pixels into text tokens.
- Whisper converts speech signals into text tokens.
- Patient attributes are collected as structured contextual features.

### 3. Feature Cleaning and Normalization

- spelling errors are corrected
- fragmented OCR tokens are merged
- synonyms are normalized
- invalid or noisy tokens are filtered

### 4. Representation Learning and Retrieval

- medicine queries are embedded into vector space
- semantically related drug and interaction evidence is retrieved from ChromaDB

### 5. Decision Prediction

- retrieved evidence and patient context are sent to the LLM
- the LLM predicts the final class:
  - SAFE
  - CAUTION
  - NOT SAFE
  - INSUFFICIENT

### 6. Post-Inference Memory

- validated cases are stored in MongoDB
- selected interaction knowledge is appended to the local knowledge base
- previous solved cases can be reused for similar future queries

This creates a practical ML-enabled inference loop with memory and incremental knowledge growth.

# 10. Model Evaluation

Because MedX is a multi-module ML system, evaluation is performed separately for each learning component. This is more appropriate than using a single metric for the whole pipeline.

## 10.1 OCR Evaluation

The OCR module is evaluated by comparing predicted medicine names against labeled ground truth.

Metrics used:

- Exact Match Accuracy
- Precision
- Recall
- F1-Score

### Interpretation

- High precision means the OCR module returns mostly correct medicine names.
- High recall means the module successfully captures most expected medicine names.
- High F1-score indicates balanced extraction performance.

## 10.2 Voice Model Evaluation

The speech recognition module is also evaluated using token-level comparison between expected and predicted medicine names.

Metrics used:

- Exact Match Accuracy
- Precision
- Recall
- F1-Score
- Status Accuracy

Status accuracy checks whether the final prescription safety result generated from the transcribed medicines matches the expected safety class.

## 10.3 Manual Query and Validation Evaluation

For direct manual medicine entry, the system evaluates the final safety prediction quality using:

- Status Accuracy
- Keyword Pass Rate
- Forbidden Keyword Clear Rate

These metrics verify whether the generated clinical response:

- predicts the correct safety category
- contains the required clinical concepts
- avoids clearly incorrect or irrelevant medical statements

## 10.4 History Retrieval Evaluation

The memory module is evaluated using:

- Success Rate
- Average Records Returned

This checks whether patient and doctor history retrieval returns the correct past records needed for context-aware validation.

## 10.5 Evaluation Procedure

The evaluation pipeline is implemented in `module2_rag/evaluation/evaluate_modules.py`. It measures module-wise performance for:

- OCR
- voice transcription
- manual query handling
- validation output
- history retrieval

This approach is suitable for a real-world ML system where multiple models contribute to the final output.

# 11. ML-Based Scoring and Decision System

The MedX project uses an ML-based safety decision mechanism rather than a numerical regression score. At runtime, the system combines predictions from multiple learned components and converts them into a clinically usable decision.

## 11.1 Intermediate ML Signals

The final prediction is influenced by several learned or similarity-based signals:

1. OCR confidence from text extraction
2. Speech transcription quality
3. Fuzzy similarity between extracted tokens and known medicine names
4. Semantic relevance of retrieved vector documents
5. LLM reasoning over drugs and patient context

These signals collectively function as an implicit confidence framework for the final prediction.

## 11.2 Final Classification

The LLM produces one of four prediction classes:

- `SAFE`
- `CAUTION`
- `NOT SAFE`
- `INSUFFICIENT`

### Interpretation

- `SAFE` indicates no major harmful interaction was inferred from available evidence.
- `CAUTION` indicates possible risk, conditional safety, or monitoring requirement.
- `NOT SAFE` indicates a clinically significant interaction or contraindication.
- `INSUFFICIENT` indicates that available evidence is not enough for a reliable decision.

## 11.3 Why This Functions as an ML Scoring System

Although the output is categorical, the system behaves like an ML scoring pipeline because:

- it extracts patterns from noisy input using trained models
- it measures similarity in learned embedding space
- it ranks retrieved evidence by semantic closeness
- it performs context-aware classification using a large language model

Therefore, the project can be described as an ML-driven clinical classification and recommendation system.

# 12. Other ML-Centric Modules

## 12.1 Multimodal Prescription Understanding

One major contribution of the project is multimodal learning support. A prescription can enter the system as:

- typed text
- image
- audio

Each modality is handled using an appropriate ML model, and all are converted into a unified normalized medicine representation.

## 12.2 Retrieval-Augmented Clinical Reasoning

The project uses a retrieval-augmented generation strategy, which is highly relevant in applied machine learning.

### Retrieval sources

- structured CSV interaction data
- ChromaDB vector collections
- previously validated interaction knowledge

### Benefit

This improves factual grounding and reduces dependence on free-form language generation alone.

## 12.3 Memory-Augmented Learning Behavior

MongoDB is used to store solved cases, patient profiles, doctor profiles, and visit history. While MongoDB is a database component, in ML terms it enables:

- case-based reuse
- historical context injection
- adaptive future responses based on past validated outcomes

This improves practical prediction consistency over time.

## 12.4 Alert Prioritization

When the predicted class is `NOT SAFE`, the system triggers an alert workflow. This means the ML decision is not only informative but actionable. The predicted label drives:

- PDF report generation
- email alerts
- SMS notifications

This makes the model output operational in a real clinical setting.

# 13. Continuous Learning Strategy

In a strict academic sense, the project does not implement reinforcement learning with reward optimization. However, it does support continuous learning and knowledge refinement.

## 13.1 Knowledge Expansion

When a clinically useful interaction result is produced:

- the interaction pair can be appended to the local CSV knowledge base
- the validated interaction can be stored in a dedicated ChromaDB collection

This allows the retrieval layer to improve over time.

## 13.2 Case Memory Reuse

Previously solved cases are stored and reused when a similar prescription appears again. This acts as a lightweight experience-based learning mechanism.

## 13.3 Human-in-the-Loop Improvement

The system can be improved by:

- adding more labeled OCR samples
- adding more labeled voice samples
- refining drug synonym mappings
- expanding interaction datasets
- reviewing generated medical outputs

Thus, the project follows a human-in-the-loop continuous improvement methodology, which is highly relevant in practical healthcare ML systems.

# 14. System Architecture

The overall architecture supports the ML workflow end to end.

## Frontend Layer

- React-based user interface for prescription entry, registration, dashboard, and history

## Backend Layer

- FastAPI backend for serving ML inference APIs

## Machine Learning Layer

- EasyOCR for image understanding
- faster-whisper for audio transcription
- RapidFuzz and Levenshtein-based normalization
- SentenceTransformer embeddings for semantic retrieval
- Gemini LLM for final safety classification and explanation

## Data Layer

- ChromaDB for vector search and retrieval
- MongoDB for patient memory, doctor memory, visits, and solved cases

## Alert and Reporting Layer

- PDF report generation
- email alerts
- SMS notifications

This architecture supports a complete machine learning lifecycle from input acquisition to prediction, explanation, memory, and action.

# 15. Conclusion

The MedX project is a machine learning-driven prescription safety system that combines multimodal input processing, semantic retrieval, and clinical reasoning. Instead of relying on a single predictive model, it uses a coordinated ML pipeline that includes OCR, speech recognition, fuzzy normalization, vector embeddings, and large language model inference.

This design makes the system suitable for real clinical environments where inputs are noisy, patient context matters, and explainability is important. The project demonstrates how modern machine learning can be applied to prescription validation by integrating perception models, retrieval models, and reasoning models into one practical healthcare solution.
