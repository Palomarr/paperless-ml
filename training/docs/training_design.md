# Training Design

## 1. Models

### HTR Model

We use a CRNN-based architecture for handwritten text recognition:

- Encoder: CNN (for feature extraction from image crops)
- Sequence modeling: BiLSTM
- Output: character sequence
- Loss: CTC Loss

Input:
- image crop from `crop_s3_url`

Output:
- htr_output (recognized text)
- htr_confidence

---

### Retrieval Model

We use a Sentence-BERT (bi-encoder) architecture:

- Encode query and document separately
- Compute similarity using cosine similarity

Input:
- query_text
- document_text

Output:
- embedding vectors
- similarity score


## 2. Loss Functions

### HTR

We use CTC Loss for sequence alignment between predicted characters and ground truth text.

### Retrieval

We use Multiple Negatives Ranking Loss:

- Positive: clicked document
- Negative: non-clicked documents



## 3. Training Data Construction

### HTR Training Data

Source:
- `htr_input_sample.json`
- user corrections

Process:
- Load image from `crop_s3_url`
- Use corrected text as label

Final format:
(image, text)

Example fields from JSON:

- crop_s3_url → used to load image
- region_id → used as sample identifier


### Retrieval Training Data

Source:
- `retrieval_input_sample.json`
- user interaction logs

Process:
- Positive: clicked document
- Negative: shown but not clicked

Final format:
(query, positive_doc, negative_doc)

Example fields from JSON:

- query_text → input query
- top_k → number of retrieved results

## 4. Initial Training (External Data)

### HTR

- Dataset: IAM Handwriting Dataset
- Training: supervised learning
- Objective: learn character recognition

Pipeline:
pretrain on IAM → fine-tune on user data

---

### Retrieval

- Dataset: SQuAD 2.0
- Training: contrastive learning
- Objective: semantic matching

Format:
(question, passage)




## 5. Retraining Pipeline

### HTR Retraining

Data:
- user-corrected text

Pipeline:
image + corrected_text → fine-tune model

Frequency:
- periodic (e.g., weekly)

Goal:
- reduce correction rate

---

### Retrieval Retraining

Data:
- user clicks

Pipeline:
(query, clicked_doc) → positive
(query, not_clicked_doc) → negative

Train:
- contrastive fine-tuning

Goal:
- improve CTR


## 6. Training Pipeline

The training pipeline follows this flow:

JSON input → DataLoader → Model → Loss → Optimization → Model Output

Steps:

1. Load JSON data from data pipeline
2. Convert to training samples
3. Pass through model
4. Compute loss
5. Backpropagation
6. Save trained model

Output:
- trained model file (e.g., model.pt)
The trained model is exported and used by the serving system for inference.


## 7. Evaluation

### HTR
Offline:
- CER (Character Error Rate)
- WER (Word Error Rate)

Online:
- Correction Rate


### Retrieval
Offline:
- Recall@K
- MRR (Mean Reciprocal Rank)
- NDCG
Online:
- CTR (Click-through rate)




