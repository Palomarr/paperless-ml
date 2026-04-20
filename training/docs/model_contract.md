# Model Contract

## HTR Model

Input:
- shape: [1, 1, 28, 28]
- dtype: float32

Output:
- name: logits
- shape: [1, 10]
- description: classification scores

---

## Retrieval Model

Input:
- shape: [1, 1, 28, 28]
- dtype: float32

Output:
- name: embedding
- shape: [1, 128]
- description: semantic embedding vector