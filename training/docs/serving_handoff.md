# Serving Handoff

## Exported Models

Location:
outputs/exported/

- htr_model.onnx
- retrieval_model.onnx

---

## Triton Repository

Location:
outputs/triton_repo/

Models:
- htr
- retrieval

Each model contains:
- config.pbtxt
- version folder (1/model.onnx)

---

## Notes

- Models are ready for Triton deployment
- Dynamic batching enabled
- Input/output defined in config.pbtxt