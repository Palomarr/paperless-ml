# Triton Setup Notes (for tomorrow)

## TrOCR Serving Strategy
Two options for serving the ONNX split (encoder_model.onnx, decoder_model.onnx, decoder_with_past_model.onnx):

1. **Encoder-only on Triton** (simpler): Serve just the encoder on Triton (where compute cost lives), keep autoregressive decoding loop in FastAPI. Gets a clean table row faster.
2. **Triton ensemble pipeline** (complete): Chain encoder → decoder as a Triton ensemble. More complete but more complex to wire up.

Recommendation: start with option 1 to get numbers, move to option 2 only if time allows.

## Key Framing
- Baseline single-request latency already meets targets (147ms HTR vs 5s budget, 9.3ms search vs 1s budget)
- Triton value is **throughput under concurrent load** via dynamic batching, not single-request latency
- Key metric to answer: how many pages/minute when a batch upload arrives?
- MiniLM headroom (100x under budget) means re-ranking or post-processing can be added later
