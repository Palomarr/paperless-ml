#!/bin/bash
# Benchmark Triton models using perf_analyzer
# Usage: ./benchmark_triton.sh [triton_url] [output_dir]

TRITON_URL="${1:-localhost:8001}"
OUTPUT_DIR="${2:-results/triton}"
CONCURRENCY_LEVELS="1,2,4,8,16"
MEASUREMENT_INTERVAL=5000  # ms

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Triton Benchmark — perf_analyzer"
echo "Target: $TRITON_URL"
echo "Output: $OUTPUT_DIR"
echo "============================================================"

# --- Search model (bi-encoder) ---
echo ""
echo ">>> search_model (bi-encoder, all-mpnet-base-v2)"
echo "------------------------------------------------------------"

# Generate input data for search model (input_ids + attention_mask)
python3 -c "
import json, numpy as np
data = {
    'data': [{
        'input_ids': np.ones((128,), dtype=np.int64).tolist(),
        'attention_mask': np.ones((128,), dtype=np.int64).tolist()
    }]
}
with open('$OUTPUT_DIR/search_input.json', 'w') as f:
    json.dump(data, f)
print('Generated search_input.json')
"

perf_analyzer \
    -m search_model \
    -u "$TRITON_URL" \
    -i grpc \
    --input-data "$OUTPUT_DIR/search_input.json" \
    --concurrency-range "$CONCURRENCY_LEVELS" \
    --measurement-interval "$MEASUREMENT_INTERVAL" \
    -f "$OUTPUT_DIR/search_model_results.csv" \
    2>&1 | tee "$OUTPUT_DIR/search_model.log"

# --- HTR model (TrOCR encoder) ---
echo ""
echo ">>> htr_model (TrOCR encoder)"
echo "------------------------------------------------------------"

# Generate input data for HTR model (pixel_values)
python3 -c "
import json, numpy as np
data = {
    'data': [{
        'pixel_values': np.random.randn(3, 384, 384).astype(np.float32).tolist()
    }]
}
with open('$OUTPUT_DIR/htr_input.json', 'w') as f:
    json.dump(data, f)
print('Generated htr_input.json')
"

perf_analyzer \
    -m htr_model \
    -u "$TRITON_URL" \
    -i grpc \
    --input-data "$OUTPUT_DIR/htr_input.json" \
    --concurrency-range "$CONCURRENCY_LEVELS" \
    --measurement-interval "$MEASUREMENT_INTERVAL" \
    -f "$OUTPUT_DIR/htr_model_results.csv" \
    2>&1 | tee "$OUTPUT_DIR/htr_model.log"

echo ""
echo "============================================================"
echo "Done. Results in $OUTPUT_DIR/"
echo "============================================================"
