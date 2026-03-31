vllm serve "/vepfs-mlp2/c20250602/500050/models/Qwen3-32B" \
  --served-model-name qwen3-32b-local \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.9 \
  --reasoning-parser qwen3 \
  2>&1 | tee /vepfs-mlp2/c20250602/500050/lqh/k12_graphbench/benchmark_output/test_result/logs/qwen3-32b