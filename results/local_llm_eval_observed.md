# Observed Local LLM Evaluation

These rows were collected from prior local `vLLM` runs on this machine using the same document set and DSPy pipeline. They are kept separate from `empirical_eval.md` because the latter is regenerated directly by `scripts/run_eval.py`.

| Label | Retrieval Hit Rate | Answer Keyword Hit Rate | Avg Latency (ms) |
|---|---:|---:|---:|
| dspy+Qwen/Qwen2.5-0.5B-Instruct+bge-small-en-v1.5 | 100.00% | 66.67% | 35004.1 |
| dspy+Qwen/Qwen2.5-1.5B-Instruct+bge-small-en-v1.5 | 100.00% | 66.67% | 74250.0 |

To reproduce these rows, start two OpenAI-compatible local servers and run:

```bash
python scripts/run_eval.py \
  --input "/path/to/Advising FAQ (12-19-24 Update).docx" \
  --input "/path/to/ug_bulletin_2025-2026.pdf" \
  --llm-configs data/eval/llm_eval_configs.example.json
```
