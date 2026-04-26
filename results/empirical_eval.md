# Empirical Evaluation

| Label | Retrieval Hit Rate | Answer Keyword Hit Rate | Avg Latency (ms) |
|---|---:|---:|---:|
| keyword | 83.33% | N/A | 9.0 |
| vector-hash | 83.33% | N/A | 18.3 |
| hybrid-hash | 100.00% | N/A | 27.3 |
| vector-bge-small-en-v1.5 | 100.00% | N/A | 81.2 |
| hybrid-bge-small-en-v1.5 | 100.00% | N/A | 33.0 |
| vector-all-MiniLM-L6-v2 | 66.67% | N/A | 23.6 |
| hybrid-all-MiniLM-L6-v2 | 100.00% | N/A | 28.6 |
| dspy+Qwen/Qwen2.5-0.5B-Instruct+bge-small-en-v1.5 | 100.00% | 66.67% | 35004.1 |
| dspy+Qwen/Qwen2.5-1.5B-Instruct+bge-small-en-v1.5 | 100.00% | 66.67% | 74250.0 |
