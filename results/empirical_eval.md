# Empirical Evaluation

| Label | Retrieval Hit Rate | Answer Keyword Hit Rate | Avg Latency (ms) |
|---|---:|---:|---:|
| keyword | 83.33% | N/A | 14.4 |
| vector-hash | 83.33% | N/A | 188.8 |
| hybrid-hash | 83.33% | N/A | 31.7 |
| vector-bge-small-en-v1.5 | 66.67% | N/A | 95.5 |
| hybrid-bge-small-en-v1.5 | 100.00% | N/A | 36.5 |
| vector-all-MiniLM-L6-v2 | 83.33% | N/A | 87.2 |
| hybrid-all-MiniLM-L6-v2 | 100.00% | N/A | 30.6 |
| langchain+qwen2.5-0.5b-vllm+BAAI/bge-small-en-v1.5 | 100.00% | 66.67% | 53.1 |
| langchain+qwen2.5-1.5b-vllm+BAAI/bge-small-en-v1.5 | 100.00% | 83.33% | 64735.8 |
