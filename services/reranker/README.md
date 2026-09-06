---
title: SeoulDoc Multilingual Evidence Reranker
emoji: 🔎
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
suggested_hardware: t4-small
startup_duration_timeout: 1h
pinned: false
---

# SeoulDoc multilingual evidence reranker

Private GPU API for reranking English and Korean review evidence with
`BAAI/bge-reranker-v2-m3`.

The service runs Hugging Face Text Embeddings Inference and exposes:

- `GET /health`
- `GET /info`
- `POST /rerank`

Request:

```json
{
  "query": "친절하고 과잉진료가 없는 진료",
  "texts": ["보통이었어요", "과잉진료 없이 친절했어요"]
}
```

The caller maps returned indexes back to locally owned `evidence_id` and
`place_id` values. The service never controls evidence ownership metadata.
