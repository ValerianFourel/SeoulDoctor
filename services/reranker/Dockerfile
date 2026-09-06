FROM ghcr.io/huggingface/text-embeddings-inference:turing-1.9

EXPOSE 7860

CMD ["--model-id", "BAAI/bge-reranker-v2-m3", "--port", "7860", "--max-client-batch-size", "256", "--auto-truncate"]
