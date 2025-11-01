# Dockerfile
FROM vllm/vllm-openai:latest

# Set secrets
ENV HUGGING_FACE_HUB_TOKEN=$HUGGINGFACE_HUB_TOKEN
ENV MODEL_NAME=$MODEL_NAME
ENV MAX_MODEL_LEN=$MAX_MODEL_LEN
ENV MAX_NUM_SEQS=$MAX_NUM_SEQS

# Entrypoint to serve the model
ENTRYPOINT ["bash", "-c", "vllm serve ${MODEL_NAME} --host 0.0.0.0 --port 8080 --max-model-len ${MAX_MODEL_LEN} --max-num-seqs ${MAX_NUM_SEQS}"]
