FROM vllm/vllm-openai:v0.16.0
ENV VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
WORKDIR /app
COPY . .
# openai is bundled in the vLLM base image; only install health sidecar deps here
RUN pip install --no-cache-dir fastapi uvicorn[standard] pyyaml
RUN chmod +x startup.sh
EXPOSE 8000 8001
CMD ["./startup.sh"]
