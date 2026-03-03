FROM python:3.12-slim
# Note: vLLM is installed on the GPU host or via a GPU-enabled base image.
# This Dockerfile documents the service structure and installs the health sidecar deps.
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir fastapi uvicorn[standard] openai pyyaml
RUN chmod +x startup.sh
EXPOSE 8000 8001
CMD ["./startup.sh"]
