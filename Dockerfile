FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY trident ./trident
RUN pip install --no-cache-dir .

EXPOSE 8080
ENTRYPOINT ["trident"]
CMD ["--config", "/etc/trident/trident.yaml", "--port", "8080"]
