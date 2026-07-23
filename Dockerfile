# Reproducible CPU evaluation image for NEXUS evidence regeneration.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app
COPY requirements.lock.txt pyproject.toml README.md ./
COPY nexus ./nexus
COPY benchmarks ./benchmarks
COPY docs ./docs
COPY sam-lm ./sam-lm
COPY tests ./tests
COPY evaluator_handoff ./evaluator_handoff

RUN pip install --no-cache-dir -r requirements.lock.txt \
    && pip install --no-cache-dir -e ".[test]"

CMD ["pytest", "-q", "tests/test_dataset_identity.py", "tests/test_holm_and_relevance.py"]
