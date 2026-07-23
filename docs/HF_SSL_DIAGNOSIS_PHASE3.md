# Hugging Face SSL diagnosis (Phase 3)

**Date (UTC):** 2026-07-22  
**Environment:** Windows 10, Python 3.12.10

## Original failure

`tests/test_entity_candidate.py` semantic-stage tests fail while
`sentence-transformers` / `huggingface_hub` request:

`https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/adapter_config.json`

Error:

```text
[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate
```

Followed by `RuntimeError: Cannot send a request, as the client has been closed.`

## Diagnosis

| Check | Result |
|-------|--------|
| `urllib` GET `https://huggingface.co` | OK (system trust store) |
| `huggingface_hub.hf_hub_download` with `SSL_CERT_FILE=certifi` | OK for `config.json` |
| `SentenceTransformer(...)` with `HF_HUB_OFFLINE=1` after cache warm | OK (local snapshot) |
| Default pytest process (no offline / incomplete trust for httpx) | FAIL on HEAD `adapter_config.json` |

Root cause: environment TLS trust for the `httpx` client used by `huggingface_hub`
does not reliably validate Hugging Face model HEAD requests. Plain site fetch can
succeed while model HEAD fails. **TLS verification was not disabled.**

## Remediation attempted (safe)

1. Point CA env vars at the certifi bundle:
   - `SSL_CERT_FILE`
   - `REQUESTS_CA_BUNDLE`
   - `CURL_CA_BUNDLE`
2. Prefer previously verified local cache with offline mode:
   - `HF_HUB_OFFLINE=1`
   - `TRANSFORMERS_OFFLINE=1`
3. Model identity remains `sentence-transformers/all-MiniLM-L6-v2`
   (snapshot under `%USERPROFILE%\.cache\huggingface\hub\...`).

## Recommended secure procedure

```powershell
$env:SSL_CERT_FILE = (python -c "import certifi; print(certifi.where())")
$env:REQUESTS_CA_BUNDLE = $env:SSL_CERT_FILE
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python -m pytest tests/test_entity_candidate.py -v
```

If the local cache is incomplete, warm it once with CA env vars set (TLS on),
then re-run offline. Do **not** set `CURL_CA_BUNDLE=""` / `HF_HUB_DISABLE_SSL` /
`verify=False`.

## Status

- Semantic HF-dependent tests: **environment-sensitive**; may remain skipped/failing
  in CI without a pre-seeded cache + offline flags.
- Does **not** affect grounded lexical/`oracle_v1` Phase 3 performance claims.
- Dense/hybrid RAG arms remain `NOT_RUN` independently (no spending auth + SSL risk).
