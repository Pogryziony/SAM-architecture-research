$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Resolve-Path ".").Path
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:SSL_CERT_FILE = (python -c "import certifi; print(certifi.where())")
$env:REQUESTS_CA_BUNDLE = $env:SSL_CERT_FILE

$arms = @(
  @{ arm = "closed_book"; out = "benchmarks/results/phase4_qwen_closed_book_oracle_v1.json" },
  @{ arm = "long_context"; out = "benchmarks/results/phase4_qwen_long_context_oracle_v1.json" },
  @{ arm = "bm25_rag"; out = "benchmarks/results/phase4_bm25_rag_qwen_oracle_v1.json" },
  @{ arm = "dense_rag"; out = "benchmarks/results/phase4_dense_rag_qwen_oracle_v1.json" },
  @{ arm = "hybrid_rag"; out = "benchmarks/results/phase4_hybrid_rag_qwen_oracle_v1.json" },
  @{ arm = "hybrid_rerank_rag"; out = "benchmarks/results/phase4_hybrid_rerank_rag_qwen_oracle_v1.json" },
  @{ arm = "nexus_graph_qwen"; out = "benchmarks/results/phase4_nexus_graph_evidence_qwen_oracle_v1.json" }
)

foreach ($a in $arms) {
  if (Test-Path $a.out) {
    Write-Host "SKIP exists $($a.out)"
    continue
  }
  Write-Host "=== START $($a.arm) $(Get-Date -Format o) ==="
  python benchmarks/run_phase4_arms.py --arm $a.arm --output $a.out
  if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED $($a.arm) exit=$LASTEXITCODE"
    exit $LASTEXITCODE
  }
  Write-Host "=== DONE $($a.arm) $(Get-Date -Format o) ==="
}
Write-Host "ALL ARMS COMPLETE"
