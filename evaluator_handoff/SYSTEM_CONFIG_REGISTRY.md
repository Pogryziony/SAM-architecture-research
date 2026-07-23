# System configuration registry

Each submitted system records:

| Field | Example |
|-------|---------|
| `system_id` | `nexus` |
| `profile` | `grounded` |
| `config_hash` | sha256 from `nexus-config-identity-v2` |
| `allow_synth_fallback` | `false` for safe profiles |
| `model_id` / revision | exact provider string |
| `checkpoint_id` | sha256 of weights |
| `domain_pack_id/version` | external pack identity |
| `graph_snapshot_id` | frozen graph hash |
| `comparison_mode` | `controlled` or `system_level` |
