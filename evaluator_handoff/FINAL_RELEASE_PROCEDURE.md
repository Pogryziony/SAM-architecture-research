# Final release procedure

1. Validate all artifacts (`tools/validate_handoff.py` + schema validators).
2. Confirm adjudication complete or mark human metrics `PENDING_ADJUDICATION`.
3. Compute content SHA-256 of the result package directory.
4. Publish package with:
   - preregistration snapshot
   - corpus hashes
   - per-question results
   - aggregates
   - statistics (if eligible)
   - signer identity / commit
5. Never upgrade exploratory runs to “sealed final” after the fact.
