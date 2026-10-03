# Packaged Data Artifacts

These archives contain the data required by the reproduction package. Extract
them from the `package/` directory.

| Archive | Restored path | Purpose |
|---|---|---|
| `package_data.zip` | `data/` | Issue manifest, 3,708 extracted instances, reference library, and embeddings |
| `rq2_data.zip` | `rqs/rq2/data/` | RQ2 case manifest, situations, output manifest, and detector outputs |
| `rq4_data.zip` | `rqs/rq4/data/` | RQ4 situations, output manifest, and Gemini ablation outputs |

From `package/`, verify and extract every archive with:

```bash
python extract_artifacts.py
```

Use `python extract_artifacts.py --verify-only` to check archive hashes without
extracting them. Hashes are listed in `SHA256SUMS.txt`.
