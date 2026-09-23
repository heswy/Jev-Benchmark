# Reproducing the benchmark

## Levels of reproduction

| Level | Inputs | Network or keys | What it checks |
|---|---|---|---|
| Statistical audit | `results/reanalysis.csv` and `results/statistics_with_upper.json` | None | Every score, per-dataset paired result, bootstrap interval and permutation p-value |
| Re-score saved predictions | Frozen upstream dataset samples and `results/raw-complete.tar.gz` | Dataset download only | Parsing, sample coverage and correctness from recorded outputs |
| New API run | Frozen samples and your own provider credentials | Dataset and model APIs | A new observation under today's provider/model conditions |

The first level is fully offline and uses Python 3.10+ with no third-party package:

```bash
python3 -m unittest discover -s tests -v
python3 -m scripts.verify_release
```

The verifier checks 1,050 unique paired items and reproduces the saved values with the recorded seed (42) and draw count (5,000). `results/reanalysis.csv` contains only item IDs and correctness bits, not the upstream text. The underlying complete model responses are packaged in `results/raw-complete.tar.gz`; the completed Qwen3.8 run snapshot and historical summaries are in `results/history.tar.gz`. Incomplete 397B predictions remain local and are not in these public archives. Both checksums are recorded in `results/ARCHIVES.sha256`.

## Reconstruct the frozen gold

The five text-bearing `datasets/*.jsonl` files are intentionally absent from the public repository. Fetch them from their upstream owners:

```bash
python3 -m pip install -r requirements-fetch.txt
python3 fetch.py
```

`fetch.py` uses seed 42, fixed sample counts and the source URLs recorded in `datasets/README.md`. It creates five JSONL files plus metadata. The sources currently use `main`/`master` or a mirror rather than content-addressed revisions, so **the upstream content can change**. The next command checks both per-file SHA-256 and full sample-ID coverage and stops on a mismatch:

```bash
tar -xzf results/raw-complete.tar.gz
python3 -m jevbench.statistics \
  --models qwen3.5-4b qwen3.5-9b qwen3.5-27b qwen3.5-35b qwen3.5-122b qwen3.8-27b \
  --draws 5000 --seed 42 --out results/recomputed.json
```

Compare `results/recomputed.json` against `results/statistics_with_upper.json` for scores and tests. The `generated_at_utc` field changes. If an upstream checksum does not match, **do not override the check** or present the new sample as a reproduction of the published run; the offline audit remains available. The historical `license` strings embedded in the frozen samples are not authoritative; see [DATA_PROVENANCE.md](DATA_PROVENANCE.md).

Regenerate the figures and descriptive efficiency file after reconstructing the gold:

```bash
python3 -m scripts.build_publication
```

This produces `report/figures/*.svg`, `results/reanalysis.csv`, and `results/efficiency.json` from the local frozen samples and model outputs. The source figures in the repository were built from the verified 2026-09-22/23 snapshot.

## Run a model again

Copy `.env.example` to `.env` and set your own `OPENROUTER_API_KEY` and/or `SILICONFLOW_API_KEY`. Never commit the key file. The provider, serving revision, queues and pricing may differ from the historical runs.

```bash
python3 -m jevbench.runner --models qwen3.8-27b \
  --datasets banking77 clinc150 sst5 boolq agnews --concurrency 6
```

The runner reuses complete valid predictions in `results/raw/` and checkpoints every five items. To force a genuinely new run, make a copy of the repository and move the target model's existing raw files out of that copy before running. Preserve the published files and hashes in the original checkout. A new run should receive its own manifest and should never be silently merged with the frozen 2026-09-23 result.

## Protocol

- Primary outcome: five-dataset mean of end-to-end accuracy, each dataset weighted 20%; failed and invalid predictions count as wrong.
- Choice: selected label; `oos_tau=0.35` only when `oos` is in that item's option set.
- Score: most probable one of five levels must match exactly.
- Noul: `p(true) >= 0.5` gives true.
- Uncertainty: paired bootstrap resamples within each dataset. The main comparison uses a stratified paired interval and paired permutation test; dataset comparisons use paired intervals and exact McNemar tests.
- Qwen protocol: zero temperature, thinking disabled, JSON response contract. Jev uses its native System One response. This is matched by task interface, not by underlying probability mechanism.

The full implementation is in `jevbench/statistics.py`, `jevbench/prompts.py`, and `jevbench/adapters.py`. Pre-run plan changes and an excluded, incomplete 397B attempt are recorded in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). Historical baseline requests did not archive exact provider versions or immutable request snapshots; the saved responses support **reanalysis**, but cannot prove a byte-for-byte replay of the supplier's historical service.
