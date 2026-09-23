# Data provenance and redistribution

The benchmark uses stratified samples from five third-party datasets. The
repository's code and model predictions are separate from the rights in the
source texts. Reproduce the frozen samples with `fetch.py`; verify their
SHA-256 values against `results/statistics_with_upper.json` before scoring.

| Dataset | Upstream source | Redistribution status for this repository |
|---|---|---|
| BANKING77 | [mteb/banking77](https://huggingface.co/datasets/mteb/banking77) | Source card should be checked for the exact revision and license. |
| CLINC150 | [clinc/oos-eval](https://github.com/clinc/oos-eval) | Upstream `LICENSE` is CC BY 3.0; attribution required. |
| SST-5 | [SetFit/sst5](https://huggingface.co/datasets/SetFit/sst5) | Source card does not provide a clear grant for this sampled text; fetch from upstream. |
| BoolQ | [Google Research BoolQ](https://github.com/google-research-datasets/boolean-questions) | Upstream states CC BY-SA 3.0; attribution and share-alike conditions apply. |
| AG News | [source CSV](https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv) | Upstream licensing for the news text is unclear; fetch from upstream. |

**The public repository omits `datasets/*.jsonl` and `datasets/*.meta.json`.**
It provides the frozen dataset hashes, sampling script, archived per-item model
predictions and a text-free `results/reanalysis.csv`. The latter supports
offline verification of every published paired score, interval and p-value.
Re-running models requires fetching the upstream datasets and API credentials.
Upstream files may change; a checksum mismatch means the exact historical
sample was not reconstructed and should not be silently scored as equivalent.

The local 2026-09-22 frozen files have historical `license` strings embedded
in their metadata. In particular, they say “CC BY 4.0” for CLINC150 and
BoolQ; upstream sources identify CC BY 3.0 and CC BY-SA 3.0 respectively.
Those historical strings are **not authoritative license grants**. We preserve
the original bytes so the recorded hashes and model comparisons remain
auditable, and document the correction here. The sampled data and any
reconstructed copies remain subject to their respective upstream terms.

For the methods and citation of each dataset, see [DESIGN.md](DESIGN.md) and
the source links above. Please cite the original dataset papers when using the
benchmark.
