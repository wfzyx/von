# Von

> **Status: Archived / Concluded (2026-10-17).**
> The open-source System One decision model. Calibrated Choice / Noul / Score inference from a 395M ModernBERT encoder, one forward pass, CPU-served. Weights, SDKs, and benchmarks preserved for reference.

**An open-source, non-autoregressive System One decision model.**
*Calibrated Choice / Noul / Score inference from a 395M ModernBERT encoder, one forward pass, CPU-served.*

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-wfzyx%2Fvon-blue)](https://huggingface.co/wfzyx/von)
[![PyPI](https://img.shields.io/pypi/v/von-sdk?label=von-sdk)](https://pypi.org/project/von-sdk/)
[![npm](https://img.shields.io/npm/v/von-sdk?label=npm)](https://www.npmjs.com/package/von-sdk)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

<p align="center">
  <img src="assets/von-doom.gif" alt="Von choosing movement and combat actions in Doom" width="640">
</p>
<p align="center"><sub><b>Von playing Doom.</b> Every action is one forward pass scoring six option descriptions against a text rendering of the depth buffer. Same shipped weights that answer routing questions; no policy network, no RL.</sub></p>

Von answers three question types over any JSON or text *state* without generating tokens: **Choice** (pick one of K described options, with a probability over all of them), **Noul** (probability that a condition holds) and **Score** (calibrated position on an ordinal scale). It is a drop-in server and client for the TypeSafe `/v1/systemone` wire protocol, ships as `von-sdk` for Python and TypeScript, and runs on CPU (OpenVINO), CUDA, ROCm and Apple MPS.

Weights, training data, calibration and the full benchmark record live on the [model card](https://huggingface.co/wfzyx/von). Training on your own labels, or on another encoder (German, smaller): [docs/finetune.md](docs/finetune.md). This README is about using it.

## Install

```bash
pip install von-sdk          # or: uv add von-sdk
bun add von-sdk              # TypeScript / Node
docker run --rm -p 8000:8000 -v von-hf:/data/huggingface ghcr.io/wfzyx/von:cpu
```

Weights (~3 GB) are fetched from the Hub on first use into `HF_HOME`.

## Use

```python
import von

r = von.decide(
    state="Database replication lag on cluster us-west-2 exceeded 45 seconds.",
    choices={
        "infrastructure": "Database, hardware, network, or server failures",
        "billing": "Invoices, payments, refunds, subscription queries",
        "feature_request": "Requests for new platform capabilities",
    },
    instructions="Classify the root cause domain of this incident.",
)
r.choice, r.confidence, r.probabilities   # 'infrastructure', 0.84, {...}

p = von.judge(state="Connection pool exhausted on port 5432.",
              instructions="Is this blocking customers?",
              criteria={"true": "Customer-facing requests fail or stall", "false": "Internal only"})
s = von.rate(state="Memory at 98%, OOM killer firing.", criteria=["Nominal", "Degraded", "Critical"],
             instructions="Assess degradation level.")
```

Several questions over one state cost one forward pass:

```python
resp = von.system_one(
    state={"ticket": "INC-4091", "message": "Payment gateway timeouts on charge authorizations. Urgent."},
    questions={
        "intent": von.choice(instructions="Nature of the ticket?", criteria={"payment_failure": "...", "access_issue": "..."}),
        "is_urgent": von.noul(instructions="Needs immediate SLA intervention?"),
        "severity": von.score(instructions="Rate severity.", criteria=["Low", "Medium", "High", "Critical"]),
    },
)
resp.answers["intent"].choice, resp.answers["is_urgent"].noul, resp.answers["severity"].score
```

TypeScript mirrors the same API (`decide`, `judge`, `rate`, `systemOne`).

Two rules that save you a bad afternoon: `judge` **without** `criteria` is the weakest path (generic "holds / is false" descriptions, surface cues win) — pass `criteria` or phrase it as a described 3-way Choice. And Von is **English only**; other languages get token matching with confidence it has not earned.

**Acting on confidence.** `von.patterns.confidence_gate(state, questions, threshold=0.80)` splits answers into `automatic` / `escalate`. The 0.80 default is measured, not guessed (`benchmarks/sweep_threshold.py`, held-out jabr v2 + coding-agent probes, Von 1.2 weights): Choice at ≥0.80 keeps 25% of items at **92.4%** accuracy (90% lower bound 89.4%, n=702; the rest sit at 59%); Noul keeps 16% at 83% (gated on `noul_raw`, the pre-band probability — the committed `noul` sits in a fixed band and carries no gate signal). Lowest cutoff that clears 90% kept-accuracy with 90% confidence: 0.82. In a cascade with Qwen3.5-4B on JevBench public, Von-first matches the 4B alone from 0.75 up (cost lever, not an accuracy lever). Out of domain the ranking itself breaks (judge pairs, agent-shadow probes: kept-accuracy flat at any cutoff) — a threshold cannot rescue that, only labels and a refit can. Full curves: `results/threshold_sweep_*.json`.

## Wire protocol

`von serve` exposes `/v1/systemone`, byte-compatible with the TypeSafe specification. Anything that talks to Jev talks to Von by changing the base URL; the SDKs do the same via `VON_BASE_URL`.

```bash
von serve --host 0.0.0.0 --port 8000        # auto-selects cuda / mps / openvino:gpu / openvino:cpu / cpu
curl -X POST localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "von-1.3.0",
  "state": {"error": "Disk volume /var/log at 98% capacity."},
  "questions": {
    "needs_action": {"type": "noul", "instructions": "Does this require operational intervention?"},
    "team": {"type": "choice", "instructions": "Who owns this?",
             "criteria": {"sre": "Infrastructure and capacity", "app": "Application code"}}
  }
}'
```

Response extras beyond the spec, all optional for clients: `usage.input_tokens` is the real tokenizer count over every encoder pass; a middle-truncated state carries a `truncation` field plus `X-Von-Truncated` / `Warning` headers; with `--on-overflow refuse` an oversize state gets HTTP 422 mentioning the context window instead (the [Decision Index](https://github.com/apolinario/decision-index) no-truncation rule). Criteria values may be strings or structured JSON. `VON_API_KEY` turns on bearer auth (clients fall back to `TYPESAFE_API_KEY`).

## CLI

| command | what it does |
|---|---|
| `von serve` | HTTP server. Flags below. |
| `von calibrate labels.jsonl` | Refit the confidence map on your own labels, frozen weights, CPU, minutes. Writes `marker_calibration.json` into the checkpoint dir (the backend prefers it) or `--out` elsewhere. One wire question per line plus `gold` (`{"state", "instructions", "choices", "gold"}` shorthand works). Picks scalar vs feature map by k-fold CV and reports NLL/ECE for raw, shipped, scalar, map. Temperature never changes an answer, only how sure Von claims to be. |

| flag / env | default | effect |
|---|---|---|
| `--device` / `VON_DEVICE` | `auto` | `cuda`, `mps`, `openvino:gpu`, `openvino:cpu`, `cpu`. Auto prefers OpenVINO CPU over plain torch CPU. |
| `--max-state-tokens N` / `VON_MAX_STATE_TOKENS` | 8192 | States longer than N tokens are middle-truncated (60 % head, 40 % tail) so question and options always fit the window. |
| `--on-overflow truncate\|refuse` / `VON_ON_OVERFLOW` | truncate | `refuse` returns HTTP 422 on oversize states. |
| `--noul-decision band\|raw` / `VON_NOUL_DECISION` | band | `band` maps Noul `P(yes)` to `0.8 + 0.1·(p−0.5)` (mirrored below 0.5) so every answer commits outside the 0.2–0.8 abstention band; argmax and ordering unchanged. `raw` returns the calibrated posterior. Edge/slope: `VON_NOUL_BAND_EDGE`, `VON_NOUL_BAND_SLOPE`. |
| `--chains DIR` / `VON_CHAINS_DIR` | bundled library | Chain-of-options library (below). `--no-chains` / `VON_CHAINS_DIR=off` disables it. |
| `VON_CHAINS_MAX_CALLS` | 16 | Encoder sub-decisions a chained item may spend. |
| `VON_CHAINS_MAX_STATE_TOKENS` | 4096 | Chains stand down on longer states. |
| `VON_MODEL_ID` | `wfzyx/von` | Hub repo or local checkpoint directory. |
| `VON_CHECKPOINT_DIR` | — | Absolute path to a local checkpoint (`option_marker.pt` + `marker_calibration.json`). Set this from hooks/cron: the relative default `checkpoints/von-1.2` depends on the working directory. |
| `VON_CALIBRATION` | — | Pin a specific `marker_calibration.json` (e.g. one written by `von calibrate`). Without a local checkpoint, `calibrate` writes to `~/.cache/von/marker_calibration.json` and serve picks it up from there. |
| `HF_HOME` | `~/.cache/huggingface` (`/data/huggingface` in the image) | Weight cache. |

Container tags: `cpu` / `latest` (OpenVINO, `linux/amd64`), `<version>-cpu`, a UTC calver. Flags after the image name go to `von serve`. A CUDA image builds from the same `Dockerfile` with `--build-arg TORCH_BACKEND=default` (not published; the wheel set is several GB).

## Chain-of-options

Deterministic multi-step computation for temporal/numeric states, driven by Von's own Choice decisions and zero generated tokens. It fires on computable structure in the state (two dates, a date and a duration, two amounts), never on the question's wording: a regex proposer lists typed spans; every chain whose typed slots can be filled is bound (Von picks among candidate spans when a slot is ambiguous) and executed by a fixed operator library (`add_duration`, `in_zone`, `elapsed_hours`, `prorate`, `cumsum`, `is_leap_year`, …); computed datetimes feed a bounded second round so chains compose; a computed value landing on exactly one option answers through a strict matcher, otherwise Von reads the state plus every computed fact with provenance. Chains are TOML files in [`src/von/chains/library/`](src/von/chains/library/) — add your own. Latency cost lands on the hard tail (hard-tier p50 4.2 s on 4 vCPU, 0.45 s on an A10G); serve chain-heavy loads on GPU or lower `VON_CHAINS_MAX_CALLS`.

## Benchmarks

JevBench v1.4, the last board revision before Von's Noul band rule (v1.5.1 numbers, sealed-tier breakdowns, Decision Index and jabr v2 are on the [model card](https://huggingface.co/wfzyx/von)). I/C/S/K = Intelligence, Calibration, Speed, Cost; composite is their geometric mean with the generalization gate applied.

| system | params | I | C | S | K | composite | hard | sealed | p50 latency | endpoint |
|---|---|---|---|---|---|---|---|---|---|---|
| Jev 1.13 (TypeSafe, closed) | — | 53.1 | 76.3 | 83.3 | 52.0 | **63.3** | 0.741 | 0.367 | 0.65 s | api |
| hopper (Qwen3.5-4B + LoRA) | 4B | 48.0 | 79.1 | 86.8 | 58.7 | 59.4 | 0.650 | 0.341 | 0.41 s | gpu |
| jeff (GLiFormer-large) | ~400M | 36.8 | 67.9 | 63.5 | 76.6 | 30.6 | 0.377 | 0.331 | 2.03 s | cpu |
| Laya (ModernBERT-large) | 421M | 36.1 | 63.7 | 71.1 | 86.2 | 30.3 | 0.341 | 0.308 | 1.72 s | cpu |
| **Von 1.2** | **395M** | 34.5 | **75.7** | 70.5¹ | 77.8 | 27.5 | 0.373 | 0.279 | **0.34 s**¹ | cpu |
| **Von 1.3** (same weights + chains, local run) | **395M** | — | — | 88.4¹ | — | — | **0.441**² | — | 0.36 s¹ | cpu |

¹ Remeasured under the JevBench protocol: raw p50 0.096 s on a 4-vCPU Xeon 8488C (OpenVINO), 0.023 s on an A10G; both inside the Jev-class line. [`results/speed_remeasure.md`](results/speed_remeasure.md). ² Chains on vs off on the 111 public hard items: 2 vs 9 discordant, McNemar p = 0.065; zero discordant on easy, standard and jabr v2. Every accuracy claim in this repo goes through [`benchmarks/stat_gate.py`](benchmarks/stat_gate.py) (paired exact McNemar + minimum detectable effect); below the MDE it is reported as UNRESOLVABLE, not as a win. Older tables: [`docs/benchmarks.md`](docs/benchmarks.md).

## License

Apache-2.0.
