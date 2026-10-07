# Von 2 — state of the research and next steps

Last updated 2026-10-07: r3 data staged (polarity pairs built, `extra_r3.jsonl` + src tarball in S3), launch pending. This is the single file to read after a machine wipe. Everything it
references is in this repo, in `s3://model-weight/`, or on HuggingFace; nothing load-bearing lives only on a laptop.

## 1. Why Von 2 exists

Von 1.x (ModernBERT-large 395M encoder + option-marker head) hit a wall that seven encoder-side interventions
could not move on held-out data: rebalancing, long-context, continue-numeric, continue-choice, gliclass trunk init,
judge/diversity data on the encoder, chains. Every one came back null on the paired McNemar gate. See
`results/` and `docs/benchmarks.md`.

Two facts settled the pivot (both from the 2026-10-01/02 sessions):

- **Decision Index's best quadrant (DI ≥ 50, < 100 ms) is 9B-plus decoders only.** Clef-flash, Surouge, Decider
  Gemma-4 all 57+. Von 1.3 sits at 13.7. A 400M encoder is not getting there on recipe.
- **Within a size class, recipe and head beat parameter count.** Kev 9B plain = 38.5 DI; Clef-flash 9B + JointSchemaHead
  + LoRA + RL = 57.1. The 18.6-point spread is head + data + RL, not scale.

Von 2 therefore keeps the sub-1B footprint but swaps the trunk to a decoder and adopts the Clef head.
Von 1.x stays as the CPU/edge SKU (96 ms, 1.9 GB, air-gapped). Von 2 is the GPU SKU.

## 2. Architecture (frozen for comparability — do not drift between runs)

| piece | value |
|---|---|
| trunk | `Qwen/Qwen3.5-0.8B`, bf16, LM layers only (no vision tower) |
| adapter | LoRA rank 64 on q/k/v/o/gate/up/down |
| head | Clef `JointSchemaHead`, width 512, 1 routing layer, 2 layers total, 8 heads, fp32 (15.8M params) |
| loss | CE + 0.5·Brier + label smoothing; soft targets honoured when a row carries `target` |
| optimiser | AdamW, LoRA 2e-4, head 5e-4, 200 warmup then decay |
| batching | token-budgeted, 16k padded tokens per micro-batch, accum 2, DDP over 4 GPUs |
| hardware | g5.12xlarge (4× A10G) on-demand, us-east-1 usually has capacity |

Code: `training/clef_head.py` (vendored Apache-2.0 head, backbone-agnostic), `training/train_decoder_head.py`,
`training/eval_decoder.py`, `training/launch_universal_training.py --trainer decoder`. Tests: `tests/test_clef_head.py`.

## 3. Gate — the only numbers that count

Paired McNemar between two `benchmarks/data/gate_cache/*.json` dumps via `benchmarks/compare_dumps.py --base X --cand Y`.
PASS needs p < 0.05 on the paired discordant pairs; everything else is UNRESOLVABLE and treated as zero.
**Nothing below was tuned on. Never train on any of these.**

| suite | n | source |
|---|---|---|
| jabr_v2 | 869 | jabr/classifier-benchmark @ afb83be, `cases/` (clone to `~/scratch/classifier-benchmark`) |
| judge_heldout | 300 | `benchmarks/data/judge_heldout.jsonl` (HelpSteer3 val + preference-test-sets, locked sha 9107f051) |
| probes | 52 | `benchmarks/data/probes_von_shadow.jsonl` (Hagetino, issue #21) |
| jev_easy/standard/hard | 48/72/111 | JevBench public, `s3://model-weight/jevbench-public/` → `~/scratch/jevbench/datasets/public` |

Dumps committed in `benchmarks/data/gate_cache/`: `von-1.2`, `jeff-0.8b-v1.2`, `von-2-nano-r1`, `von-2-nano-r2`.
(The directory is gitignored for scratch; `git add -f` the dumps that matter.)

## 4. Results so far — absolute accuracy, held-out

| suite | n | Von-1.2 | Jeff-0.8B v1.2 | r1 | r2 |
|---|---|---|---|---|---|
| jabr_v2 | 869 | 71.3 | 72.4 | **79.1** | 75.9 |
| judge_heldout | 300 | 51.7 | 57.0 | 55.0 | **64.3** |
| probes | 52 | 65.4 | **94.2** | 67.3 | 76.9 |
| jev_hard | 111 | 37.8 | — | 35.1 | **41.4** |
| jev_standard | 72 | 55.6 | — | **73.6** | 63.9 |
| pooled | 1452 | 64.7 | 69.5¹ | 70.7 | 71.1 |

¹ Jeff pooled is over 1221 (jev tiers not scored for Jeff).

**Statistically settled (p < 0.05, paired):**
- r1 > Von-1.2 pooled +6.0, jabr, jev_standard, Noul, Choice. First PASS in the program's history.
- r2 > Von-1.2 pooled +6.5, jabr +4.6, judge +12.7, Choice +8.3.
- r2 > r1 on judge (+9.3, p=0.017). r2 < r1 on jabr (−3.1, p=0.035). The rest between them is noise.
- r2 < Jeff on probes (−17.3, p=0.022) — 13 of those 14 points are evidence_noul (6/14 vs Jeff 14/14).

**Not settled at these n:** every jev tier, every probe family, Score type. Swings of ±6–10 there are inside the MDE.

**Plain-language read:** Von 2 nano is reliably ~6 pp better than Von 1.2 on held-out data. r2 is the first
checkpoint above chance on pairwise judge. r2 traded 3 pp of jabr for 9 pp of judge — same model, different
trade-off. We do not yet beat Jeff where Jeff is strong, and that gap is one defect, not many.

Per-run detail: `results/v2-nano/r1.md`, `results/v2-nano/r2.md`.

### What each run was

| run | train rows | extra | wall | cost | val (in-dist) |
|---|---|---|---|---|---|
| r1 | 150k universal + 20k synth + 20k long | — | 2h40 | ~$27 incl. 3 failed launches (OOM×2, capacity) | 95.5 |
| r2 | same | +92k Jeff public mix, +75.5k judge/diversity | 5h42 | ~$33 | 95.7 |

Extra data: `training/prepare_jeff_mix.py` (output of firelex/jeff's public converters → Von rows, leak-filtered against
all 1443 held-out states + 880 unique option texts; 0 leaks hit) and `training/prepare_judge_diversity.py`.
Artefacts: `s3://model-weight/data_jeff_mix/{train,extra_r2}.jsonl`, `s3://model-weight/data_judge/train_clipped.jsonl`.
Checkpoints: `s3://model-weight/von-2-nano-r{1,2}/adapter/` + head + run.log.

## 5. The one named defect: Noul yes-bias

Inherited from Von-1.2, survived a trunk swap and two independent training runs — so it is data, not weights.

| | picked yes / no | expected | evidence_noul yes |
|---|---|---|---|
| r1 | 253 / 147 | 195 / 205 | 14/14 |
| r2 | 224 / 176 | 195 / 205 | 12/14 |

The Jeff mix added 30k Noul rows and moved it a third of the way. Volume alone will not finish it. The corpus has no
rows where the *same state* carries a *negated criterion* with a flipped label, so the model learns "criterion present
→ yes" instead of reading polarity.

## 6. Next steps, ranked by expected return per dollar

**Do now (one run, r3, ~$35):**

1. **Noul polarity data — BUILT 2026-10-07.** `training/prepare_noul_polarity.py` (tests: `tests/test_noul_polarity.py`)
   takes Noul rows from any corpus and emits (original, twin) pairs: same state, negated criterion, flipped label
   and reversed soft target. Twin kinds: auxiliary negation of the question with yes/no descriptions swapped (50%),
   `Is this statement false: "<yes description>"` (18%), `Is this statement true: "<no description>"` (18%), and a
   same-label quoted control (14%) so quoting alone does not read as flipping. Built from a local 200k universal
   build + the Jeff mix: 10k pairs = 20k rows, 4776 yes / 5224 no, leak-filtered (0 hits).
   Artefacts: `s3://model-weight/data_polarity/train.jsonl`; `s3://model-weight/data_jeff_mix/extra_r3.jsonl`
   = extra_r2 (168,030) + polarity (20,000) = 188,030 rows. Src tarball rebuilt with the new script.
   Checks after r3: polarity probe on evidence_noul (held-out, never tuned) and the picked-yes ratio on the 400
   held-out Noul items moving toward 195/205.
2. **Restore universal share.** r2 dropped it to 47% and paid in jabr; r3 runs `--max-train 200000`.
   Still open (not in r3): remaining Jeff converters (firelex/jeff `docs/data-sources.md`, `src/jeff/extra.py`) and
   higher per-source caps in `prepare_judge_diversity.py`.

**r3 launch (not yet run):**

```bash
./.venv/bin/python training/launch_universal_training.py --trainer decoder --on-demand \
    --region us-east-1 --only-type g5.12xlarge,g6.12xlarge,g6e.12xlarge \
    --epochs 1 --max-train 200000 --long-context 20000 --synthetic-n 20000 \
    --extra-train-s3 s3://model-weight/data_jeff_mix/extra_r3.jsonl --extra-rows 188030 \
    --s3-target s3://model-weight/von-2-nano-r3
```
Gate r3 against r2 and Jeff. Target: hold judge ≥ 63, recover jabr toward 79, evidence_noul ≥ 10/14.

**Later (only if r3 plateaus):**

3. **Qwen3.5-9B pseudo-labels** (the original "step 2"). Run a vLLM box, **measure real throughput before committing**
   (approved 2026-10-02 — do not assume the $10–13 estimate). Soft targets over ~200k unlabelled states. Weaker case
   than it looked: teacher is 83% on JevBench public, r2 is already 76% jabr / 64% judge, and distillation does not fix
   polarity.
4. **RLCD** — only if 1–3 all land and the Decision Index read still wants it.

**Shipping, when a checkpoint is worth it:** package as `von-2-nano`, GPU serving path, submit to Decision Index
(rules in the 2026-09-28 observations: no truncation, `VON_ON_OVERFLOW=refuse`), JevBench row refresh.

## 7. Operational notes that cost money to relearn

- `launch_universal_training.py --trainer decoder` auto-sizes the watchdog from total rows at 11 rows/s; pass
  `--extra-rows N` when using `--extra-train-s3`. r2 needed a manual re-arm because the default was 420 min.
- Rows that do not fit `max_length` are dropped and counted, not fatal (r1's second OOM taught that).
- Capacity: try us-east-1 first, then us-east-2, us-west-2; `--only-type g5.12xlarge,g6.12xlarge,g6e.12xlarge`.
- Tarball `s3://model-weight/von-marker-src.tar.gz` must include `benchmarks/data` or eval fails on the box. Rebuild
  after any change to `training/` or `benchmarks/`.
- AWS creds in `~/.aws/credentials`; `./.venv/bin/aws`. Never `pgrep -f` from the shell that owns the pattern.
- `/tmp` does not survive reboot; scratch belongs in `~/scratch/`.

## 8. Rebuild from zero

```bash
git clone <this repo> && cd von && uv sync
git clone https://github.com/jabr/classifier-benchmark ~/scratch/classifier-benchmark && (cd ~/scratch/classifier-benchmark && git checkout afb83be)
./.venv/bin/aws s3 sync s3://model-weight/jevbench-public/ ~/scratch/jevbench/datasets/public
./.venv/bin/python benchmarks/compare_dumps.py --base von-1.2 --cand von-2-nano-r2   # should reproduce §4
```

For r3 data: `git clone --depth 1 https://github.com/firelex/jeff ~/scratch/ext/jeff && cd $_ && uv sync --no-dev`,
run the `jeff-extra` / `jeff-probability` / `jeff.longlists` entry points, then `training/prepare_jeff_mix.py`.
MASSIVE needs a manual download to `data/public/raw/massive-1.1.tar.gz` (Amazon S3 URL in jeff's longlists module).
