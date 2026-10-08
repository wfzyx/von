"""Train the Von nano-decoder variant: Qwen3.5-0.8B (frozen) + LoRA + JointSchemaHead.

Reads the same jsonl rows as train_option_marker.py ({state, question, options, label[, target]}),
packs each row with training/clef_head.py (one code path with eval), and optimises
label-smoothed cross-entropy + Brier against the (soft or one-hot) target over options.
Only LoRA weights and the head train. torchrun-launched, one process per GPU.

Output directory layout (what eval_decoder.py loads):
    adapter/                 PEFT LoRA adapter (adapter_config.json, adapter_model.safetensors)
    head.safetensors         JointSchemaHead state
    decoder_config.json      {base_model_id, max_length, head: {...}}
    tokenizer files
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.clef_head import (  # noqa: E402
    DecoderHeadModel, JointSchemaHead, collate_records, encode_record, row_to_record, soft_target_for, text_model_of)

# Linear modules of the Qwen3.5 text trunk: full attention, gated-deltanet linear attention, and MLP.
# Scoped to the language model so the (unused) vision tower gets no adapters.
LORA_TARGET_REGEX = (r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj"
                     r"|in_proj_qkvz|in_proj_ba|in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)$")


def log(msg: str) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_rows(path: str, limit: int = 0) -> List[dict]:
    rows: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: malformed JSON ({exc.msg}); refusing a partially readable corpus") from exc
            opts = row.get("options")
            if not isinstance(opts, list) or len(opts) < 2:
                raise ValueError(f"{path}:{lineno}: record needs >=2 options")
            if str(row.get("label")) not in [str(o.get("id")) for o in opts]:
                raise ValueError(f"{path}:{lineno}: label {row.get('label')!r} matches no option id")
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"{path}: corpus is empty")
    return rows


def estimate_tokens(row: dict) -> int:
    n = len(row.get("state", "")) + len(row.get("question", "")) + sum(len(str(o.get("description", ""))) + 24 for o in row["options"])
    return 64 + n // 4


def make_batches(rows: List[dict], batch_size: int, seed: int, world: int, rank: int, max_tokens: int = 0,
                 max_length: int = 4096) -> List[List[int]]:
    """Length-bucketed, token-budgeted batches, sharded by rank.

    Shuffle globally, sort inside mega-chunks, then fill each batch until it holds ``batch_size`` rows or
    ``rows * longest_row_tokens`` would exceed ``max_tokens`` (padded cost). A fixed row count OOMs on a
    bucket of long rows and starves the GPU on short ones; the budget keeps the padded token count flat.
    """
    rng = random.Random(seed)
    # Rows sharing a pair_id (polarity twins) travel as one unit so both land in the same micro-batch and
    # the contrastive term can see them together. Units are shuffled/sorted/budgeted in place of rows.
    by_pair: Dict[str, List[int]] = {}
    units: List[List[int]] = []
    for i, r in enumerate(rows):
        pid = r.get("pair_id")
        if pid:
            if pid not in by_pair:
                by_pair[pid] = []
                units.append(by_pair[pid])
            by_pair[pid].append(i)
        else:
            units.append([i])
    rng.shuffle(units)
    chunk = batch_size * 64
    batches: List[List[int]] = []
    for start in range(0, len(units), chunk):
        piece = sorted(units[start:start + chunk], key=lambda u: max(estimate_tokens(rows[i]) for i in u))
        cur: List[int] = []
        for u in piece:
            longest = min(max_length, max(estimate_tokens(rows[i]) for i in u))  # sorted ascending
            if cur and (len(cur) >= batch_size or (max_tokens and (len(cur) + len(u)) * longest > max_tokens)):
                # Close at a power-of-two row count so (rows, padded_len) shapes repeat for Triton autotune.
                keep = 1 << (len(cur).bit_length() - 1)
                batches.append(cur[:keep])
                cur = cur[keep:]
            cur.extend(u)
        if cur:
            batches.append(cur)
    rng.shuffle(batches)
    usable = (len(batches) // world) * world  # every rank sees the same number of steps
    return batches[rank:usable:world]


SKIPPED_ROWS = 0  # rows whose schema alone exceeds max_length (options too long to pack); counted, not fatal
PAD_TO = 512  # padded length granularity; see clef_head.collate_records


def encode_batch(tokenizer: Any, rows: List[dict], max_length: int, device: torch.device) -> Tuple[Optional[Dict[str, Any]], List[int], List[Optional[List[float]]]]:
    global SKIPPED_ROWS
    records, targets, softs, kept_rows = [], [], [], []
    for row in rows:
        record, target = row_to_record(row)
        try:
            enc = encode_record(tokenizer, record, max_length=max_length)
        except ValueError:
            SKIPPED_ROWS += 1
            continue
        q = enc.questions[0]
        records.append(enc)
        kept_rows.append(row)
        targets.append(q.option_ids.index(target))
        softs.append(soft_target_for(row, q.option_ids))
    if not records:
        return None, targets, softs
    batch = collate_records(records, tokenizer.pad_token_id, device, pad_to=PAD_TO)
    batch["pairs"] = polarity_pairs(kept_rows)
    return batch, targets, softs


def polarity_pairs(kept_rows: List[dict]) -> List[Tuple[int, int, int]]:
    """(index_a, index_b, sign) over the encoded records (same order as kept_rows) for every pair_id present
    exactly twice in this batch and two-option on both sides.

    sign=1: the twin carries a negated criterion, so P(yes) must move the opposite way to the original.
    sign=0: same-label control (quoted criterion) -- P(yes) must *not* move.
    """
    where: Dict[str, List[Tuple[int, int]]] = {}
    for n, row in enumerate(kept_rows):
        pid = row.get("pair_id")
        if pid and len(row.get("options") or []) == 2:
            where.setdefault(pid, []).append((n, int(row.get("pair_sign", 1))))
    return [(m[0][0], m[1][0], m[0][1]) for m in where.values() if len(m) == 2]


def polarity_loss(logits_per_row: List[torch.Tensor], pairs: List[Tuple[int, int, int]], margin: float) -> torch.Tensor:
    """Contrastive term on polarity pairs: the yes-logit margin (logit_yes - logit_no) must flip sign across a
    negated pair by at least ``margin`` (hinge), and must stay put across a same-label control (L2 on the gap).
    Returns 0 when the batch holds no pairs. Rows must be two-option (noul)."""
    terms = []
    for a, b, sign in pairs:
        la, lb = logits_per_row[a].float(), logits_per_row[b].float()
        if la.numel() != 2 or lb.numel() != 2:
            continue
        ma, mb = la[0] - la[1], lb[0] - lb[1]
        if sign:
            terms.append(torch.nn.functional.softplus(margin - (ma * -mb)))  # want ma and mb opposite in sign
        else:
            terms.append((ma - mb) ** 2)
    if not terms:
        return logits_per_row[0].new_zeros(())
    return torch.stack(terms).mean()


def decision_loss(logits_per_row: List[torch.Tensor], targets: List[int], softs: List[Optional[List[float]]],
                  brier_weight: float, label_smoothing: float) -> Tuple[torch.Tensor, int]:
    """Mean over rows of CE(dist, p) + brier_weight * ||p - dist||^2, dist label-smoothed. Returns (loss, n_correct)."""
    ce_terms, brier_terms, correct = [], [], 0
    for logits, target, soft in zip(logits_per_row, targets, softs):
        probs = torch.softmax(logits.float(), dim=-1)
        k = probs.shape[0]
        if soft is not None and len(soft) == k:
            dist = torch.tensor(soft, dtype=probs.dtype, device=probs.device)
        else:
            dist = torch.zeros_like(probs)
            dist[target] = 1.0
        if label_smoothing > 0:
            dist = (1.0 - label_smoothing) * dist + label_smoothing / k
        ce_terms.append(-torch.sum(dist * torch.log(probs + 1e-8)))
        brier_terms.append(torch.sum((probs - dist) ** 2))
        correct += int(torch.argmax(probs).item() == target)
    return torch.stack(ce_terms).mean() + brier_weight * torch.stack(brier_terms).mean(), correct


def build_model(base_model_id: str, head_cfg: Dict[str, int], lora_r: int, lora_alpha: int, lora_dropout: float,
                device: torch.device, adapter_dir: str = "", head_path: str = "") -> Tuple[DecoderHeadModel, Any]:
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(base_model_id, dtype=torch.bfloat16, device_map={"": str(device)})
    backbone.config.use_cache = False
    for p in backbone.parameters():
        p.requires_grad_(False)
    if adapter_dir:
        backbone = PeftModel.from_pretrained(backbone, adapter_dir, is_trainable=True)
    else:
        backbone = get_peft_model(backbone, LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                                       target_modules=LORA_TARGET_REGEX, bias="none"))
    hidden = backbone.get_base_model().config.text_config.hidden_size
    head = JointSchemaHead(hidden_size=hidden, **head_cfg).to(device)
    if head_path:
        from safetensors.torch import load_file
        head.load_state_dict(load_file(head_path), strict=True)
    return DecoderHeadModel(backbone, head), tokenizer


def save_checkpoint(model: DecoderHeadModel, tokenizer: Any, out_dir: str, base_model_id: str, max_length: int, extra: Dict[str, Any]) -> None:
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    model.language_model.save_pretrained(os.path.join(out_dir, "adapter"))
    save_file({k: v.detach().cpu().contiguous() for k, v in model.head.state_dict().items()}, os.path.join(out_dir, "head.safetensors"))
    tokenizer.save_pretrained(out_dir)
    cfg = {"base_model_id": base_model_id, "max_length": max_length, "head": model.head.config, **extra}
    tmp = os.path.join(out_dir, "decoder_config.json.tmp")
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, os.path.join(out_dir, "decoder_config.json"))


def s3_sync(local: str, target: str) -> None:
    if not target:
        return
    r = subprocess.run(["aws", "s3", "sync", local, target.rstrip("/") + "/", "--only-show-errors"], capture_output=True, text=True)
    log(f"s3 sync -> {target}: rc={r.returncode} {r.stderr.strip()[:200]}")


@torch.no_grad()
def evaluate(model: DecoderHeadModel, tokenizer: Any, rows: List[dict], batch_size: int, max_length: int,
             device: torch.device, world: int, rank: int, max_tokens: int = 0) -> Dict[str, Any]:
    model.eval()
    mine = rows[rank::world]
    correct = torch.zeros(3, device=device)
    total = torch.zeros(3, device=device)
    nll = torch.zeros((), device=device)
    for idx in make_batches(mine, batch_size, 0, 1, 0, max_tokens, max_length):
        chunk = [mine[i] for i in idx]
        batch, targets, _ = encode_batch(tokenizer, chunk, max_length, device)
        if batch is None:
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outs = model(batch)
        for rec, logits, target in zip(batch["records"], outs, targets):
            probs = torch.softmax(logits[0].float(), -1)
            t = rec.questions[0].question_type
            total[t] += 1
            correct[t] += int(torch.argmax(probs).item() == target)
            nll += -torch.log(probs[target] + 1e-8)
    if world > 1:
        dist.all_reduce(correct); dist.all_reduce(total); dist.all_reduce(nll)
    model.train()
    names = ["noul", "choice", "score"]
    n = float(total.sum().item())
    return {"n": int(n), "acc": float(correct.sum().item()) / max(n, 1), "nll": float(nll.item()) / max(n, 1),
            "by_type": {names[i]: {"n": int(total[i].item()), "acc": float(correct[i].item()) / max(float(total[i].item()), 1)} for i in range(3)}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_data", required=True)
    ap.add_argument("--val_data", default="")
    ap.add_argument("--base_model_id", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--output_dir", default="checkpoints/von-2-nano")
    ap.add_argument("--s3_target", default="")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=0, help="optimizer steps; 0 = epochs")
    ap.add_argument("--max_train", type=int, default=0)
    ap.add_argument("--max_val", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=16, help="max rows per micro-batch")
    ap.add_argument("--max_tokens", type=int, default=16384, help="max padded tokens per micro-batch (rows x longest); 0 = rows only")
    ap.add_argument("--grad_accum_steps", type=int, default=4)
    ap.add_argument("--polarity_weight", type=float, default=0.0,
                    help="weight of the contrastive term on rows sharing a pair_id (prepare_noul_polarity.py); 0 = off")
    ap.add_argument("--polarity_margin", type=float, default=2.0, help="hinge margin on the product of yes-logit margins")
    ap.add_argument("--max_length", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-4, help="LoRA learning rate")
    ap.add_argument("--head_lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--head_width", type=int, default=512)
    ap.add_argument("--head_routing_layers", type=int, default=1)
    ap.add_argument("--head_layers", type=int, default=2)
    ap.add_argument("--head_heads", type=int, default=8)
    ap.add_argument("--head_feedforward", type=int, default=2048)
    ap.add_argument("--head_dropout", type=float, default=0.1)
    ap.add_argument("--brier_weight", type=float, default=0.5)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    a = ap.parse_args()

    distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if distributed:
        dist.init_process_group("nccl")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(a.seed + rank)

    head_cfg = {"width": a.head_width, "routing_layers": a.head_routing_layers, "layers": a.head_layers,
                "heads": a.head_heads, "feedforward": a.head_feedforward, "dropout": a.head_dropout}
    model, tokenizer = build_model(a.base_model_id, head_cfg, a.lora_r, a.lora_alpha, a.lora_dropout, device)
    if a.gradient_checkpointing:
        base, _ = text_model_of(model.language_model)
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
    lora_params = [p for n, p in model.language_model.named_parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    log(f"trainable: lora {sum(p.numel() for p in lora_params)/1e6:.1f}M, head {sum(p.numel() for p in head_params)/1e6:.1f}M; "
        f"backbone {sum(p.numel() for p in model.language_model.parameters())/1e6:.0f}M")

    train_rows = load_rows(a.train_data, a.max_train)
    val_rows = load_rows(a.val_data, a.max_val) if a.val_data else []
    log(f"rows: train {len(train_rows)}, val {len(val_rows)}")

    ddp_model: Any = model
    if distributed:
        ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)
    opt = torch.optim.AdamW([{"params": lora_params, "lr": a.lr}, {"params": head_params, "lr": a.head_lr}],
                            weight_decay=a.weight_decay, betas=(0.9, 0.98))

    epoch_batches = len(make_batches(train_rows, a.batch_size, a.seed, world, rank, a.max_tokens, a.max_length))
    total_steps = a.max_steps or max(1, int(math.ceil(epoch_batches * a.epochs / a.grad_accum_steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, a.warmup_steps)) * max(0.05, 1.0 - s / max(1, total_steps)))
    log(f"steps: {total_steps} optimizer steps ({epoch_batches} micro-batches/epoch/rank, accum {a.grad_accum_steps})")

    step, micro, seen, run_loss, run_correct, run_n, run_batches = 0, 0, 0, 0.0, 0, 0, 0
    run_pairs, run_ploss = 0, 0.0
    t0 = time.time()
    model.train()
    epoch = 0
    done = False
    while not done:
        for idx in make_batches(train_rows, a.batch_size, a.seed + epoch, world, rank, a.max_tokens, a.max_length):
            rows = [train_rows[i] for i in idx]
            batch, targets, softs = encode_batch(tokenizer, rows, a.max_length, device)
            if batch is None:
                # DDP needs every rank to step together; feed a zero contribution instead of skipping.
                batch, targets, softs = encode_batch(tokenizer, [train_rows[0]], a.max_length, device)
                if batch is None:
                    raise ValueError("train_rows[0] does not pack; pick a corpus whose first row fits --max_length")
                targets = targets[:1]; softs = [None]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outs = ddp_model(batch)
            loss, correct = decision_loss([o[0] for o in outs], targets, softs, a.brier_weight, a.label_smoothing)
            if a.polarity_weight > 0 and batch.get("pairs"):
                ploss = polarity_loss([o[0] for o in outs], batch["pairs"], a.polarity_margin)
                loss = loss + a.polarity_weight * ploss
                run_pairs += len(batch["pairs"]); run_ploss += float(ploss.item()) * len(batch["pairs"])
            (loss / a.grad_accum_steps).backward()
            run_loss += loss.item(); run_correct += correct; run_n += len(rows); seen += len(rows); run_batches += 1
            micro += 1
            if micro % a.grad_accum_steps:
                continue
            torch.nn.utils.clip_grad_norm_(lora_params + head_params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0 or step == 1:
                el = time.time() - t0
                log(f"step {step}/{total_steps} loss {run_loss / max(1, run_batches):.4f} "
                    f"acc {run_correct/max(1,run_n):.3f} lr {sched.get_last_lr()[0]:.2e} {seen*world/el:.1f} rows/s eta {el/step*(total_steps-step)/60:.0f}min"
                    + (f" pairs {run_pairs} ploss {run_ploss/run_pairs:.3f}" if run_pairs else ""))
                run_pairs, run_ploss = 0, 0.0
                if SKIPPED_ROWS:
                    log(f"  skipped so far (schema > max_length): {SKIPPED_ROWS}")
                run_loss, run_correct, run_n, run_batches = 0.0, 0, 0, 0
            if val_rows and step % a.val_every == 0:
                log(f"val @ {step}: {json.dumps(evaluate(model, tokenizer, val_rows, a.batch_size, a.max_length, device, world, rank, a.max_tokens))}")
            if rank == 0 and step % a.save_every == 0:
                save_checkpoint(model, tokenizer, a.output_dir, a.base_model_id, a.max_length, {"step": step, "args": vars(a)})
                s3_sync(a.output_dir, a.s3_target)
            if step >= total_steps:
                done = True
                break
        epoch += 1

    final = evaluate(model, tokenizer, val_rows, a.batch_size, a.max_length, device, world, rank, a.max_tokens) if val_rows else {}
    log(f"final val: {json.dumps(final)}")
    if rank == 0:
        save_checkpoint(model, tokenizer, a.output_dir, a.base_model_id, a.max_length, {"step": step, "args": vars(a), "val": final})
        s3_sync(a.output_dir, a.s3_target)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
