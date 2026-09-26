"""Native-logit decoder scorer on JevBench public tiers -- the Von 2 feasibility probe.

The 46-59 composite systems on the v1.4 board (semif/localjev/jobe/hopper) are
all the same recipe: a frozen or lightly-LoRA'd Qwen3.5-4B read out through
its OWN next-token distribution over the answer keys, one forward per item.
No custom head (the kev-* systems that bolted a pointer head onto a decoder
sit at 25-36). This reproduces that readout with our harness's item shapes so
the number is comparable to Von's 57.1% public accuracy.

Protocol per item:
  prompt = document + question + lettered answers, ending in "Answer:"
  one forward; take logits at the last position restricted to the letter
  tokens " A", " B", ...; softmax over those = the answer distribution.
  Choice -> argmax letter; Noul -> letters map to yes/no; Score -> letters
  map to ordered levels (argmax, as JevBench scores it).

One forward per item makes a 4B feasible on this 4-core CPU overnight; the
per-option mean-logprob protocol in probe_decoder_ceiling.py is k forwards.

Usage:
  uv run --with accelerate python -m benchmarks.probe_native_logit \\
      --model Qwen/Qwen3.5-4B --tiers hard standard easy \\
      --dump benchmarks/data/native_logit_qwen35_4b.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from benchmarks.eval_hard_fast import TIER_FILES, build_request, load_rows  # noqa: E402

LETTERS = "ABCDEFGHIJKLMNOP"


PROMPT_STYLES = ("qa", "framed")


def _prompt(state: str, instructions: str, keys: List[str], crit: Dict[str, str], style: str = "framed") -> str:
    if style == "qa":
        opts = "\n".join(f"{LETTERS[i]}. {crit[k]}" for i, k in enumerate(keys))
        return (
            "You are a careful decision engine. Read the document, then answer the question "
            "by choosing exactly one lettered option. Respond with the letter only.\n\n"
            f"DOCUMENT:\n{state}\n\nQUESTION: {instructions}\n\nOPTIONS:\n{opts}\n\nAnswer:"
        )
    # "framed": the instruction is task framing that comes BEFORE the input, so
    # its caveat sentences ("...use coding_agent even if...", "a mention does
    # not establish intent") stop reading as an answer hint sitting next to
    # the options. Keys are shown with their descriptions.
    opts = "\n".join(f"{LETTERS[i]}. {k.replace('_', ' ')} \u2014 {crit[k]}" for i, k in enumerate(keys))
    return (
        f"Task: {instructions}\n\n"
        f"Input:\n{state}\n\n"
        f"Options:\n{opts}\n\n"
        "Which single option is correct for this input? Reply with the letter only.\nAnswer:"
    )


def _letter_token_ids(tok, n: int) -> List[int]:
    ids = []
    for i in range(n):
        cand = tok(" " + LETTERS[i], add_special_tokens=False)["input_ids"]
        if len(cand) != 1:
            cand = tok(LETTERS[i], add_special_tokens=False)["input_ids"]
        ids.append(cand[-1])
    return ids


def score_item_server(url: str, prompt: str, n_opts: int) -> List[float]:
    """llama-server path: one /completion call, next-token distribution read
    from top_logprobs and renormalized over the option letters. Same readout
    as the in-process path, on a quantized GGUF at llama.cpp speed."""
    import math
    import urllib.request
    letters = LETTERS[:n_opts]
    body = json.dumps({
        "prompt": prompt, "n_predict": 1, "temperature": 0, "n_probs": 30, "cache_prompt": False,
        # Grammar constrains the *sampled* token to a letter; top_logprobs stay
        # pre-grammar, so the distribution is still the model's own. If no
        # letter carries mass in the top-30 (tokenizers that want "\n" or "**"
        # first), fall back to the constrained sample as an argmax-only answer.
        "grammar": "root ::= [" + letters + "]",
    }).encode()
    req = urllib.request.Request(url.rstrip("/") + "/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.loads(r.read())
    probs = [0.0] * n_opts
    cands = out.get("completion_probabilities") or []
    top = (cands[0].get("top_logprobs") or cands[0].get("top_probs") or []) if cands else []
    for c in top:
        t = (c.get("token") or "").strip().rstrip(".")
        if len(t) == 1 and t in letters:
            lp = c.get("logprob")
            probs[letters.index(t)] += math.exp(lp) if lp is not None else float(c.get("prob", 0.0))
    tot = sum(probs)
    if tot <= 0:
        t = (out.get("content") or "").strip()
        if t in letters:
            probs[letters.index(t)] = 1.0
            tot = 1.0
        else:
            return [1.0 / n_opts] * n_opts
    return [p / tot for p in probs]


@torch.no_grad()
def score_item(model, tok, prompt: str, n_opts: int, max_ctx: int, use_chat: bool) -> List[float]:
    if use_chat:
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        ids = tok(text, add_special_tokens=False)["input_ids"]
    else:
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
    if len(ids) > max_ctx:
        # keep the tail: question + options + answer cue live there
        ids = ids[-max_ctx:]
    logits = model(torch.tensor([ids])).logits[0, -1].float()
    lt = _letter_token_ids(tok, n_opts)
    return torch.log_softmax(logits[lt], dim=-1).exp().tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tiers", nargs="+", default=["hard", "standard", "easy"], choices=list(TIER_FILES))
    ap.add_argument("--max-ctx", type=int, default=6144)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--no-chat", action="store_true", help="raw prompt instead of the chat template")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="", help="JSON list of item ids to restrict to (dev slice)")
    ap.add_argument("--dump", default="")
    ap.add_argument("--style", default="framed", choices=PROMPT_STYLES)
    ap.add_argument("--template", default="qwen", choices=["qwen", "server", "none"],
                    help="server path only: qwen = ChatML w/ empty think block; server = model's own template via /apply-template")
    ap.add_argument("--server", default="", help="llama-server base URL (e.g. http://127.0.0.1:8080); skips in-process loading")
    args = ap.parse_args()

    if args.server:
        tok = model = None
        use_chat = not args.no_chat
    else:
        torch.set_num_threads(args.threads)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype))
        model.eval()
        use_chat = (not args.no_chat) and tok.chat_template is not None

    rows: List[dict] = []
    for t in args.tiers:
        rows.extend(load_rows(TIER_FILES[t]))
    if args.ids:
        keep = set(json.load(open(args.ids)))
        rows = [r for r in rows if r["id"] in keep]
    if args.limit:
        rows = rows[:args.limit]
    # Longest first so a stall shows up early and progress estimates are honest.
    rows.sort(key=lambda r: -len(json.dumps(r["state"])))
    print(f"{len(rows)} items on {args.model} dtype={args.dtype} chat={use_chat}", flush=True)

    results = []
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        state, instr, crit = build_request(row, "plain")
        keys = list(crit)
        qtype = row["question"].get("type", "choice")
        prompt = _prompt(state, instr, keys, crit, args.style)
        if args.server:
            if args.template == "qwen":  # Qwen ChatML, thinking off; /completion applies no template itself
                prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            elif args.template == "server":  # the GGUF's own chat template via llama-server
                import urllib.request
                body = json.dumps({"messages": [{"role": "user", "content": prompt}]}).encode()
                req = urllib.request.Request(args.server.rstrip("/") + "/apply-template", data=body,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    prompt = json.loads(r.read())["prompt"]
                # Thinking-mode templates leave "<think>" open, so the next token
                # is the start of a rationale ("The...") and no letter has mass.
                # Close the block: we read the answer, we don't decode reasoning.
                if prompt.rstrip().endswith("<think>"):
                    prompt = prompt.rstrip() + "\n\n</think>\n\n"
            probs = score_item_server(args.server, prompt, len(keys))
        else:
            probs = score_item(model, tok, prompt, len(keys), args.max_ctx, use_chat)
        pick_key = keys[max(range(len(keys)), key=lambda j: probs[j])]
        pick = {"true": "yes", "false": "no"}.get(pick_key, pick_key) if qtype == "noul" else pick_key
        gold = str(row["expected"]).strip().lower()
        results.append({
            "id": row["id"], "tier": row["id"].split("-")[0], "family": row.get("family"), "type": qtype,
            "n_options": len(keys), "pick": pick, "expected": gold, "hit": str(pick).lower() == gold,
            "probs": dict(zip(keys, [round(p, 4) for p in probs])), "confidence": round(max(probs), 4),
        })
        if i % 5 == 0 or i == len(rows):
            acc = sum(r["hit"] for r in results) / len(results)
            el = time.time() - t0
            print(f"  [{i}/{len(rows)}] acc {acc:.1%}  {el:.0f}s  eta {el/i*(len(rows)-i)/60:.0f}m", flush=True)
            if args.dump:
                json.dump({"model": args.model, "partial": i < len(rows), "results": results},
                          open(args.dump, "w", encoding="utf-8"), indent=1)

    print(f"\n== {args.model} native-logit ==")
    for tier in ["hard", "original", "easy"]:
        sel = [r for r in results if r["tier"] == tier]
        if sel:
            print(f"  {tier:9s} {sum(r['hit'] for r in sel)}/{len(sel)} = {sum(r['hit'] for r in sel)/len(sel):.1%}")
    n = len(results)
    print(f"  public    {sum(r['hit'] for r in results)}/{n} = {sum(r['hit'] for r in results)/n:.1%}   (Von-1.2: 57.1%)")
    fams = sorted({r["family"] for r in results if r["family"]})
    for f in fams:
        sel = [r for r in results if r["family"] == f]
        print(f"    {f:18s} {sum(r['hit'] for r in sel)}/{len(sel)}")
    if args.dump:
        json.dump({"model": args.model, "partial": False, "results": results},
                  open(args.dump, "w", encoding="utf-8"), indent=1)
        print(f"wrote {args.dump}")


if __name__ == "__main__":
    main()
