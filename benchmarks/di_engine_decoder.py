"""Decision Index adapter for von-2-nano checkpoints (Qwen3.5-0.8B + LoRA + Clef JointSchemaHead).

Harness `Engine` contract (decision-index docs/engines.md): `__call__(state, questions) -> (response, raw)`;
never truncate, never drop options, never adapt the prompt per benchmark; raise `Unsupported` when a declared
capacity limit is hit. `training.clef_head.encode_record` *does* truncate the state to fit `max_length`, so this
engine measures the untruncated sequence first and refuses instead.

The pass/fail bar for the Von 2 programme (decided 2026-10-07, docs/von-2.md §0): the public index from this
engine on the 0.3 suite must beat LiquidAI d1-omni-600M's 17.9, or the project stops.

Usage (repo root; the kit at $DI, suite built per its README):
    export HF_HUB_DISABLE_XET=1
    uv run --with-editable . --with-editable $DI \\
        python -m decision_index pipeline --engine benchmarks.di_engine_decoder:VonDecoderEngine \\
        --option checkpoint_dir=checkpoints/von-2-nano-r3 --out runs/von-2-nano-r3
Smoke test: --option limit... use `suite sample --n 100` and `run` first.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))


def _text(x: Any) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, separators=(",", ":"))


class VonDecoderEngine:
    """One joint forward pass per request: every question of the request is packed into the same sequence."""

    name = "von-2-nano"
    latency = ("In-process request wall time including prompt construction; excludes model loading. "
               "GPU bf16 autocast when available, one request at a time.")

    def __init__(self, **options: Any) -> None:
        self.options = options
        self._checkpoint_dir = options.get("checkpoint_dir") or "checkpoints/von-2-nano"
        self._device_name = options.get("device")
        # 0 = the checkpoint's training max_length. Longer values run the trunk past its trained positions.
        self._max_length = int(options.get("max_length") or 0)
        self._model = None
        self._tokenizer = None
        self._cfg: Dict[str, Any] = {}
        self.provenance = {
            "repo": "wfzyx/von", "checkpoint": self._checkpoint_dir,
            "kind": "LoRA r64 + JointSchemaHead", "base_model": "Qwen/Qwen3.5-0.8B",
        }

    # -- loading --------------------------------------------------------------------------------------------
    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from training.eval_decoder import load_checkpoint

        device = torch.device(self._device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
        if device.type == "cpu":
            torch.set_num_threads(max(1, os.cpu_count() or 4))
        self._device = device
        self._model, self._tokenizer, self._cfg = load_checkpoint(self._checkpoint_dir, device)
        if not self._max_length:
            self._max_length = int(self._cfg.get("max_length", 4096))
        self.provenance["base_model"] = self._cfg.get("base_model_id", self.provenance["base_model"])
        self.provenance["max_length"] = self._max_length

    def warmup(self) -> None:
        self._load()
        self("The color is red.", {"c": {"type": "choice", "instructions": "Which color is named?",
                                         "criteria": {"red": "red", "blue": "blue"}}})

    # -- inference ------------------------------------------------------------------------------------------
    def __call__(self, state: Any, questions: Dict[str, dict]) -> Tuple[dict, Any]:
        from decision_index.engines.base import Unsupported
        import torch
        from training.clef_head import collate_records, encode_record, systemone_answer

        self._load()
        tok = self._tokenizer
        wire_questions: Dict[str, dict] = {}
        for key, q in questions.items():
            qtype = q.get("type")
            if qtype not in ("choice", "noul", "score"):
                raise Unsupported(f"unknown question type {qtype!r}")
            wq: Dict[str, Any] = {"type": qtype, "instructions": _text(q.get("instructions") or key)}
            crit = q.get("criteria")
            if qtype == "choice":
                if not isinstance(crit, dict) or not (2 <= len(crit) <= 255):
                    raise Unsupported(f"choice needs a 2-255 option criteria map, got {type(crit).__name__} of {len(crit) if crit else 0}")
                wq["criteria"] = {str(k): (v if v is None or isinstance(v, str) else _text(v)) for k, v in crit.items()}
            elif qtype == "score":
                if not isinstance(crit, list) or len(crit) < 2:
                    raise Unsupported("score needs an ordered list of 2+ levels")
                wq["criteria"] = [str(c) for c in crit]
            else:
                wq["criteria"] = crit if isinstance(crit, dict) else None
            wire_questions[key] = wq

        record = {"id": "req", "state": state, "questions": wire_questions}
        # Measure untruncated: encode with a huge budget and compare to the real one.
        probe = encode_record(tok, record, max_length=1 << 30)
        n_tokens = len(probe.input_ids)
        if n_tokens > self._max_length:
            raise Unsupported(f"request packs to {n_tokens} tokens, over the {self._max_length}-token window; "
                              "refusing rather than truncating the state")

        t0 = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self._device.type == "cuda"):
            per_question = self._model(collate_records([probe], tok.pad_token_id, self._device))[0]

        answers: Dict[str, dict] = {}
        for enc_q, logits in zip(probe.questions, per_question):
            probs = dict(zip(enc_q.option_ids, logits.float().softmax(-1).tolist()))
            wq = wire_questions[enc_q.question_id]
            ans = systemone_answer(wq, probs)
            if wq["type"] == "choice":
                # Every declared option must carry a probability and the set must sum to 1 within 0.01:
                # use the raw softmax, not systemone_answer's 4-decimal rounding (drifts on 200+ options).
                full = {str(o): float(probs.get(str(o), 0.0)) for o in wq["criteria"]}
                total = sum(full.values())
                if total <= 0:
                    raise Unsupported("degenerate zero probability mass across all options")
                ans["probabilities"] = {o: v / total for o, v in full.items()}
            elif wq["type"] == "noul":
                ans["probabilities"] = {"true": float(probs["true"]), "false": float(probs["false"])}
            answers[enc_q.question_id] = ans

        response = {"model": f"von-2-nano@{os.path.basename(self._checkpoint_dir.rstrip('/'))}",
                    "answers": answers, "usage": {"input_tokens": n_tokens}}
        return response, {"seconds": time.perf_counter() - t0, "tokens": n_tokens}
