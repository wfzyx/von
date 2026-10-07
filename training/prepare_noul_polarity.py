"""Contrastive Noul polarity pairs: same state, negated criterion, flipped label.

Why: Von-1.2, von-2-nano r1 and r2 all over-pick "yes" on held-out Noul (r2: 224/176 vs 195/205 expected,
evidence_noul 12/14 yes). The corpus has ~19 Noul question templates over hundreds of states each, and no row
where the *same state* carries a *negated criterion* with the opposite label, so the model learns
"criterion text present -> yes" instead of reading polarity. See docs/von-2.md §5.

Input: Von rows ({state, question, options:[{id:yes},{id:no}], label[, target]}) from any corpus file.
Output: for each sampled Noul row, the original plus one negated twin (--pairs), with the twin built by one of:

  neg_question   auxiliary negation of the question ("Does the response contain X?" -> "Does the response not
                 contain X?"; statements -> "It is false that ..."); specific yes/no descriptions are swapped so
                 they still describe the yes/no sides of the new question; generic ones (Yes/No, True/False) stay.
  claim_false    'Is this statement false: "<true description>"' with descriptions swapped.
  claim_of_neg   'Is this statement true: "<false description>"' with descriptions swapped.
  control        'Is this statement true: "<true description>"', same label (so quoting != flipping).

Every twin flips the label and reverses a soft `target` (except control). Rows touching any held-out gate
are dropped via prepare_jeff_mix.LeakGuard. Nothing here reads the probes.

    python training/prepare_noul_polarity.py --in data_universal/train.jsonl ~/scratch/data/jeff_train.jsonl \\
        --out ~/scratch/data_polarity/train.jsonl --pairs 10000
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_jeff_mix import LeakGuard, heldout_rows, norm  # noqa: E402

GENERIC = {
    norm(x)
    for x in (
        "Yes", "No", "True", "False", "Yes.", "No.",
        "Yes, condition holds true.", "No, condition is false.",
        "Condition is satisfied", "Condition is not satisfied",
        "Yes, it happens.", "No, it does not happen.",
        "Condition holds", "Condition does not hold",
    )
}

_AUX = r"(does|do|is|are|was|were|will|can|could|has|have|had|did|should|would|must|may)"
_PREFIX = re.compile(r"^((?:based|given|considering|according|using|from)\b[^,]*,\s*)(.+)$", re.I | re.S)
_AUX_HEAD = re.compile(rf"^{_AUX}\s+", re.I)
# Auxiliaries that take a bare-infinitive verb: "not" goes before the verb. For be-auxiliaries the complement is an
# adjective / participle / NP, so "not" goes right after the subject NP.
_DO_AUX = {"does", "do", "did", "will", "can", "could", "should", "would", "must", "may"}
_DETS = {"the", "this", "that", "these", "those", "a", "an", "any", "each", "every", "either", "both", "all", "some"}
_PRONOUNS = {"it", "he", "she", "they", "we", "you", "i", "there"}
_NP_LINKERS = {"to", "of", "and", "or", "in", "for", "with", "on", "from", "by", "at", "about", "between", "under", "'s"}
_VERBS = {
    "contain", "include", "involve", "violate", "represent", "support", "have", "win", "meet", "indicate",
    "mention", "express", "describe", "show", "require", "qualify", "match", "exceed", "state", "conflict",
    "comply", "need", "use", "refer", "make", "pose", "appear", "seem", "sound", "read", "look", "apply",
    "happen", "occur", "hold", "land", "draw", "get", "reach", "fall", "pass", "fail", "answer", "address",
    "follow", "satisfy", "fulfil", "fulfill", "disclose", "reveal", "present", "provide", "offer", "give",
    "ask", "request", "demand", "want", "intend", "plan", "agree", "accept", "reject", "deny", "admit",
    "claim", "assert", "imply", "suggest", "recommend", "allow", "permit", "prohibit", "restrict", "block",
    "come", "go", "work", "run", "fit", "belong", "depend", "rely", "count", "warrant", "justify", "cover",
    "carry", "bear", "share", "compare", "differ", "relate", "correspond", "align", "agree", "cause", "lead",
    "result", "affect", "change", "break", "breach", "cross", "trigger", "raise", "lower", "increase",
    "decrease", "stay", "remain", "keep", "continue", "stop", "start", "end", "finish", "complete", "resolve",
    "solve", "fix", "handle", "manage", "deal", "treat", "consider", "regard", "treat", "take", "put", "set",
    "say", "tell", "speak", "talk", "write", "report", "note", "mark", "flag", "signal", "convey", "carry",
    "display", "exhibit", "demonstrate", "prove", "confirm", "verify", "check", "test", "measure", "count",
    "identify", "name", "list", "cite", "quote", "reference", "link", "point", "attempt", "try", "manage",
    "know", "understand", "believe", "think", "feel", "like", "prefer", "expect", "anticipate", "predict",
    "owe", "pay", "charge", "cost", "spend", "save", "earn", "lose", "gain", "benefit", "suffer", "hurt",
    "help", "serve", "assist", "enable", "let", "force", "compel", "push", "pull", "move", "turn", "roll",
    "pick", "choose", "select", "decide", "determine", "conclude", "infer", "deduce", "derive", "compute",
    "calculate", "estimate", "guess", "assume", "presume", "suppose", "hypothesize", "posit", "argue",
    "entail", "necessitate", "mandate", "oblige", "bind", "commit", "promise", "guarantee", "ensure", "assure",
}


def is_generic(desc: str) -> bool:
    d = norm(desc)
    return d in GENERIC or len(d.split()) <= 2


def _negate_inverted(body: str) -> Optional[str]:
    """'Does the inspection sample contain X' -> 'Does the inspection sample not contain X'."""
    m = _AUX_HEAD.match(body)
    if not m:
        return None
    aux = m.group(1)
    toks = body[m.end():].split()
    if not toks:
        return None
    # Subject NP: det + word | `ticked` | pronoun | Capitalised name(s), then any 'linker + word' continuations.
    i = 0
    first = toks[0]
    if first.lower() in _DETS:
        i = 2
    elif first.startswith("`") or first.lower() in _PRONOUNS or first[0].isupper():
        i = 1
        while i < len(toks) and toks[i][0].isupper() and toks[i].lower() not in _VERBS:
            i += 1  # multi-word proper name
    else:
        return None
    while i + 1 < len(toks) and toks[i].lower() in _NP_LINKERS:
        i += 2
        if i < len(toks) and toks[i - 1].lower() in _DETS:
            i += 1
    if i >= len(toks):
        return None
    if aux.lower() in _DO_AUX:
        # Walk to the first known verb; a direct hit at the NP boundary is the common case.
        j = i
        while j < len(toks) and j < i + 4 and toks[j].lower().strip(",;:") not in _VERBS:
            j += 1
        if j >= len(toks) or toks[j].lower().strip(",;:") not in _VERBS:
            return None
        i = j
    elif toks[i].lower() in _VERBS:
        return None  # "Is this dish or ingredient list vegan": NP boundary ambiguous, let a claim template handle it
    if any(t.lower() in ("not", "never", "no") or t.lower().endswith("n't") for t in toks[: i + 1]):
        return None  # main clause already negated
    return " ".join([aux] + toks[:i] + ["not"] + toks[i:])


def negate_question(question: str) -> Optional[str]:
    """Return a grammatical negation of a yes/no question or statement, or None when no rule applies."""
    q = question.strip()
    # Negate the first sentence only; trailing instructions ("Give probabilities ...") are kept verbatim.
    tail = ""
    qm = q.find("?")
    if 0 <= qm < len(q) - 1:
        q, tail = q[: qm + 1], q[qm + 1:]
    prefix = ""
    pm = _PREFIX.match(q)
    if pm:
        prefix, q = pm.group(1), pm.group(2)
    q_ends_q = q.endswith("?")
    body = q[:-1] if q_ends_q else q
    out = _negate_inverted(body)
    if out is not None:
        return prefix + out + ("?" if q_ends_q else "") + tail
    if not q_ends_q and body and body[0].isupper() and not body.lower().startswith(("does ", "is ", "do ", "are ")):
        # Declarative criterion ("The message conveys urgency ...").
        if re.search(r"\bnot\b|n't\b", body):
            return None
        return prefix + "It is false that " + body[0].lower() + body[1:] + tail
    return None


_LEAD = re.compile(r"^(?:yes|no|true|false)\s*[,.:;-]\s*", re.I)


def quotable(desc: str) -> str:
    """Description as a bare claim: drop a leading 'Yes, ' / 'No, ' and the final period."""
    d = _LEAD.sub("", desc.strip()).rstrip(".")
    return d[:1].upper() + d[1:] if d else d


def noul_sides(row: dict) -> Optional[Tuple[str, str]]:
    opts = row.get("options") or []
    if [o.get("id") for o in opts] != ["yes", "no"]:
        return None
    return str(opts[0].get("description", "")), str(opts[1].get("description", ""))


def flip_label(label: str) -> str:
    return "no" if label == "yes" else "yes"


def flipped_target(row: dict) -> Optional[List[float]]:
    t = row.get("target")
    if isinstance(t, list) and len(t) == 2:
        return [float(t[1]), float(t[0])]
    return None


def make_twin(row: dict, kind: str, rng: random.Random) -> Optional[dict]:
    sides = noul_sides(row)
    if sides is None or row.get("label") not in ("yes", "no"):
        return None
    yes_d, no_d = sides
    specific = not (is_generic(yes_d) or is_generic(no_d))
    label = str(row["label"])
    out: Dict[str, Any] = {"state": row["state"]}

    if kind == "neg_question":
        nq = negate_question(str(row["question"]))
        if nq is None:
            return None
        out["question"] = nq
        out["options"] = ([{"id": "yes", "description": no_d}, {"id": "no", "description": yes_d}] if specific
                          else [{"id": "yes", "description": yes_d}, {"id": "no", "description": no_d}])
        out["label"] = flip_label(label)
        ft = flipped_target(row)
        if ft:
            out["target"] = ft
    elif kind == "claim_false":
        if not specific:
            return None
        tmpl = rng.choice([
            'Is this statement false for the state above: "{d}"',
            'Does the state contradict the following? "{d}"',
            'Is the following claim incorrect here? "{d}"',
        ])
        out["question"] = tmpl.format(d=quotable(yes_d))
        out["options"] = [{"id": "yes", "description": no_d}, {"id": "no", "description": yes_d}]
        out["label"] = flip_label(label)
        ft = flipped_target(row)
        if ft:
            out["target"] = ft
    elif kind == "claim_of_neg":
        if not specific:
            return None
        tmpl = rng.choice([
            'Is this statement true for the state above: "{d}"',
            'Does the state support the following? "{d}"',
            'Is the following claim correct here? "{d}"',
        ])
        out["question"] = tmpl.format(d=quotable(no_d))
        out["options"] = [{"id": "yes", "description": no_d}, {"id": "no", "description": yes_d}]
        out["label"] = flip_label(label)
        ft = flipped_target(row)
        if ft:
            out["target"] = ft
    elif kind == "control":
        if not specific:
            return None
        tmpl = rng.choice([
            'Is this statement true for the state above: "{d}"',
            'Does the state support the following? "{d}"',
            'Is the following claim correct here? "{d}"',
        ])
        out["question"] = tmpl.format(d=quotable(yes_d))
        out["options"] = [{"id": "yes", "description": yes_d}, {"id": "no", "description": no_d}]
        out["label"] = label
        if isinstance(row.get("target"), list):
            out["target"] = list(row["target"])
    else:
        raise ValueError(kind)

    src = row.get("source", "?")
    out["source"] = f"polarity:{kind}:{src if isinstance(src, str) else json.dumps(src, sort_keys=True)}"
    if row.get("licence"):
        out["licence"] = row["licence"]
    return out


KIND_WEIGHTS = {"neg_question": 0.45, "claim_false": 0.2, "claim_of_neg": 0.2, "control": 0.15}


def pick_kind(rng: random.Random) -> str:
    kinds = list(KIND_WEIGHTS)
    return rng.choices(kinds, weights=[KIND_WEIGHTS[k] for k in kinds], k=1)[0]


def read_rows(paths: List[str]) -> List[dict]:
    rows: List[dict] = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh.read().split("\n"):  # not splitlines(): U+2028 inside JSON strings
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inputs", nargs="+", required=True, help="Von jsonl corpora to draw Noul rows from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", type=int, default=10000, help="number of (original, twin) pairs to emit")
    ap.add_argument("--no-original", action="store_true", help="emit twins only")
    ap.add_argument("--max-per-question", type=int, default=1500,
                    help="cap source rows per original question template so a few templates do not dominate")
    ap.add_argument("--seed", type=int, default=31)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    guard = LeakGuard(heldout_rows())
    print(f"leak guard: {len(guard.states)} states, {len(guard.options)} unique option texts", file=sys.stderr)

    rows = [r for r in read_rows(a.inputs) if noul_sides(r) is not None and r.get("label") in ("yes", "no")]
    rng.shuffle(rows)
    per_q: collections.Counter = collections.Counter()
    pool: List[dict] = []
    for r in rows:
        k = norm(r["question"])
        if per_q[k] >= a.max_per_question:
            continue
        per_q[k] += 1
        pool.append(r)
    print(f"{len(rows)} noul rows -> {len(pool)} after per-question cap ({len(per_q)} templates)", file=sys.stderr)

    stats: collections.Counter = collections.Counter()
    out_rows: List[dict] = []
    seen_states: set = set()
    for r in pool:
        if len(out_rows) >= a.pairs * (1 if a.no_original else 2):
            break
        key = (norm(r["state"]), norm(r["question"]))
        if key in seen_states:
            stats["dup_state"] += 1
            continue
        if guard.leaks(r["state"], [o["description"] for o in r["options"]]):
            stats["leak"] += 1
            continue
        kind = pick_kind(rng)
        twin = make_twin(r, kind, rng)
        if twin is None:
            # fall back to any rule that applies
            for alt in ("neg_question", "claim_false", "claim_of_neg"):
                twin = make_twin(r, alt, rng)
                if twin is not None:
                    kind = alt
                    break
        if twin is None:
            stats["no_rule"] += 1
            continue
        seen_states.add(key)
        stats[kind] += 1
        stats[f"label:{twin['label']}"] += 1
        if not a.no_original:
            out_rows.append(r)
        out_rows.append(twin)

    rng.shuffle(out_rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out_rows)} rows to {a.out}", file=sys.stderr)
    for k, v in sorted(stats.items()):
        print(f"  {k:16s} {v}", file=sys.stderr)


if __name__ == "__main__":
    main()
