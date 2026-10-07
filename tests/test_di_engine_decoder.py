"""Decision Index adapter contract for von-2-nano: validated with the kit's own `validate`, model stubbed."""
import os
import sys
from types import SimpleNamespace

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DI = os.path.expanduser("~/scratch/ext/decision-index")
if os.path.isdir(DI):
    sys.path.insert(0, DI)
pytest.importorskip("decision_index.engines.base", reason="decision-index kit not checked out at ~/scratch/ext/decision-index")
torch = pytest.importorskip("torch")

from decision_index.engines.base import Unsupported, validate  # noqa: E402
from benchmarks.di_engine_decoder import VonDecoderEngine  # noqa: E402


class _Tok:
    """Deterministic whitespace tokenizer: one id per word, so lengths are predictable."""
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [1 + (hash(w) % 50000) for w in text.replace("\n", " \n ").split(" ") if w]

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=self.encode(text))


class _Model(torch.nn.Module):
    """Returns one logit per option span, favouring the first option of each question."""
    def forward(self, batch):
        out = []
        for rec in batch["records"]:
            per_q = []
            for q in rec.questions:
                k = len(q.option_spans)
                logits = torch.full((k,), -2.0)
                logits[0] = 3.0
                per_q.append(logits)
            out.append(per_q)
        return out


def _engine(max_length=4096):
    e = VonDecoderEngine(checkpoint_dir="/nonexistent", max_length=max_length)
    e._model, e._tokenizer, e._cfg = _Model(), _Tok(), {"max_length": 4096}
    e._device = torch.device("cpu")
    return e


def test_choice_and_noul_pass_kit_validation():
    e = _engine()
    questions = {
        "dept": {"type": "choice", "instructions": "Which department?", "criteria": {"billing": "money", "tech": "bugs", "legal": None}},
        "urgent": {"type": "noul", "instructions": "Is it urgent?", "criteria": None},
    }
    resp, raw = e({"ticket": "Charged twice, fix today"}, questions)
    validate(questions, resp)
    assert resp["answers"]["dept"]["choice"] == "billing"
    assert set(resp["answers"]["dept"]["probabilities"]) == {"billing", "tech", "legal"}
    assert 0 <= resp["answers"]["urgent"]["noul"] <= 1
    assert raw["tokens"] == resp["usage"]["input_tokens"] > 0


def test_many_options_sum_within_tolerance():
    e = _engine()
    crit = {f"opt{i}": f"option number {i}" for i in range(255)}
    q = {"pick": {"type": "choice", "instructions": "Pick one", "criteria": crit}}
    resp, _ = e("state", q)
    validate(q, resp)  # 255 keys, sum within 0.01


def test_refuses_instead_of_truncating():
    e = _engine(max_length=200)
    q = {"a": {"type": "choice", "instructions": "x", "criteria": {"a": "a", "b": "b"}}}
    with pytest.raises(Unsupported, match="refusing rather than truncating"):
        e(" ".join(["word"] * 500), q)


def test_refuses_bad_option_counts_and_types():
    e = _engine()
    with pytest.raises(Unsupported):
        e("s", {"a": {"type": "choice", "instructions": "x", "criteria": {"only": "one"}}})
    with pytest.raises(Unsupported):
        e("s", {"a": {"type": "freeform", "instructions": "x"}})
