import os, sys
import pytest
torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.train_decoder_head import make_batches, polarity_loss, polarity_pairs  # noqa: E402


def _noul(pid=None, sign=1, n=2):
    r = {"state": "s " * 20, "question": "q", "options": [{"id": "yes", "description": "y"}, {"id": "no", "description": "n"}][:n]
         if n == 2 else [{"id": str(i), "description": "d"} for i in range(n)], "label": "yes" if n == 2 else "0"}
    if pid:
        r["pair_id"], r["pair_sign"] = pid, sign
    return r


def test_pairs_stay_in_one_batch():
    rows = [_noul() for _ in range(40)] + [r for k in range(10) for r in (_noul(f"p{k}"), _noul(f"p{k}"))]
    batches = make_batches(rows, batch_size=8, seed=1, world=1, rank=0, max_tokens=0, max_length=512)
    where = {}
    for b, idx in enumerate(batches):
        for i in idx:
            pid = rows[i].get("pair_id")
            if pid:
                where.setdefault(pid, set()).add(b)
    split = [p for p, bs in where.items() if len(bs) > 1]
    assert len(split) <= 1  # the power-of-two trim may split at most the boundary pair
    assert sum(len(b) for b in batches) == 60


def test_polarity_pairs_indexing_and_filters():
    rows = [_noul("a"), _noul("a"), _noul("b"), _noul("c", n=5), _noul("c", n=5), _noul()]
    assert polarity_pairs(rows) == [(0, 1, 1)]


def test_polarity_loss_direction():
    # pair 0: original says yes (margin +3), twin must say no -> margin -3 is satisfied, +3 is punished
    good = [torch.tensor([3.0, 0.0]), torch.tensor([0.0, 3.0])]
    bad = [torch.tensor([3.0, 0.0]), torch.tensor([3.0, 0.0])]
    assert polarity_loss(good, [(0, 1, 1)], 2.0) < polarity_loss(bad, [(0, 1, 1)], 2.0)
    # control: identical margins -> zero; moving -> positive
    assert polarity_loss(bad, [(0, 1, 0)], 2.0).item() == 0.0
    assert polarity_loss(good, [(0, 1, 0)], 2.0).item() > 0
    assert polarity_loss(good, [], 2.0).item() == 0.0
