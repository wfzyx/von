import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training"))

from prepare_noul_polarity import make_twin, negate_question, quotable  # noqa: E402


def test_negate_inverted_questions():
    cases = {
        "Does the response contain information that is not supported by the knowledge/document?":
            "Does the response not contain information that is not supported by the knowledge/document?",
        "Will the inspection sample contain at least one defective unit? Give probabilities that reflect the evidence in the state.":
            "Will the inspection sample not contain at least one defective unit? Give probabilities that reflect the evidence in the state.",
        "Do `sentence1` and `sentence2` have the same meaning?": "Do `sentence1` and `sentence2` not have the same meaning?",
        "Does `evidence` support `claim`?": "Does `evidence` not support `claim`?",
        "Based only on `passage`, is the answer to `question` yes?": "Based only on `passage`, is the answer to `question` not yes?",
        "Does Lucia win the bet?": "Does Lucia not win the bet?",
        "Is this shipment restricted as dangerous goods for air transport?":
            "Is this shipment not restricted as dangerous goods for air transport?",
        "Does Kenji's hand contain at least one ace?": "Does Kenji's hand not contain at least one ace?",
    }
    for q, want in cases.items():
        assert negate_question(q) == want, q


def test_negate_statement_and_refusals():
    assert negate_question("The message conveys urgency or time-sensitivity") == \
        "It is false that the message conveys urgency or time-sensitivity"
    # Already negated, or no safe insertion point: refuse rather than emit garbage.
    assert negate_question("Does the response not contain errors?") is None
    assert negate_question("Is this dish or ingredient list strictly vegan?") is None
    assert negate_question("Does frobnicating the widget help?") is None  # unknown verb, lowercase subject


def test_quotable_strips_polarity_lead():
    assert quotable("Yes, the passage supports an affirmative answer.") == "The passage supports an affirmative answer"
    assert quotable("No, it does not happen.") == "It does not happen"
    assert quotable("Contains credential harvesting indicators.") == "Contains credential harvesting indicators"


def _row(label="yes", target=None):
    r = {"state": "s", "question": "Does the response contain unsupported claims?",
         "options": [{"id": "yes", "description": "The response contains unsupported claims."},
                     {"id": "no", "description": "Every claim in the response is supported."}],
         "label": label, "source": {"name": "x"}}
    if target:
        r["target"] = target
    return r


def test_twins_flip_label_target_and_swap_sides():
    rng = random.Random(0)
    t = make_twin(_row("yes", [0.8, 0.2]), "neg_question", rng)
    assert t["label"] == "no" and t["target"] == [0.2, 0.8]
    assert t["options"][0]["description"].startswith("Every claim") and t["options"][1]["description"].startswith("The response contains")
    assert t["question"] == "Does the response not contain unsupported claims?"
    assert t["source"].startswith("polarity:neg_question:")

    for kind in ("claim_false", "claim_of_neg"):
        t = make_twin(_row("no"), kind, rng)
        assert t["label"] == "yes" and "target" not in t
        assert t["options"][0]["description"].startswith("Every claim")

    c = make_twin(_row("no", [0.1, 0.9]), "control", rng)
    assert c["label"] == "no" and c["target"] == [0.1, 0.9]
    assert c["options"][0]["description"].startswith("The response contains")


def test_generic_descriptions_stay_put_and_block_claim_kinds():
    rng = random.Random(0)
    r = _row("yes")
    r["options"] = [{"id": "yes", "description": "Yes"}, {"id": "no", "description": "No"}]
    t = make_twin(r, "neg_question", rng)
    assert t["label"] == "no" and [o["description"] for o in t["options"]] == ["Yes", "No"]
    assert make_twin(r, "claim_false", rng) is None
    assert make_twin(r, "control", rng) is None
