"""Exercise cached exports with tiny encoders, without downloading weights."""

from unittest.mock import patch

import pytest
import torch
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput

from von.backends import option_marker_backend as backend

ov = pytest.importorskip("openvino")


class TinyEncoder(torch.nn.Module):
    def __init__(self, weight, scale=1.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([weight], dtype=torch.float32))
        self.config = PretrainedConfig(scale=scale)

    def forward(self, input_ids, attention_mask, position_ids=None):
        if isinstance(attention_mask, dict):
            mask = attention_mask["full_attention"].sum(-1).squeeze(1)
            mask = mask + attention_mask["sliding_attention"].sum(-1).squeeze(1)
            values = input_ids.float() + position_ids.float() + mask
        else:
            values = input_ids.float() + attention_mask.float()
        return BaseModelOutput(last_hidden_state=(values * self.weight * self.config.scale).unsqueeze(-1))


@pytest.mark.parametrize("independent", [False, True])
def test_cached_encoder_tracks_weights_and_config(tmp_path, monkeypatch, independent):
    core = ov.Core()
    core.set_property("CPU", {"CACHE_DIR": str(tmp_path)})
    monkeypatch.setattr(backend, "_ov_core_with_cache", lambda target: (core, str(tmp_path)))
    compile_encoder = (backend._compile_openvino_independent_encoder if independent
                       else backend._compile_openvino_encoder)
    inputs = {"input_ids": torch.ones((1, 128), dtype=torch.long)}
    if independent:
        mask = torch.ones((1, 1, 128, 128), dtype=torch.bool)
        inputs.update(attention_mask={"full_attention": mask, "sliding_attention": mask},
                      position_ids=torch.arange(128).unsqueeze(0))
    else:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    def check(encoder):
        compiled = compile_encoder(encoder, target="CPU")
        with torch.no_grad():
            expected_inputs = dict(inputs)
            if independent:
                expected_inputs["attention_mask"] = {
                    name: backend._bool_to_additive(mask)
                    for name, mask in inputs["attention_mask"].items()
                }
            expected = encoder(**expected_inputs).last_hidden_state
            torch.testing.assert_close(compiled(**inputs).last_hidden_state, expected)

    with patch.object(ov, "convert_model", wraps=ov.convert_model) as convert:
        check(TinyEncoder(2.0).eval())
        check(TinyEncoder(2.0).eval())
        assert convert.call_count == 1, "identical encoders should reuse the export"

        encoder = TinyEncoder(3.0).eval()
        check(encoder)
        assert convert.call_count == 2, "different checkpoint weights need a new export"

        with torch.no_grad():
            encoder.weight.fill_(4.0)
        check(encoder)
        assert convert.call_count == 3, "replacing weights in place must invalidate the export"

        encoder.config.scale = 2.0
        check(encoder)
        assert convert.call_count == 4, "configuration changes must invalidate the export"
