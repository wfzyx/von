"""Option-Marker Backend for Von.

Executes single-pass non-autoregressive decision evaluation:
- Choice: Pack all K options into 1 sequence with [MASK] markers.
- Noul: Single-pass binary verification against dual affirmative/negative options.
- Score: Single-pass ordinal rating over all levels simultaneously.
"""

import hashlib
import json
import math
import os
import threading
import warnings
from typing import Any, Dict, List, Optional, Union

import torch
from ..types import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
)
from .base import BaseBackend
from ..device import _detect_device, OpenVINODevice
from ..models.option_marker import OptionMarkerModel



def _margin_confidence(probs: List[float]) -> float:
    """TypeSafe's confidence metric: (n * p_max - 1) / (n - 1).

    Equal to the top1-minus-top2 margin only at n=2; for n>2 it stays anchored
    to the n-way chance baseline (1/n) instead of drifting down as options are
    added, which a raw top1-vs-top2 margin does even when p_max is unambiguous.
    n<=1 has no second option to be uncertain against, so it is maximally
    confident.
    """
    n = len(probs)
    if n <= 1:
        return 1.0
    p_max = max(probs)
    conf = (n * p_max - 1) / (n - 1)
    return round(max(0.0, min(1.0, conf)), 3)
def _format_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        parts = []
        for k, v in state.items():
            parts.append(f"{k}: {v}")
        return "\n".join(parts)
    return str(state)


# Public release identifier. Von reports version numbers, never architecture
# names, so callers are not coupled to how the current model is built.
# Hugging Face repo serving Von's weights. Kept as one constant because the repo
# id is a deployment detail, not a model version: the repo was renamed from
# "von-1.0" once model naming moved to versions, and the Hub redirects the old
# id, so pinned installs keep resolving.
VON_HF_REPO = "wfzyx/von"

VON_MODEL_ID = "von-1.2.0"



TEMP_SANITY_MAX = 50.0


def _validate_calibration_map(raw: object) -> Optional[Dict[str, float]]:
    """Coerce a calibration map to floats once, at load time.

    A malformed map shipped inside a weights file must not be able to raise on
    every inference call, so it is validated here and dropped wholesale if it is
    unusable. Dropping it falls back to the scalar temperature, which is always
    safe.
    """
    if not isinstance(raw, dict):
        return None
    out: Dict[str, float] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            return None
    if not any(k in out for k in ("bias", "entropy", "log_tokens", "n_options")):
        return None
    out.setdefault("lo", 0.5)
    out.setdefault("hi", 12.0)
    if out["lo"] > out["hi"]:
        return None
    # A checkpoint whose bounds would distort every confidence it reports is a
    # broken calibration file, not a preference. Clamping it silently hides
    # that; say so once at load.
    if out["lo"] <= 0 or out["hi"] > TEMP_SANITY_MAX:
        warnings.warn(
            f"calibration map bounds [{out['lo']:g}, {out['hi']:g}] are outside the "
            f"sane range (0, {TEMP_SANITY_MAX:g}]; confidences may be distorted.",
            UserWarning,
            stacklevel=2,
        )
    return out


def _validate_noul_prior(raw: object) -> Optional[Dict[str, float]]:
    """Coerce a fitted zero-shot noul prior {"a": .., "b": ..} to floats, or None.

    Absent or malformed always falls back to the original hardcoded 0.7*bias
    correction in evaluate_noul, so a bad or missing entry never breaks inference.
    """
    if not isinstance(raw, dict):
        return None
    try:
        return {"a": float(raw["a"]), "b": float(raw["b"])}
    except (KeyError, TypeError, ValueError):
        return None


class _IndependentOptionsEncoder(torch.nn.Module):
    """Flat-tensor view of the encoder for tracing in independent_options mode.

    The PyTorch path hands ModernBERT a dict of two bool 4D masks plus custom
    position_ids. Tracers want tensors, not dicts, so this takes the two masks
    as additive float tensors (0 = attend, large negative = blocked) and
    rebuilds the dict inside the graph. Same math, same outputs.
    """

    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, full_mask, sliding_mask, position_ids):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask={"full_attention": full_mask, "sliding_attention": sliding_mask},
            position_ids=position_ids,
        )
        return out.last_hidden_state


def _bool_to_additive(mask: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    # Blocked positions get the dtype's most negative finite value, the same
    # convention HF's mask utils use, so softmax sends them to exactly zero
    # without producing NaN on fully-masked rows (the eye term guarantees
    # none exist anyway).
    return torch.zeros(mask.shape, dtype=dtype).masked_fill(~mask, torch.finfo(dtype).min)


class OpenVINOEncoderWrapper(torch.nn.Module):
    """Wraps OpenVINO CompiledModels to replace the PyTorch ModernBERT encoder.

    Holds up to two graphs: the plain one (2D padding mask, default position
    ids) and the independent-options one (two 4D masks + position_ids). Which
    one runs is decided per call from the mask type, so the order-invariance
    guarantee of independent_options checkpoints is preserved on the fast path.
    """

    def __init__(self, compiled_plain=None, compiled_independent=None, config=None):
        super().__init__()
        self.compiled_plain = compiled_plain
        self.compiled_independent = compiled_independent
        # OptionMarkerModel.forward reads encoder.config.sliding_window to build
        # the order-invariant local mask; keep the HF config reachable.
        self.config = config
        self._local = threading.local()

    def _req(self, name: str):
        req = getattr(self._local, name, None)
        if req is None:
            compiled = getattr(self, name)
            if compiled is None:
                raise RuntimeError(f"OpenVINO encoder has no compiled graph for '{name}'")
            req = compiled.create_infer_request()
            setattr(self._local, name, req)
        return req

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        position_ids = kwargs.get("position_ids")
        if isinstance(attention_mask, dict):
            if position_ids is None:
                raise RuntimeError("independent_options mask dict requires position_ids")
            req = self._req("compiled_independent")
            res = req.infer({
                "input_ids": input_ids.cpu().numpy(),
                "full_mask": _bool_to_additive(attention_mask["full_attention"]).numpy(),
                "sliding_mask": _bool_to_additive(attention_mask["sliding_attention"]).numpy(),
                "position_ids": position_ids.cpu().numpy(),
            })
        else:
            if position_ids is not None:
                raise RuntimeError("plain OpenVINO graph does not accept custom position_ids")
            req = self._req("compiled_plain")
            res = req.infer({
                "input_ids": input_ids.cpu().numpy(),
                "attention_mask": attention_mask.cpu().numpy(),
            })
        last_hidden = torch.from_numpy(res[0])

        class _Output:
            pass

        out = _Output()
        out.last_hidden_state = last_hidden
        return out


def _ov_core_with_cache(target: str):
    import openvino as ov

    core = ov.Core()
    cache_dir = os.path.expanduser("~/.cache/von/openvino_cache")
    os.makedirs(cache_dir, exist_ok=True)
    try:
        core.set_property(target, {"CACHE_DIR": cache_dir})
    except Exception:
        pass
    return core, cache_dir


def _openvino_encoder_cache_key(encoder: torch.nn.Module) -> str:
    import openvino as ov

    # A release name or checkpoint path does not identify fine-tuned/replaced weights.
    digest = hashlib.sha256()
    digest.update(f"{type(encoder).__module__}.{type(encoder).__qualname__}:{torch.__version__}:{ov.__version__}".encode())
    digest.update(encoder.config.to_json_string(use_diff=False).encode())
    for name, tensor in sorted(encoder.state_dict().items()):
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        # View bytes directly: supports bfloat16 without allocating a second weights blob.
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy())
    return digest.hexdigest()


def _compile_openvino_encoder(encoder: torch.nn.Module, target: str = "GPU") -> torch.nn.Module:
    """Compiles the PyTorch ModernBERT encoder (plain mode) to OpenVINO with disk caching."""
    import openvino as ov

    core, cache_dir = _ov_core_with_cache(target)
    xml_path = os.path.join(cache_dir, f"encoder_{VON_MODEL_ID}_{_openvino_encoder_cache_key(encoder)}.xml")
    if os.path.exists(xml_path):
        compiled_model = core.compile_model(xml_path, device_name=target)
    else:
        dummy_ids = torch.ones((1, 128), dtype=torch.long)
        dummy_mask = torch.ones((1, 128), dtype=torch.long)
        ov_model = ov.convert_model(
            encoder,
            example_input={"input_ids": dummy_ids, "attention_mask": dummy_mask},
        )
        ov.save_model(ov_model, xml_path)
        compiled_model = core.compile_model(ov_model, device_name=target)

    dev_name = core.get_property(target, "FULL_DEVICE_NAME") if target in core.available_devices else target
    print(f"[von] Accelerated ModernBERT encoder on {dev_name} via OpenVINO")
    return OpenVINOEncoderWrapper(compiled_plain=compiled_model, config=encoder.config)


def _compile_openvino_independent_encoder(encoder: torch.nn.Module, target: str = "GPU") -> torch.nn.Module:
    """Compiles the encoder in independent_options mode (4D masks + position_ids).

    Shapes are fully dynamic in batch and sequence so one graph serves every
    packed length; the XML is cached next to the plain graph.
    """
    import openvino as ov

    core, cache_dir = _ov_core_with_cache(target)
    xml_path = os.path.join(cache_dir, f"encoder_indep_{VON_MODEL_ID}_{_openvino_encoder_cache_key(encoder)}.xml")
    if os.path.exists(xml_path):
        compiled_model = core.compile_model(xml_path, device_name=target)
    else:
        wrapper = _IndependentOptionsEncoder(encoder).eval()
        S = 128
        ex = {
            "input_ids": torch.ones((1, S), dtype=torch.long),
            "full_mask": torch.zeros((1, 1, S, S), dtype=torch.float32),
            "sliding_mask": torch.zeros((1, 1, S, S), dtype=torch.float32),
            "position_ids": torch.arange(S, dtype=torch.long).unsqueeze(0),
        }
        dyn = ov.Dimension(-1)
        ov_model = ov.convert_model(
            wrapper,
            example_input=ex,
            input=[
                ("input_ids", ov.PartialShape([dyn, dyn]), ov.Type.i64),
                ("full_mask", ov.PartialShape([dyn, 1, dyn, dyn]), ov.Type.f32),
                ("sliding_mask", ov.PartialShape([dyn, 1, dyn, dyn]), ov.Type.f32),
                ("position_ids", ov.PartialShape([dyn, dyn]), ov.Type.i64),
            ],
        )
        ov.save_model(ov_model, xml_path)
        compiled_model = core.compile_model(ov_model, device_name=target)

    dev_name = core.get_property(target, "FULL_DEVICE_NAME") if target in core.available_devices else target
    print(f"[von] Accelerated ModernBERT encoder (independent_options) on {dev_name} via OpenVINO")
    return OpenVINOEncoderWrapper(compiled_independent=compiled_model, config=encoder.config)

class OptionMarkerBackend(BaseBackend):
    """Native System One decision backend powered by Option-Marker joint attention."""

    # Checked in order; first directory containing option_marker.pt wins. A
    # default pointing at a directory that does not exist silently degrades to a
    # Hub download, which is how a stale cache surfaced as a load failure.
    DEFAULT_CHECKPOINT_DIRS = (
        "checkpoints/von-1.2",
        "checkpoints/von-option-marker-universal",
        "checkpoints/von-option-marker",
    )

    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        if checkpoint_dir is None:
            checkpoint_dir = next(
                (d for d in self.DEFAULT_CHECKPOINT_DIRS
                 if os.path.exists(os.path.join(d, "option_marker.pt"))),
                self.DEFAULT_CHECKPOINT_DIRS[0],
            )
        self.checkpoint_dir = checkpoint_dir
        self.raw_device = device
        resolved = _detect_device(device)
        self.resolved_device = resolved
        self.is_openvino = (
            isinstance(resolved, OpenVINODevice)
            or (isinstance(device, str) and device.lower().strip() in ("openvino", "ov", "intel", "intel_gpu", "openvino:gpu", "ov:gpu"))
        )
        if self.is_openvino:
            self.device = torch.device("cpu")
            self.openvino_target = getattr(resolved, "target", "GPU")
        else:
            self.device = resolved if isinstance(resolved, torch.device) else torch.device("cpu")
            self.openvino_target = None
        self._model = None
        self._default_temp = 1.0
        self._calib_map: Optional[dict] = None
        self._noul_prior: Optional[dict] = None
        self._independent_options = False
        self._lock = threading.Lock()

    def _effective_temperature(
        self,
        logits: "torch.Tensor",
        state_text: str,
        n_options: int,
        tokenizer,
        override: Optional[float] = None,
    ) -> float:
        """Resolve the softmax temperature for one request.

        Confidence has to track difficulty, and difficulty is not constant: an
        easy routing question the model answers 94% of the time should stay
        sharp, while a long multi-clause policy question it answers near chance
        must report near-chance confidence. A single global temperature cannot
        do both, so when a fitted map is present the temperature is a bounded
        linear function of the request's own features.

        Temperature is monotonic, so this never moves the argmax: it changes how
        sure Von claims to be, never what Von answers.
        """
        if override is not None:
            return override
        params = self._calib_map
        if not params:
            return self._default_temp

        probs = torch.softmax(logits.float(), dim=-1)
        n = max(probs.numel(), 1)
        if n > 1:
            ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum().item() / math.log(n)
        else:
            ent = 0.0

        # Must match the feature used when fitting the map (benchmarks/fit_calibration.py):
        # real tokenizer count on the state text, not a character proxy.
        tokens = max(len(tokenizer.encode(state_text, add_special_tokens=False)), 1)
        feats = {
            "bias": 1.0,
            "entropy": ent,
            "log_tokens": math.log10(tokens) / 4.0,
            "n_options": n_options / 8.0,
        }
        raw = sum(params.get(k, 0.0) * v for k, v in feats.items())
        return min(params["hi"], max(params["lo"], raw))

    def _get_model(self) -> OptionMarkerModel:
        with self._lock:
            if self._model is None:
                pt_path = os.path.join(self.checkpoint_dir, "option_marker.pt")
                loaded_from = None
                hub_calib_path = None

                if os.path.exists(pt_path):
                    # Load trained OptionMarkerModel from local checkpoint
                    model = OptionMarkerModel(base_model_id=self.checkpoint_dir)
                    state_dict = torch.load(pt_path, map_location=self.device, weights_only=True)
                    model.load_state_dict(state_dict, strict=True)
                    loaded_from = f"local file '{pt_path}'"
                else:
                    # Download from Hugging Face Hub
                    try:
                        from huggingface_hub import hf_hub_download
                        cached_pt = hf_hub_download(repo_id=VON_HF_REPO, filename="option_marker.pt")
                        model = OptionMarkerModel(base_model_id=VON_HF_REPO)
                        state_dict = torch.load(cached_pt, map_location=self.device, weights_only=True)
                        model.load_state_dict(state_dict, strict=True)
                        loaded_from = f"Hugging Face Hub '{VON_HF_REPO}:option_marker.pt' ({cached_pt})"
                        try:
                            hub_calib_path = hf_hub_download(repo_id=VON_HF_REPO, filename="marker_calibration.json")
                        except Exception:
                            hub_calib_path = None
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed to load Option-Marker decision weights: could not find local '{pt_path}' "
                            f"and failed to fetch 'option_marker.pt' from Hugging Face Hub ('{VON_HF_REPO}'). "
                            f"Refusing to run with an untrained random scoring head. Error: {exc}"
                        ) from exc

                model = model.to(self.device).eval()

                # Load fitted temperature if present: prefer local calibration file, fall back
                # to the one fetched alongside the weights from the Hub.
                calib_path = os.path.join(self.checkpoint_dir, "marker_calibration.json")
                if not os.path.exists(calib_path) and hub_calib_path:
                    calib_path = hub_calib_path
                if os.path.exists(calib_path):
                    try:
                        with open(calib_path, "r", encoding="utf-8") as f:
                            cdata = json.load(f)
                            self._default_temp = float(cdata.get("temperature", 1.0))
                            self._calib_map = _validate_calibration_map(cdata.get("calibration_map"))
                            self._noul_prior = _validate_noul_prior(cdata.get("noul_zero_shot_prior"))
                            self._independent_options = bool(cdata.get("independent_options", False))
                            model.digit_split = bool(cdata.get("digit_split", False))
                    except Exception:
                        self._default_temp = 1.0
                        self._calib_map = None
                        self._noul_prior = None
                        self._independent_options = False
                else:
                    self._default_temp = 1.0
                    self._calib_map = None
                    self._noul_prior = None
                    self._independent_options = False

                if self._calib_map:
                    print(f"[von] Loaded {VON_MODEL_ID} weights from {loaded_from} "
                          f"(input-conditioned calibration map active)")
                elif self._default_temp != 1.0:
                    print(f"[von] Loaded {VON_MODEL_ID} weights from {loaded_from} (temperature {self._default_temp})")
                else:
                    print(f"[von] Loaded {VON_MODEL_ID} weights from {loaded_from} (uncalibrated, T=1.0)")

                # OpenVINO needs a different traced graph per attention mode: the
                # plain one takes a 2D padding mask; the order-invariant one takes
                # two 4D masks plus custom position_ids. Decide only after the
                # calibration file has told us which mode this checkpoint needs.
                if self.is_openvino:
                    if self._independent_options:
                        model.encoder = _compile_openvino_independent_encoder(
                            model.encoder, target=self.openvino_target)
                    else:
                        model.encoder = _compile_openvino_encoder(model.encoder, target=self.openvino_target)

                self._model = model
            return self._model

    def evaluate_choice(
        self,
        q_id: str,
        state_text: str,
        q: Choice,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> ChoiceAnswer:
        options = list(q.criteria.keys())
        if not options:
            return ChoiceAnswer(choice="", probabilities={}, confidence=0.0)

        model = self._get_model()
        tok = model.tokenizer

        descriptions = []
        for opt in options:
            desc = q.criteria.get(opt)
            descriptions.append(desc.strip() if desc else opt.strip())

        packed_text = model.pack_sequence(state_text, q.instructions, descriptions)

        inputs = tok(packed_text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"][0]
        pos_list = (input_ids == model.mask_token_id).nonzero(as_tuple=True)[0].tolist()

        with torch.no_grad():
            batch_logits = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                mask_positions=[pos_list],
                independent_options=self._independent_options,
            )
            logits = batch_logits[0]  # (K,)
            eff_temp = self._effective_temperature(
                logits, state_text, len(options), tok, temperature
            )
            scaled = logits / max(eff_temp, 1e-4)
            probs = torch.softmax(scaled, dim=-1).cpu().tolist()

        best_idx = torch.argmax(logits).item()
        best_choice = options[best_idx]
        prob_dict = {opt: round(p, 4) for opt, p in zip(options, probs)}

        conf = _margin_confidence(probs)

        return ChoiceAnswer(choice=best_choice, probabilities=prob_dict, confidence=conf)

    def evaluate_noul(
        self,
        q_id: str,
        state_text: str,
        q: Noul,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> NoulAnswer:
        model = self._get_model()
        tok = model.tokenizer

        crit = q.criteria or {}
        pos_desc = crit.get("true")
        neg_desc = crit.get("false")

        has_explicit = bool(pos_desc or neg_desc)
        if not pos_desc:
            pos_desc = "Yes, condition holds true."
        if not neg_desc:
            neg_desc = "No, condition is false."

        descriptions = [pos_desc, neg_desc]
        packed_text = model.pack_sequence(state_text, q.instructions, descriptions)

        inputs = tok(packed_text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"][0]
        pos_list = (input_ids == model.mask_token_id).nonzero(as_tuple=True)[0].tolist()

        with torch.no_grad():
            batch_logits = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                mask_positions=[pos_list],
                independent_options=self._independent_options,
            )
            logits = batch_logits[0]

            # In zero-shot Noul without explicit criteria, cancel out the intrinsic negative polarity prior
            if not has_explicit:
                null_packed = model.pack_sequence("", q.instructions, descriptions)
                null_inputs = tok(null_packed, return_tensors="pt").to(self.device)
                null_pos = (null_inputs["input_ids"][0] == model.mask_token_id).nonzero(as_tuple=True)[0].tolist()
                null_logits = model(
                    input_ids=null_inputs["input_ids"],
                    attention_mask=null_inputs["attention_mask"],
                    mask_positions=[null_pos],
                    independent_options=self._independent_options,
                )[0]
                # Zero-shot debiasing. The context-free bias is positive on nearly
                # every task the model has no criteria for (the model prefers "yes"
                # with no state at all), so subtracting a coefficient times it pulls
                # predictions toward "no". A fitted (a, b) replaces the original flat
                # 0.7 coefficient when the checkpoint ships one (see
                # benchmarks/noul_prior_fit.py); falls back to the original behaviour
                # exactly when absent, so an unfitted checkpoint is unaffected.
                bias = null_logits[0] - null_logits[1]
                if self._noul_prior is not None:
                    correction = self._noul_prior["a"] * bias + self._noul_prior["b"]
                else:
                    correction = 0.7 * bias
                logits = torch.stack([logits[0] - correction, logits[1]])

            eff_temp = self._effective_temperature(
                logits, state_text, 2, tok, temperature
            )
            scaled = logits / max(eff_temp, 1e-4)
            probs = torch.softmax(scaled, dim=-1).cpu().tolist()

        prob_true = round(max(0.0, min(1.0, probs[0])), 4)
        return NoulAnswer(noul=prob_true)

    def evaluate_score(
        self,
        q_id: str,
        state_text: str,
        q: Score,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> ScoreAnswer:
        levels = q.criteria
        if not levels:
            return ScoreAnswer(score=0.0, confidence=0.0, legend={}, probabilities={})

        model = self._get_model()
        tok = model.tokenizer

        legend: Dict[str, str] = {}
        descriptions = []
        inst_clean = q.instructions.strip() if q.instructions else ""

        for i, item in enumerate(levels):
            idx_str = str(i)
            if isinstance(item, dict):
                what = item.get("what", "")
                examples = item.get("examples", [])
                ex_str = f" Examples: {', '.join(examples)}" if examples else ""
                desc = f"{what}{ex_str}".strip()
            else:
                desc = str(item).strip()
            legend[idx_str] = desc
            descriptions.append(desc)

        packed_text = model.pack_sequence(state_text, q.instructions, descriptions)

        inputs = tok(packed_text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"][0]
        pos_list = (input_ids == model.mask_token_id).nonzero(as_tuple=True)[0].tolist()

        with torch.no_grad():
            batch_logits = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                mask_positions=[pos_list],
                independent_options=self._independent_options,
            )
            logits = batch_logits[0]
            eff_temp = self._effective_temperature(
                logits, state_text, len(descriptions), tok, temperature
            )
            scaled = logits / max(eff_temp, 1e-4)
            probs = torch.softmax(scaled, dim=-1).cpu().tolist()

        prob_dict = {str(i): round(p, 4) for i, p in enumerate(probs)}
        weighted_score = round(sum(i * p for i, p in enumerate(probs)), 2)

        conf = _margin_confidence(probs)

        return ScoreAnswer(
            score=weighted_score,
            confidence=conf,
            legend=legend,
            probabilities=prob_dict,
        )

    def evaluate(
        self,
        state: Any,
        questions: Dict[str, Union[Question, Dict[str, Any]]],
        model: str = VON_MODEL_ID,
    ) -> SystemOneResponse:
        state_str = _format_state(state)
        answers: Dict[str, Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]] = {}
        total_q_chars = 0

        for q_id, q_data in questions.items():
            if isinstance(q_data, dict):
                q_type = q_data.get("type", "choice")
                if q_type == "choice":
                    q_obj = Choice(**q_data)
                elif q_type == "noul":
                    q_obj = Noul(**q_data)
                elif q_type == "score":
                    q_obj = Score(**q_data)
                else:
                    raise ValueError(f"Unknown question type '{q_type}'")
            else:
                q_obj = q_data

            if isinstance(q_obj, Choice):
                answers[q_id] = self.evaluate_choice(q_id, state_str, q_obj)
                total_q_chars += len(q_obj.instructions or "")
            elif isinstance(q_obj, Noul):
                answers[q_id] = self.evaluate_noul(q_id, state_str, q_obj)
                total_q_chars += len(q_obj.instructions or "")
            elif isinstance(q_obj, Score):
                answers[q_id] = self.evaluate_score(q_id, state_str, q_obj)
                total_q_chars += len(q_obj.instructions or "")

        resolved_model = model or VON_MODEL_ID
        state_tokens = max(1, len(state_str) // 4)
        q_tokens = max(1, total_q_chars // 4)

        usage = Usage(
            input_tokens=state_tokens + q_tokens,
            output_tokens=len(answers),
        )

        return SystemOneResponse(
            model=resolved_model,
            answers=answers,
            usage=usage,
        )
