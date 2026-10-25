"""Lazy local inference with architecture discovery and temporary neuron hooks.

The hook changes the input of the FFN output projection. This works for dense
gated and ungated MLPs without assuming a tokenizer, layer count, or FFN width.
Mixture-of-experts projections are intentionally not resolved as dense layers.
"""

import threading
from contextlib import contextmanager
from typing import ClassVar

from .providers import Provider, ProviderError

LAYER_PATHS = (
    "model.layers",
    "model.language_model.layers",
    "language_model.model.layers",
    "language_model.layers",
    "transformer.h",
    "model.decoder.layers",
    "gpt_neox.layers",
)
PROJECTION_PATHS = (
    "mlp.down_proj",
    "mlp.fc2",
    "mlp.dense_4h_to_h",
    "mlp.c_proj",
    "fc2",
)


def resolve(obj, path):
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def discover(model):
    layers = next(
        (value for path in LAYER_PATHS if (value := resolve(model, path)) is not None),
        None,
    )
    if layers is None or len(layers) == 0:
        raise ProviderError(
            "No supported decoder layer stack found. Add its path to LAYER_PATHS in explorer/local.py."
        )
    projections, widths = [], []
    for layer in layers:
        projection = next(
            (
                value
                for path in PROJECTION_PATHS
                if (value := resolve(layer, path)) is not None
            ),
            None,
        )
        if projection is None or not hasattr(projection, "weight"):
            raise ProviderError(
                "This architecture does not expose a supported dense MLP output projection. MoE models need a dedicated adapter."
            )
        # Hugging Face Conv1D stores [in, out]; torch Linear stores [out, in].
        width = getattr(projection, "in_features", None)
        if width is None and type(projection).__name__ == "Conv1D":
            width = projection.weight.shape[0]
        if width is None:
            raise ProviderError(
                "Cannot determine the neuron width of this output projection."
            )
        projections.append(projection)
        widths.append(int(width))
    return projections, widths


@contextmanager
def neuron_hook(projection, neuron, strength=None, capture=None):
    def hook(module, inputs):
        hidden = inputs[0]
        if capture is not None:
            capture.append(hidden[0, :, neuron].detach().float().cpu().tolist())
        if strength is not None:
            changed = hidden.clone()
            changed[..., neuron] *= strength
            return (changed, *inputs[1:])
        return None

    handle = projection.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


class LocalProvider(Provider):
    capabilities: ClassVar[dict[str, bool]] = {
        "chat": True,
        "inspect": True,
        "steer": True,
    }

    def __init__(self, profile):
        super().__init__(profile)
        self.lock = threading.RLock()
        self.model = None

    def load(self):
        if self.model is not None:
            return
        try:
            import torch
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise ProviderError(
                'Install local inference first: uv sync --extra local (or pip install -e ".[local]").'
            ) from exc
        try:
            config = AutoConfig.from_pretrained(
                self.profile["model"], trust_remote_code=False
            )
            cls = (
                AutoModelForImageTextToText
                if getattr(config, "text_config", None) is not None
                else AutoModelForCausalLM
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.profile["model"], trust_remote_code=False
            )
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
            model = (
                cls.from_pretrained(
                    self.profile["model"], dtype="auto", trust_remote_code=False
                )
                .to(device)
                .eval()
            )
            projections, widths = discover(model)
            self.torch, self.tokenizer, self.device = torch, tokenizer, device
            self.projections, self.widths = projections, widths
            self.model = model
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"Could not load this local model ({type(exc).__name__}). Check its ID, access permissions, available memory, and installed Transformers version."
            ) from exc

    def info(self):
        with self.lock:
            self.load()
            return dict(
                super().info(),
                layers=len(self.widths),
                widths=self.widths,
                architecture=type(self.model).__name__,
                status=f"Loaded on {self.device}",
            )

    def validate_neuron(self, layer, neuron):
        if not 0 <= layer < len(self.widths) or not 0 <= neuron < self.widths[layer]:
            raise ProviderError(
                "The selected neuron is outside this model's architecture."
            )

    def encode(self, text, chat=False):
        if chat and getattr(self.tokenizer, "chat_template", None):
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = self.tokenizer(
                text, return_tensors="pt", add_special_tokens=False
            )
        else:
            encoded = self.tokenizer(text, return_tensors="pt")
        if encoded["input_ids"].shape[-1] == 0:
            raise ProviderError(
                "The tokenizer produced no input tokens. Check that this model directory contains its matching tokenizer files."
            )
        # A fixed ceiling bounds activation capture memory even on long-context models.
        limit = min(2048, getattr(self.model.config, "max_position_embeddings", 2048))
        if encoded["input_ids"].shape[-1] > limit:
            raise ProviderError(
                f"This prompt exceeds the workspace limit of {limit} input tokens. Shorten it and try again."
            )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def inspect(self, text, layer, neuron):
        with self.lock:
            self.load()
            self.validate_neuron(layer, neuron)
            encoded = self.encode(text)
            values = []
            with (
                neuron_hook(self.projections[layer], neuron, capture=values),
                self.torch.inference_mode(),
            ):
                self.model(**encoded, use_cache=False)
            ids = encoded["input_ids"][0].tolist()
            tokens = [
                {
                    "text": self.tokenizer.decode([token]),
                    "value": round(float(value), 5),
                }
                for token, value in zip(ids, values[0])
            ]
            return {
                "tokens": tokens,
                "source": "live",
                "layer": layer,
                "neuron": neuron,
            }

    def generate(self, encoded, max_tokens):
        with self.torch.inference_mode():
            output = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            output[0, encoded["input_ids"].shape[-1] :], skip_special_tokens=True
        )

    def chat(self, prompt, max_tokens):
        with self.lock:
            self.load()
            return {
                "text": self.generate(self.encode(prompt, chat=True), max_tokens),
                "source": "live",
            }

    def compare(self, prompt, layer, neuron, strength, max_tokens):
        with self.lock:
            self.load()
            self.validate_neuron(layer, neuron)
            encoded = self.encode(prompt, chat=True)
            baseline = self.generate(encoded, max_tokens)
            with neuron_hook(self.projections[layer], neuron, strength=strength):
                steered = self.generate(encoded, max_tokens)
            return {
                "baseline": baseline,
                "steered": steered,
                "source": "live",
                "note": "Greedy decoding with identical prompts. Gain scales this neuron's activation at every prompt and generated position.",
            }
