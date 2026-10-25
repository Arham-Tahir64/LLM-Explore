"""Provider contract and lightweight demo / HTTP implementations.

Only local weight access can provide neuron inspection. HTTP providers never
advertise steering or return synthetic neuron data.
"""

import hashlib
import json
import math
import os
import re
from typing import ClassVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ProviderError(Exception):
    pass


class Provider:
    capabilities: ClassVar[dict[str, bool]] = {
        "chat": True,
        "inspect": False,
        "steer": False,
    }

    def __init__(self, profile):
        self.profile = profile

    def info(self):
        return {
            "capabilities": self.capabilities,
            "layers": 0,
            "widths": [],
            "source": "live",
            "status": "Available on request",
        }

    def inspect(self, text, layer, neuron):
        raise ProviderError(
            "This connection does not expose neuron activations. Add a local Hugging Face model to inspect weights."
        )

    def compare(self, prompt, layer, neuron, strength, max_tokens):
        raise ProviderError(
            "Steering requires a local model with access to its MLP activations."
        )


THEMES = [
    (
        "Spatial relationships",
        ["above", "beside", "through"],
        "Spatial language and descriptions of where things are.",
    ),
    (
        "Code & structure",
        ["function", "return", "class"],
        "Programming syntax and structured, step-by-step explanations.",
    ),
    (
        "Questions & curiosity",
        ["why", "how", "wonder"],
        "Questions, uncertainty, and requests for an explanation.",
    ),
    (
        "Warmth & gratitude",
        ["thank", "happy", "kind"],
        "Appreciation and an encouraging conversational tone.",
    ),
    (
        "Nature & landscapes",
        ["forest", "river", "mountain"],
        "Descriptions of the natural world and outdoor settings.",
    ),
    (
        "Extraterrestrial concepts",
        ["alien", "planet", "cosmic"],
        "Space, imagined worlds, and extraterrestrial identities.",
    ),
]


def demo_neuron(layer, neuron):
    theme = THEMES[5 if neuron == 2742 else (layer * 3 + neuron) % len(THEMES)]
    return {
        "layer": layer,
        "neuron": neuron,
        "title": theme[0],
        "tokens": theme[1],
        "description": theme[2],
        "source": "synthetic",
    }


class DemoProvider(Provider):
    capabilities: ClassVar[dict[str, bool]] = {
        "chat": True,
        "inspect": True,
        "steer": True,
    }

    def info(self):
        return dict(
            super().info(),
            layers=12,
            widths=[4096] * 12,
            source="synthetic",
            status="Demo workspace",
            architecture="Illustrative dense MLP",
        )

    def inspect(self, text, layer, neuron):
        tokens = re.findall(r"\s*\S+", text)
        theme = demo_neuron(layer, neuron)
        values = []
        for i, token in enumerate(tokens):
            digest = hashlib.sha256(f"{layer}:{neuron}:{i}:{token}".encode()).digest()
            value = (digest[0] / 255 - 0.25) * 1.4
            if any(w in token.lower() for w in theme["tokens"]):
                value += 3.5
            values.append({"text": token, "value": round(value, 3)})
        return {
            "tokens": values,
            "source": "synthetic",
            "layer": layer,
            "neuron": neuron,
        }

    def chat(self, prompt, max_tokens):
        return {
            "text": f"This is a simulated response to “{prompt}”. Connect a local model or an API in Models to generate a real answer.",
            "source": "synthetic",
        }

    def compare(self, prompt, layer, neuron, strength, max_tokens):
        theme = demo_neuron(layer, neuron)
        baseline = f"An illustrative, neutral response to “{prompt}”."
        steered = (
            baseline
            if strength == 1
            else f"An illustrative response to “{prompt}”, with {theme['title'].lower()} {'reduced' if strength < 1 else 'emphasized'}. Example vocabulary: {', '.join(theme['tokens'])}."
        )
        return {
            "baseline": baseline,
            "steered": steered,
            "source": "synthetic",
            "note": "Synthetic examples demonstrate the workflow. They are not measured model outputs.",
        }


class HTTPProvider(Provider):
    def _request(self, path, payload):
        endpoint = self.profile["endpoint"].rstrip("/") + path
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        env_name = self.profile.get("key_env", "")
        if env_name:
            key = os.environ.get(env_name)
            if not key:
                raise ProviderError(
                    f"Set the {env_name} environment variable and restart the server."
                )
            headers["Authorization"] = f"Bearer {key}"
        request = Request(
            endpoint, json.dumps(payload).encode(), headers, method="POST"
        )
        try:
            with urlopen(request, timeout=180) as response:
                raw = response.read(4_000_001)
                if len(raw) > 4_000_000:
                    raise ProviderError("The model response was too large.")
                return json.loads(raw)
        except HTTPError as exc:
            raise ProviderError(
                f"The model endpoint returned HTTP {exc.code}. Check the URL, model ID, and API key environment variable."
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ProviderError(
                "Could not reach the model endpoint. Check that the service is running and the URL is correct."
            ) from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderError(
                "The endpoint did not return a supported JSON response."
            ) from exc

    def chat(self, prompt, max_tokens):
        payload = {
            "model": self.profile["model"],
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if self.profile["provider"] == "ollama":
            payload["options"] = {"num_predict": max_tokens, "temperature": 0}
            data = self._request("/api/chat", payload)
            result = data.get("message", {}).get("content")
        else:
            payload.update(max_tokens=max_tokens, temperature=0)
            data = self._request("/chat/completions", payload)
            choices = data.get("choices") or []
            result = choices[0].get("message", {}).get("content") if choices else None
        if not isinstance(result, str) or not result.strip():
            raise ProviderError(
                "The endpoint returned no text. Confirm it supports the selected chat API."
            )
        return {"text": result, "source": "live"}


def atlas(info, layer=None):
    """A bounded, index-based sample. Coordinates do not imply semantic similarity."""
    total = info["layers"]
    if layer is not None:
        indices = [layer]
    elif total > 1:
        indices = sorted(
            {round(i * (total - 1) / min(total - 1, 11)) for i in range(min(total, 12))}
        )
    else:
        indices = [0] if total else []
    nodes = []
    for layer_index in indices:
        width = info["widths"][layer_index]
        neurons = sorted(
            set(
                [round(i * (width - 1) / 17) for i in range(18)]
                + ([2742] if width > 2742 else [])
            )
        )
        for neuron in neurons:
            node = {"layer": layer_index, "neuron": neuron}
            if info["source"] == "synthetic":
                node.update(
                    demo_neuron(layer_index, neuron),
                    score=round((math.sin(neuron * 0.17 + layer_index) + 1) / 2, 2),
                )
            else:
                node.update(
                    title=f"Neuron {neuron}",
                    tokens=[],
                    description="Select this neuron and inspect its activation on your own text.",
                    score=None,
                    source="live",
                )
            nodes.append(node)
    return {
        "nodes": nodes,
        "layers": indices,
        "source": info["source"],
        "layout": "index",
        "sampled": True,
    }
