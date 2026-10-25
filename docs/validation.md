# Validation — October 25, 2026

## Automated

`python -m unittest discover -s tests -v`: **12 tests passed** with Torch 2.14.1 CPU and Transformers 5.20.1 installed. The suite validates:

- HTTP routes, input bounds, origin/host rejection, and a static asset allowlist.
- Synthetic activation inspection, unchanged gain, suppression, amplification, and automatic persistence across application instances.
- Provider capability isolation and independent model profiles.
- Ollama and compatible chat request/response protocols against a local fake server.
- Architecture discovery and layer-specific neuron bounds.
- Hook cleanup after a raised exception and exact scaling of one tensor coordinate.
- Real forward passes, activation capture, baseline/steered generation, gain-1 equality, gain-0 execution, and hook removal on **tiny randomly initialized Llama, GPT-2, and Qwen3.5 multimodal models**. No pretrained weights or external datasets were downloaded for these tests.

The dependency-free environment also passes the suite, with the two optional local inference tests skipped. `uv sync --frozen` builds and installs the core package. Both browser JavaScript modules pass `node --check`.

## Browser

Checked the desktop workspace at 1,440px: 3D layer rendering, keyboard neuron selection, zoom, direct neuron selection outside the sample, list/search, signed activation highlights, comparison generation and saving, playground generation, model connection creation, API-only capability display, and switching back to the 3D demo. The temporary API connection was removed after verification.

The visual smoke checks use the explicit synthetic demo. Pretrained full-size model outputs, GPU behavior, every advertised structural model family, and a real external inference service have not been validated. The 3D view samples neurons by index and displays no measured circuit edges.
