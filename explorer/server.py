"""Single-origin local web app, using only the Python standard library."""

import argparse
import gc
import json
import math
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .providers import DemoProvider, HTTPProvider, ProviderError, atlas, demo_neuron
from .store import Store

STATIC = Path(__file__).parent / "static"


class InputError(Exception):
    pass


def text_field(body, key, maximum=4000, required=True):
    value = body.get(key, "")
    if (
        not isinstance(value, str)
        or len(value) > maximum
        or (required and not value.strip())
    ):
        raise InputError(
            f"{key.replace('_', ' ').capitalize()} must be text between 1 and {maximum} characters."
        )
    return value.strip()


def number(body, key, default, low, high, integer=False):
    value = body.get(key, default)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not low <= value <= high
        or (integer and int(value) != value)
    ):
        raise InputError(
            f"{key.replace('_', ' ').capitalize()} must be between {low} and {high}."
        )
    return int(value) if integer else float(value)


class Application:
    def __init__(self, directory):
        self.store = Store(directory)
        self.provider_lock = threading.RLock()
        self.local = None

    def provider(self, profile):
        if profile["provider"] == "demo":
            return DemoProvider(profile)
        if profile["provider"] == "local":
            from .local import LocalProvider

            if self.local is None or self.local.profile["id"] != profile["id"]:
                self.local = None
                gc.collect()
                self.local = LocalProvider(profile)
            return self.local
        return HTTPProvider(profile)

    def profile(self, identifier):
        profile = self.store.profile(identifier)
        if profile is None:
            raise InputError(
                "Model connection not found. Choose a model from the model picker."
            )
        return profile

    def execute(self, method, path, query, body):
        if method == "GET" and path == "/api/health":
            return {"ok": True}
        if method == "GET" and path == "/api/models":
            return {"models": self.store.profiles()}
        if method == "POST" and path == "/api/models":
            provider = text_field(body, "provider", 30)
            if provider not in ("local", "ollama", "compatible"):
                raise InputError("Choose local, ollama, or compatible as the provider.")
            profile = {
                "name": text_field(body, "name", 80),
                "provider": provider,
                "model": text_field(body, "model", 300),
                "endpoint": text_field(body, "endpoint", 1000, required=False),
                "key_env": text_field(body, "key_env", 80, required=False),
            }
            if provider != "local":
                url = urlparse(profile["endpoint"])
                if (
                    url.scheme not in ("http", "https")
                    or not url.hostname
                    or url.username
                    or url.password
                    or url.query
                    or url.fragment
                ):
                    raise InputError(
                        "Endpoint must be an http(s) base URL without credentials, query parameters, or fragments."
                    )
            if profile["key_env"] and not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*", profile["key_env"]
            ):
                raise InputError(
                    "API key variable must be an environment variable name, such as LLM_API_KEY."
                )
            return self.store.save_profile(profile)
        if method == "GET" and path == "/api/experiments":
            return {"experiments": self.store.experiments()}
        if method == "DELETE" and path.startswith("/api/experiments/"):
            return {"deleted": self.store.delete_experiment(path.rsplit("/", 1)[-1])}
        if (
            method == "POST"
            and path in ("/api/chat", "/api/compare", "/api/probe", "/api/neuron")
        ) or (method == "GET" and path == "/api/workspace"):
            identifier = (
                query.get("model", ["demo"])[0]
                if method == "GET"
                else text_field(body, "model_id", 80)
            )
            profile = self.profile(identifier)
            with self.provider_lock:
                provider = self.provider(profile)
                if path == "/api/workspace":
                    info = provider.info()
                    layer_raw = query.get("layer", [""])[0]
                    try:
                        layer = int(layer_raw) if layer_raw else None
                    except ValueError as exc:
                        raise InputError("Layer must be a whole number.") from exc
                    if layer is not None and not 0 <= layer < info["layers"]:
                        raise InputError("Layer is outside this model's architecture.")
                    return {
                        "profile": profile,
                        "info": info,
                        "atlas": atlas(info, layer),
                    }
                prompt = (
                    text_field(body, "prompt", 12000) if path != "/api/neuron" else ""
                )
                max_tokens = number(body, "max_tokens", 160, 16, 512, integer=True)
                started = time.monotonic()
                if path == "/api/chat":
                    return dict(
                        provider.chat(prompt, max_tokens),
                        seconds=round(time.monotonic() - started, 2),
                    )
                layer = number(body, "layer", 8, 0, 1000, integer=True)
                neuron = number(body, "neuron", 2742, 0, 1000000, integer=True)
                info = provider.info()
                if not info["capabilities"]["inspect"]:
                    raise ProviderError(
                        "This API connection supports chat. Add a local model for neuron inspection and steering."
                    )
                if layer >= info["layers"] or neuron >= info["widths"][layer]:
                    raise InputError(
                        "The selected neuron is outside this model's architecture."
                    )
                if path == "/api/neuron":
                    if info["source"] == "synthetic":
                        return dict(
                            demo_neuron(layer, neuron),
                            score=round((math.sin(neuron * 0.17 + layer) + 1) / 2, 2),
                        )
                    return {
                        "layer": layer,
                        "neuron": neuron,
                        "source": "live",
                        "title": f"Neuron {neuron}",
                        "tokens": [],
                        "description": "Select this neuron and inspect its activation on your own text.",
                        "score": None,
                    }
                if path == "/api/probe":
                    return provider.inspect(prompt, layer, neuron)
                strength = number(body, "strength", 1.8, 0, 3)
                result = provider.compare(prompt, layer, neuron, strength, max_tokens)
                result.update(
                    model_id=identifier,
                    model_name=profile["name"],
                    prompt=prompt,
                    layer=layer,
                    neuron=neuron,
                    strength=strength,
                    seconds=round(time.monotonic() - started, 2),
                )
                result["id"] = self.store.save_experiment(result)
                return result
        raise InputError("API route not found.")


def handler(application):
    class Handler(BaseHTTPRequestHandler):
        server_version = "LLMExplore/0.1"

        def send(self, status, payload, content_type="application/json; charset=utf-8"):
            data = (
                json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
                if isinstance(payload, (dict, list))
                else payload
            )
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
            )
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def dispatch(self):
            port = self.server.server_port
            hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}
            origin = self.headers.get("Origin")
            if (
                self.headers.get("Host") not in hosts
                or origin
                and origin not in {f"http://{host}" for host in hosts}
                or self.headers.get("Sec-Fetch-Site") == "cross-site"
            ):
                return self.send(
                    403, {"error": "Only this local workspace can access the server."}
                )
            parsed = urlparse(self.path)
            try:
                if parsed.path.startswith("/api/"):
                    body = {}
                    if self.command == "POST":
                        try:
                            length = int(self.headers.get("Content-Length", "0"))
                        except ValueError as exc:
                            raise InputError("Invalid content length.") from exc
                        if not 0 < length <= 100_000:
                            return self.send(
                                413,
                                {
                                    "error": "Request must be between 1 and 100,000 bytes."
                                },
                            )
                        if self.headers.get_content_type() != "application/json":
                            return self.send(415, {"error": "Send application/json."})
                        body = json.loads(self.rfile.read(length))
                        if not isinstance(body, dict):
                            raise InputError("Request body must be a JSON object.")
                    result = application.execute(
                        self.command, parsed.path, parse_qs(parsed.query), body
                    )
                    return self.send(200, result)
                if self.command != "GET":
                    return self.send(405, {"error": "Method not allowed."})
                assets = {
                    "/": ("index.html", "text/html; charset=utf-8"),
                    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                    "/atlas3d.js": ("atlas3d.js", "text/javascript; charset=utf-8"),
                    "/style.css": ("style.css", "text/css; charset=utf-8"),
                    "/favicon.svg": ("favicon.svg", "image/svg+xml"),
                }
                asset = assets.get(parsed.path)
                if asset is None:
                    return self.send(404, {"error": "Page not found."})
                return self.send(200, (STATIC / asset[0]).read_bytes(), asset[1])
            except (InputError, ValueError, UnicodeDecodeError) as exc:
                self.send(
                    400,
                    {
                        "error": str(exc)
                        if isinstance(exc, InputError)
                        else "Invalid JSON request."
                    },
                )
            except ProviderError as exc:
                self.send(422, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 - isolate failures at the HTTP boundary
                self.log_error("Request failed: %s", type(exc).__name__)
                self.send(
                    500,
                    {
                        "error": "The operation failed. Check the server terminal, model compatibility, and available memory."
                    },
                )

        do_GET = do_POST = do_DELETE = dispatch

    return Handler


def create_server(port=8791, directory=None):
    application = Application(
        directory or os.environ.get("LLM_EXPLORE_DATA", Path.cwd() / "data")
    )
    return ThreadingHTTPServer(("127.0.0.1", port), handler(application))


def main():
    parser = argparse.ArgumentParser(description="LLM Explore local workspace")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    with create_server(args.port, args.data_dir) as server:
        print(f"LLM Explore: http://127.0.0.1:{server.server_port}", flush=True)
        print(
            "Demo is ready. Local model weights load only when you select that connection.",
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
