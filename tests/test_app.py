import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from explorer.local import discover, neuron_hook
from explorer.providers import HTTPProvider, ProviderError, atlas
from explorer.server import Application, create_server


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.server = create_server(0, cls.temp.name)
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.temp.cleanup()

    def request(self, path, body=None, headers=None, method=None):
        data = json.dumps(body).encode() if body is not None else None
        request = Request(
            self.url + path,
            data,
            {"Content-Type": "application/json", **(headers or {})},
            method=method,
        )
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            with error:
                return error.code, json.load(error)

    def test_demo_inspect_compare_and_persistence(self):
        status, workspace = self.request("/api/workspace")
        self.assertEqual(status, 200)
        self.assertEqual(workspace["info"]["source"], "synthetic")
        self.assertTrue(workspace["info"]["capabilities"]["steer"])
        self.assertEqual(len(workspace["atlas"]["layers"]), 12)
        status, custom = self.request(
            "/api/neuron", {"model_id": "demo", "layer": 3, "neuron": 1234}
        )
        self.assertEqual(status, 200)
        self.assertEqual((custom["layer"], custom["neuron"]), (3, 1234))
        payload = {
            "model_id": "demo",
            "layer": 8,
            "neuron": 2742,
            "prompt": "alien planet",
            "strength": 1,
        }
        status, probe = self.request("/api/probe", payload)
        self.assertEqual(status, 200)
        self.assertEqual(len(probe["tokens"]), 2)
        status, result = self.request("/api/compare", payload)
        self.assertEqual(status, 200)
        self.assertEqual(result["baseline"], result["steered"])
        self.assertEqual(result["source"], "synthetic")
        other = Application(self.temp.name)
        saved = next(
            row for row in other.store.experiments() if row["id"] == result["id"]
        )
        self.assertEqual(saved["model_id"], "demo")
        self.assertEqual(saved["prompt"], "alien planet")
        status, _ = self.request("/api/experiments/" + result["id"], method="DELETE")
        self.assertEqual(status, 200)
        self.assertFalse(
            any(row["id"] == result["id"] for row in other.store.experiments())
        )
        for strength in (0, 1.8):
            status, steered = self.request(
                "/api/compare", dict(payload, strength=strength)
            )
            self.assertEqual(status, 200)
            self.assertNotEqual(steered["baseline"], steered["steered"])
            self.assertIn(
                "reduced" if strength == 0 else "emphasized", steered["steered"]
            )
            self.request("/api/experiments/" + steered["id"], method="DELETE")

    def test_provider_capabilities_and_isolation(self):
        profile = {
            "name": "Local API",
            "provider": "ollama",
            "model": "test-model",
            "endpoint": "http://localhost:11434",
            "key_env": "",
        }
        status, saved = self.request("/api/models", profile)
        self.assertEqual(status, 200)
        self.assertNotEqual(saved["id"], "demo")
        status, workspace = self.request("/api/workspace?model=" + saved["id"])
        self.assertEqual(workspace["atlas"]["nodes"], [])
        self.assertFalse(workspace["info"]["capabilities"]["steer"])
        status, result = self.request(
            "/api/compare", {"model_id": saved["id"], "prompt": "test"}
        )
        self.assertEqual(status, 422)
        self.assertIn("chat", result["error"])

    def test_invalid_inputs(self):
        for field, value in [
            ("strength", float("nan")),
            ("strength", 4),
            ("layer", -1),
            ("layer", True),
            ("max_tokens", 900),
            ("neuron", 4096),
        ]:
            status, _ = self.request(
                "/api/compare", {"model_id": "demo", "prompt": "test", field: value}
            )
            self.assertEqual(status, 400, (field, value))
        self.assertEqual(self.request("/api/workspace?layer=200")[0], 400)
        self.assertEqual(self.request("/api/workspace?layer=invalid")[0], 400)
        self.assertEqual(self.request("/api/chat", ["not", "an", "object"])[0], 400)
        self.assertEqual(
            self.request("/api/chat", {"model_id": "missing", "prompt": "test"})[0], 400
        )

    def test_origin_host_and_static_allowlist(self):
        self.assertEqual(
            self.request("/api/models", headers={"Origin": "https://example.com"})[0],
            403,
        )
        self.assertEqual(
            self.request("/api/models", headers={"Host": "attacker.example"})[0], 403
        )
        self.assertEqual(
            self.request("/api/models", headers={"Sec-Fetch-Site": "cross-site"})[0],
            403,
        )
        self.assertEqual(
            self.request("/api/models", headers={"Origin": self.url})[0], 200
        )
        self.assertEqual(self.request("/../pyproject.toml")[0], 404)
        with urlopen(self.url) as response:
            self.assertIn(b"Look a little closer.", response.read())
            self.assertIn(
                "frame-ancestors 'none'", response.headers["Content-Security-Policy"]
            )

    def test_profile_validation(self):
        for endpoint in [
            "file:///tmp/test",
            "http://user:key@localhost:8000",
            "http://localhost:8000?key=secret",
        ]:
            self.assertEqual(
                self.request(
                    "/api/models",
                    {
                        "name": "test",
                        "provider": "compatible",
                        "model": "x",
                        "endpoint": endpoint,
                    },
                )[0],
                400,
            )
        self.assertEqual(
            self.request(
                "/api/models",
                {
                    "name": "test",
                    "provider": "compatible",
                    "model": "x",
                    "endpoint": "http://localhost:8000",
                    "key_env": "actual-secret-value",
                },
            )[0],
            400,
        )


class FakeEndpoint(BaseHTTPRequestHandler):
    last_request = None

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).last_request = (self.path, data, self.headers.get("Authorization"))
        result = (
            {"message": {"content": "ollama reply"}}
            if self.path == "/api/chat"
            else {"choices": [{"message": {"content": "compatible reply"}}]}
        )
        raw = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


class ProviderTests(unittest.TestCase):
    def test_http_provider_protocols(self):
        with ThreadingHTTPServer(("127.0.0.1", 0), FakeEndpoint) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                for provider, endpoint, path, reply in [
                    ("ollama", base, "/api/chat", "ollama reply"),
                    (
                        "compatible",
                        base + "/v1",
                        "/v1/chat/completions",
                        "compatible reply",
                    ),
                ]:
                    client = HTTPProvider(
                        {
                            "provider": provider,
                            "endpoint": endpoint,
                            "model": "different-model",
                            "key_env": "",
                        }
                    )
                    self.assertEqual(client.chat("hello", 80)["text"], reply)
                    request_path, body, _ = FakeEndpoint.last_request
                    self.assertEqual(request_path, path)
                    self.assertEqual(body["model"], "different-model")
                    self.assertFalse(body["stream"])
                    self.assertEqual(body["messages"][0]["content"], "hello")
                    self.assertEqual(
                        body.get(
                            "max_tokens", body.get("options", {}).get("num_predict")
                        ),
                        80,
                    )
            finally:
                server.shutdown()

    def test_missing_key_does_not_send(self):
        os.environ.pop("LLM_EXPLORE_NONEXISTENT_TEST_KEY", None)
        provider = HTTPProvider(
            {
                "endpoint": "http://127.0.0.1:1",
                "provider": "compatible",
                "model": "x",
                "key_env": "LLM_EXPLORE_NONEXISTENT_TEST_KEY",
            }
        )
        with self.assertRaisesRegex(ProviderError, "environment variable"):
            provider.chat("hello", 80)

    def test_architecture_discovery(self):
        projection = SimpleNamespace(in_features=123, weight=object())
        model = SimpleNamespace(
            model=SimpleNamespace(
                language_model=SimpleNamespace(
                    layers=[SimpleNamespace(mlp=SimpleNamespace(down_proj=projection))]
                )
            )
        )
        projections, widths = discover(model)
        self.assertEqual(projections, [projection])
        self.assertEqual(widths, [123])
        with self.assertRaises(ProviderError):
            discover(
                SimpleNamespace(
                    model=SimpleNamespace(
                        layers=[SimpleNamespace(block_sparse_moe=object())]
                    )
                )
            )

    def test_atlas_uses_actual_layer_widths(self):
        info = {"layers": 40, "widths": [64 + i for i in range(40)], "source": "live"}
        result = atlas(info)
        self.assertEqual(len(result["layers"]), 12)
        self.assertEqual(result["layers"][0], 0)
        self.assertEqual(result["layers"][-1], 39)
        for node in result["nodes"]:
            self.assertLess(node["neuron"], info["widths"][node["layer"]])
            self.assertIsNone(node["score"])
            self.assertEqual(node["tokens"], [])
        self.assertEqual(atlas(info, 31)["layers"], [31])

    def test_hook_cleanup_on_failure(self):
        class Projection:
            removed = False

            def register_forward_pre_hook(self, hook):
                return SimpleNamespace(remove=lambda: setattr(self, "removed", True))

        projection = Projection()
        with self.assertRaises(RuntimeError), neuron_hook(projection, 2, strength=2):
            raise RuntimeError("forward failed")
        self.assertTrue(projection.removed)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "Optional Torch is not installed")
class TorchHookTests(unittest.TestCase):
    def test_only_selected_neuron_changes_and_hook_is_removed(self):
        projection = torch.nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            projection.weight.copy_(torch.eye(3))
        value = torch.tensor([[[2.0, 3.0, 4.0]]])
        captured = []
        with neuron_hook(projection, 1, strength=2, capture=captured):
            self.assertTrue(
                torch.equal(projection(value), torch.tensor([[[2.0, 6.0, 4.0]]]))
            )
        self.assertEqual(captured, [[3.0]])
        self.assertTrue(torch.equal(projection(value), value))
        self.assertTrue(torch.equal(value, torch.tensor([[[2.0, 3.0, 4.0]]])))


@unittest.skipUnless(
    torch is not None, "Optional local inference dependencies are not installed"
)
class LocalIntegrationTests(unittest.TestCase):
    def test_tiny_models_use_real_activations_and_identity_gain(self):
        try:
            from tokenizers import Tokenizer
            from tokenizers.decoders import ByteLevel as ByteLevelDecoder
            from tokenizers.models import BPE
            from tokenizers.pre_tokenizers import ByteLevel
            from tokenizers.trainers import BpeTrainer
            from transformers import (
                GPT2Config,
                GPT2LMHeadModel,
                LlamaConfig,
                LlamaForCausalLM,
                PreTrainedTokenizerFast,
                Qwen3_5Config,
                Qwen3_5ForConditionalGeneration,
                Qwen3_5TextConfig,
                Qwen3_5VisionConfig,
            )
        except ImportError:
            self.skipTest("Transformers and tokenizers are required")
        from explorer.local import LocalProvider

        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        tokenizer.train_from_iterator(
            ["hello world a model test the"] * 5,
            BpeTrainer(vocab_size=32, special_tokens=["<unk>", "<eos>"]),
        )
        fast = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="<unk>",
            eos_token="<eos>",
            pad_token="<eos>",
        )
        cases = [
            LlamaForCausalLM(
                LlamaConfig(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    max_position_embeddings=128,
                    eos_token_id=1,
                    pad_token_id=1,
                )
            ),
            GPT2LMHeadModel(
                GPT2Config(
                    vocab_size=32,
                    n_embd=16,
                    n_layer=2,
                    n_head=2,
                    n_positions=128,
                    bos_token_id=1,
                    eos_token_id=1,
                    pad_token_id=1,
                )
            ),
            Qwen3_5ForConditionalGeneration(
                Qwen3_5Config(
                    text_config=Qwen3_5TextConfig(
                        vocab_size=32,
                        hidden_size=16,
                        intermediate_size=32,
                        num_hidden_layers=2,
                        num_attention_heads=2,
                        num_key_value_heads=2,
                        head_dim=8,
                        layer_types=["full_attention", "full_attention"],
                        max_position_embeddings=128,
                        eos_token_id=1,
                        pad_token_id=1,
                        rope_parameters={
                            "rope_type": "default",
                            "rope_theta": 10000.0,
                            "partial_rotary_factor": 1.0,
                            "mrope_section": [1, 1, 2],
                        },
                    ),
                    vision_config=Qwen3_5VisionConfig(
                        depth=1,
                        hidden_size=16,
                        intermediate_size=32,
                        num_heads=2,
                        out_hidden_size=16,
                        patch_size=2,
                        spatial_merge_size=2,
                        num_position_embeddings=16,
                    ),
                )
            ),
        ]
        for model in cases:
            with (
                self.subTest(architecture=type(model).__name__),
                tempfile.TemporaryDirectory() as directory,
            ):
                model.save_pretrained(directory)
                fast.save_pretrained(directory)
                provider = LocalProvider(
                    {"id": "tiny", "model": directory, "provider": "local"}
                )
                info = provider.info()
                self.assertEqual(info["layers"], 2)
                probe = provider.inspect("hello world", 0, 1)
                self.assertEqual(
                    len(probe["tokens"]),
                    provider.encode("hello world")["input_ids"].shape[-1],
                )
                self.assertEqual(probe["source"], "live")
                result = provider.compare("hello world", 0, 1, 1, 16)
                self.assertEqual(result["baseline"], result["steered"])
                self.assertFalse(provider.projections[0]._forward_pre_hooks)
                provider.compare("hello world", 0, 1, 0, 16)
                self.assertFalse(provider.projections[0]._forward_pre_hooks)


if __name__ == "__main__":
    unittest.main()
