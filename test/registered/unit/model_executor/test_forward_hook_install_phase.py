"""CPU-only hook-phase and capture-lifecycle checks; no server/GPU imports.

The lifecycle checks execute the real setup functions with fake capture runners.
Only their heavyweight import environment is replaced, so phase ordering and
context restoration are exercised without loading an inference stack.
"""

import ast
import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SOURCE_ROOT = Path(__file__).resolve().parents[4] / "python/sglang/srt/model_executor"


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hooks = load_source("_hook_phase_manager", SOURCE_ROOT / "hook_manager.py")
context = load_source("_hook_phase_context", SOURCE_ROOT / "forward_context.py")


class FakeModule:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return object()


class FakeModel:
    def __init__(self):
        self.layers = [FakeModule(), FakeModule()]

    def named_modules(self):
        return [(f"layers.{i}", layer) for i, layer in enumerate(self.layers)]


def hook_spec(**overrides):
    return {
        "name": "test",
        "target_modules": ["layers.*"],
        "hook_factory": "test_hooks:factory",
        "config": {"value": 1},
        **overrides,
    }


class TestForwardHookInstallPhase(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.hook = mock.Mock()
        self.factory = mock.Mock(return_value=self.hook)
        patcher = mock.patch.object(
            hooks, "resolve_callable", return_value=self.factory
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_default_stays_post_capture(self):
        specs = [hook_spec()]
        hooks.register_forward_hooks(self.model, specs, install_phase="pre_capture")
        self.factory.assert_not_called()
        hooks.register_forward_hooks(self.model, specs)
        self.factory.assert_called_once_with({"value": 1})
        self.assertEqual([len(layer.hooks) for layer in self.model.layers], [1, 1])

    def test_pre_capture_and_post_capture_do_not_cross_register(self):
        specs = [hook_spec(install_phase="pre_capture"), hook_spec(name="late")]
        hooks.register_forward_hooks(self.model, specs, install_phase="pre_capture")
        self.assertEqual(self.factory.call_count, 1)
        hooks.register_forward_hooks(self.model, specs)
        self.assertEqual(self.factory.call_count, 2)
        self.assertEqual([len(layer.hooks) for layer in self.model.layers], [2, 2])

    def test_recapture_does_not_repeat_factory_or_registration(self):
        for phase in ("pre_capture", "post_capture"):
            with self.subTest(phase=phase):
                model = FakeModel()
                specs = [hook_spec(install_phase=phase)]
                hooks.register_forward_hooks(model, specs, install_phase=phase)
                before = self.factory.call_count
                hooks.register_forward_hooks(
                    model, copy.deepcopy(specs), install_phase=phase
                )
                self.assertEqual(self.factory.call_count, before)
                self.assertEqual([len(layer.hooks) for layer in model.layers], [1, 1])

    def test_installation_is_per_model_not_global(self):
        for model in (self.model, FakeModel()):
            hooks.register_forward_hooks(model, [hook_spec()])
        self.assertEqual(self.factory.call_count, 2)

    def test_invalid_phase_rejected_before_any_factory(self):
        for bad_phase in (None, "before_capture", "PRE_CAPTURE", 1):
            with self.subTest(phase=bad_phase):
                specs = [
                    hook_spec(install_phase="pre_capture"),
                    hook_spec(install_phase=bad_phase),
                ]
                with self.assertRaisesRegex(ValueError, "Unknown install_phase"):
                    hooks.register_forward_hooks(
                        self.model, specs, install_phase="pre_capture"
                    )
        self.factory.assert_not_called()

    def test_invalid_selected_phase_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown forward hook install phase"):
            hooks.register_forward_hooks(
                self.model, [hook_spec()], install_phase="later"
            )
        self.factory.assert_not_called()

    def test_changed_installed_configuration_rejected(self):
        specs = [hook_spec()]
        hooks.register_forward_hooks(self.model, specs)
        specs[0]["config"]["value"] = 2
        with self.assertRaisesRegex(ValueError, "specifications changed"):
            hooks.register_forward_hooks(self.model, specs)
        self.factory.assert_called_once()

    def test_removing_installed_phase_is_rejected(self):
        hooks.register_forward_hooks(self.model, [hook_spec()])
        with self.assertRaisesRegex(ValueError, "specifications changed"):
            hooks.register_forward_hooks(self.model, [])
        self.factory.assert_called_once()

    def test_partial_factory_failure_cannot_be_retried(self):
        self.factory.side_effect = RuntimeError("factory failed")
        specs = [hook_spec()]
        with self.assertRaisesRegex(RuntimeError, "factory failed"):
            hooks.register_forward_hooks(self.model, specs)
        with self.assertRaisesRegex(RuntimeError, "already started or failed"):
            hooks.register_forward_hooks(self.model, specs)
        self.factory.assert_called_once()

    def test_pre_capture_preflights_all_targets_before_side_effects(self):
        specs = [
            hook_spec(install_phase="pre_capture"),
            hook_spec(
                name="missing",
                install_phase="pre_capture",
                target_modules=["absent.*"],
            ),
        ]
        with self.assertRaisesRegex(ValueError, "requires a hook_factory and matching"):
            hooks.register_forward_hooks(self.model, specs, install_phase="pre_capture")
        self.factory.assert_not_called()

    def test_default_missing_target_retains_warning_behavior(self):
        with self.assertLogs(hooks.logger, level="WARNING") as logged:
            hooks.register_forward_hooks(
                self.model, [hook_spec(target_modules=["absent.*"])]
            )
        self.factory.assert_called_once()
        self.assertIn("No modules matched", " ".join(logged.output))


def setup_lifecycle_functions(namespace):
    path = SOURCE_ROOT / "model_runner_components/cuda_graph_setup.py"
    source = ast.parse(path.read_text(), filename=str(path))
    selected = [
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("capture_cuda_graphs", "finalize_cuda_graph_capture")
    ]
    assert len(selected) == 2
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *selected,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)


class TestForwardHookCaptureLifecycle(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.backend = SimpleNamespace(
            req_to_token_pool=object(), token_to_kv_pool=object()
        )
        self.model = FakeModel()
        self.specs = [
            hook_spec(name="early", install_phase="pre_capture"),
            hook_spec(name="late"),
        ]
        self.model_runner = SimpleNamespace(
            model=self.model,
            attn_backend=self.backend,
            is_draft_worker=False,
            device="cpu",
            forward_stream=None,
            canary_manager=None,
            maybe_init_uocr_greedy_graph=lambda: None,
        )
        self.namespace = {
            "get_observability": lambda: SimpleNamespace(forward_hooks=self.specs),
            "register_forward_hooks": hooks.register_forward_hooks,
            "ForwardContext": context.ForwardContext,
            "forward_context": context.forward_context,
            "GraphSharedOutput": SimpleNamespace(
                create_for_model_runner=lambda _: object()
            ),
            "EagerRunner": self.eager_warmup,
            "refresh_deep_gemm_layout_memory_budget": lambda _: None,
            "capture_prefill_graph": lambda **_: self.capture("prefill"),
            "capture_decode_graph": lambda **_: self.capture("decode"),
            "current_platform": SimpleNamespace(
                capabilities=SimpleNamespace(graph_capture=True)
            ),
            "GraphCapture": lambda **kwargs: SimpleNamespace(**kwargs),
            "CudaGraphsCapture": lambda **kwargs: SimpleNamespace(**kwargs),
            "get_exec": lambda: SimpleNamespace(
                comm=SimpleNamespace(enable_symm_mem=False)
            ),
            "prealloc_symmetric_memory_pool": lambda **_: None,
        }
        setup_lifecycle_functions(self.namespace)
        patcher = mock.patch.object(
            hooks, "resolve_callable", side_effect=lambda _: self.factory
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(context.set_forward_context, None)

    def factory(self, _config):
        if len(self.events) == 0:
            self.assertIs(context.get_attn_backend(), self.backend)
            self.assertIs(
                context.get_req_to_token_pool(), self.backend.req_to_token_pool
            )
            self.assertIs(context.get_token_to_kv_pool(), self.backend.token_to_kv_pool)
            self.events.append("early")
        else:
            self.events.append("late")
        return lambda *_: None

    def eager_warmup(self, _runner, *, run_warmup):
        self.events.append("eager_warmup")
        self.assertEqual(len(self.model.layers[0].hooks), 1)
        return object()

    def capture(self, name):
        self.events.append(name)
        self.assertEqual(len(self.model.layers[0].hooks), 1)
        return SimpleNamespace(runner=object())

    def test_install_before_warmup_and_capture_with_pool_context(self):
        outer = context.ForwardContext(attn_backend=object())
        with context.forward_context(outer):
            self.namespace["capture_cuda_graphs"](model_runner=self.model_runner)
            self.assertIs(context.get_forward_context(), outer)
        self.assertEqual(
            self.events, ["early", "eager_warmup", "prefill", "decode", "late"]
        )
        self.assertEqual([len(layer.hooks) for layer in self.model.layers], [2, 2])

    def test_deferred_finalize_installs_late_only_at_finalize(self):
        self.namespace["capture_cuda_graphs"](
            model_runner=self.model_runner,
            capture_decode_cuda_graph=False,
            finalize=False,
            defer_distributed_setup=True,
        )
        self.assertEqual(self.events, ["early", "eager_warmup", "prefill"])
        self.namespace["finalize_cuda_graph_capture"](self.model_runner)
        self.namespace["finalize_cuda_graph_capture"](self.model_runner)
        self.assertEqual(self.events, ["early", "eager_warmup", "prefill", "late"])

    def test_failed_pre_capture_restores_context_and_never_warms(self):
        self.specs[1]["install_phase"] = "typo"
        outer = context.ForwardContext(attn_backend=object())
        with context.forward_context(outer):
            with self.assertRaisesRegex(ValueError, "Unknown install_phase"):
                self.namespace["capture_cuda_graphs"](model_runner=self.model_runner)
            self.assertIs(context.get_forward_context(), outer)
        self.assertEqual(self.events, [])


if __name__ == "__main__":
    unittest.main()
