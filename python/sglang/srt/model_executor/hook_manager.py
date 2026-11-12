from __future__ import annotations

import copy
import fnmatch
import importlib
import logging
from typing import TYPE_CHECKING, Any, Callable, List, Optional

if TYPE_CHECKING:
    import torch.nn as nn

logger = logging.getLogger(__name__)


_INSTALL_PHASES = ("pre_capture", "post_capture")


def register_forward_hooks(
    model: nn.Module,
    hook_specs: List[dict[str, Any]],
    *,
    install_phase: str = "post_capture",
) -> None:
    """
    hook_specs is a list of dicts from server_args.forward_hooks.
    Attaches forward hooks to the matching modules once per model and phase.

    The default is post_capture, preserving hook-free graph capture. Explicit
    pre_capture factories may install graph-safe numerical interventions before
    warmup; their hooks also run during warmup/capture, not during graph replay.
    Factories retain the factory(config) interface. Recapture must not run a
    factory again or attach duplicate hooks. Changing installed specifications
    requires a new model instance.
    """
    if install_phase not in _INSTALL_PHASES:
        raise ValueError(f"Unknown forward hook install phase: {install_phase!r}")
    # Validate the entire specification before invoking any factory, including
    # specs for the later phase: a typo must not leave partial interventions.
    for spec in hook_specs:
        phase = spec.get("install_phase", "post_capture")
        if phase not in _INSTALL_PHASES:
            raise ValueError(
                f"Unknown install_phase {phase!r} for forward hook "
                f"{spec.get('name', '')!r}; expected one of {_INSTALL_PHASES}"
            )
    selected = [
        spec
        for spec in hook_specs
        if spec.get("install_phase", "post_capture") == install_phase
    ]
    installations = getattr(model, "_sglang_forward_hook_installations", None)
    previous = installations.get(install_phase) if installations is not None else None
    if previous is not None:
        if previous["specs"] != selected:
            raise ValueError(
                "Forward hook specifications changed after "
                f"{install_phase} installation"
            )
        if previous["status"] != "installed":
            raise RuntimeError(
                f"Forward hook {install_phase} installation already started or failed; "
                "use a new model instance"
            )
        return

    if not selected:
        return
    if installations is None:
        installations = {}
        model._sglang_forward_hook_installations = installations
    installation = {"specs": copy.deepcopy(selected), "status": "installing"}
    installations[install_phase] = installation
    try:
        _register_selected_forward_hooks(
            model, selected, require_matches=install_phase == "pre_capture"
        )
    except BaseException:
        # A factory can have side effects beyond registered handles. Do not
        # retry a partial installation against the same model.
        installation["status"] = "failed"
        raise
    installation["status"] = "installed"


def _register_selected_forward_hooks(
    model: nn.Module,
    hook_specs: List[dict[str, Any]],
    *,
    require_matches: bool,
) -> None:
    name_to_module = dict(model.named_modules())
    if require_matches:
        # Opt-in numerical factories must not silently run without the intended
        # targets. Preflight all targets before invoking any such factory.
        for spec in hook_specs:
            patterns = spec.get("target_modules", [])
            if not spec.get("hook_factory") or not any(
                fnmatch.fnmatch(name, pattern)
                for name in name_to_module
                for pattern in patterns
            ):
                raise ValueError(
                    f"Pre-capture hook {spec.get('name', '')!r} requires a "
                    "hook_factory and matching target_modules"
                )

    for spec in hook_specs:
        spec_name = spec.get("name", "")
        target_patterns = spec.get("target_modules", [])
        if not target_patterns:
            logger.warning(f"Hook spec '{spec_name}' has no 'target_modules', skipping")
            continue

        hook_factory_path = spec.get("hook_factory")
        if not hook_factory_path:
            logger.warning(f"Hook spec '{spec_name}' has no 'hook_factory', skipping")
            continue

        config = spec.get("config") or {}
        hook_factory = resolve_callable(hook_factory_path)

        hook = hook_factory(config) if hook_factory else None
        if hook is None:
            logger.warning(
                f"Hook factory '{hook_factory_path}' for spec '{spec_name}' "
                "returned None, not registering any hook"
            )
            continue

        # Resolve patterns like "model.layers.*.mlp"
        matched = []
        for name, module in name_to_module.items():
            if any(fnmatch.fnmatch(name, pattern) for pattern in target_patterns):
                matched.append((name, module))

        if not matched:
            logger.warning(
                f"No modules matched hook spec '{spec_name}' patterns={target_patterns}"
            )
            continue

        for module_name, module in matched:
            _ = module.register_forward_hook(hook)
            logger.info(f"Registered forward hook '{spec_name}' on {module_name}")


def resolve_callable(path: Optional[str]) -> Optional[Callable]:
    if path is None:
        return None

    if ":" in path:
        module_name, fn_name = path.split(":", 1)
    else:
        parts = path.split(".")
        if len(parts) < 2:
            raise ValueError(
                f"Invalid hook callable path '{path}'. "
                "Expected 'module.submodule:factory' or 'module.submodule.factory'."
            )
        *mod_parts, fn_name = parts
        module_name = ".".join(mod_parts)

    module = importlib.import_module(module_name)
    try:
        return getattr(module, fn_name)
    except AttributeError as e:
        raise AttributeError(
            f"Module '{module_name}' has no attribute '{fn_name}' "
            f"(from hook path '{path}')"
        ) from e
