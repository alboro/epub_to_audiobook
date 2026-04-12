from __future__ import annotations

import importlib
import io
import sys
import types
from importlib import resources


def install_pkg_resources_compat():
    try:
        import pkg_resources  # noqa: F401
        return
    except ImportError:
        pass

    compat_module = types.ModuleType("pkg_resources")

    def resource_stream(package_or_requirement, resource_name):
        package_name = _resolve_package_name(package_or_requirement)
        data = resources.files(package_name).joinpath(resource_name).read_bytes()
        return io.BytesIO(data)

    compat_module.resource_stream = resource_stream
    sys.modules["pkg_resources"] = compat_module


def load_tsnorm_backend():
    install_pkg_resources_compat()
    from tsnorm import Normalizer as TSNormBackend

    return TSNormBackend


def create_tsnorm_backend(
    *,
    stress_mark: str,
    stress_mark_pos: str,
    stress_monosyllabic: bool,
    stress_yo: bool,
    min_word_len: int,
):
    backend_class = load_tsnorm_backend()
    return backend_class(
        stress_mark=stress_mark,
        stress_mark_pos=stress_mark_pos,
        stress_yo=stress_yo,
        stress_monosyllabic=stress_monosyllabic,
        min_word_len=min_word_len,
    )


def _resolve_package_name(package_or_requirement):
    if hasattr(package_or_requirement, "__spec__"):
        package_name = package_or_requirement.__spec__.name
    else:
        package_name = str(package_or_requirement)

    module = importlib.import_module(package_name)
    if getattr(module, "__spec__", None) and module.__spec__.submodule_search_locations is None:
        return module.__package__ or package_name.rpartition(".")[0]
    return package_name
