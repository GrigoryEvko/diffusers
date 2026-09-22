"""CPU tests of load_safetensors_file and load_component_state_dict."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from diffusers.models import model_loading_utils as M


WANT = {"a.weight": torch.arange(6.0).view(2, 3), "b.bias": torch.ones(4)}


def _assert_same(got: dict[str, torch.Tensor]) -> None:
    """Make sure that `got` has the tensors of WANT, bit for bit."""
    assert got.keys() == WANT.keys()
    assert all(torch.equal(got[k], WANT[k]) for k in WANT)


def _single(folder: Path, name: str = "model.safetensors") -> None:
    """Write WANT as one file."""
    folder.mkdir(parents=True, exist_ok=True)
    save_file(WANT, str(folder / name))


def _sharded(folder: Path, variant: str = "") -> None:
    """Write WANT as two shards and their index, with `variant` in each name when it is given."""
    folder.mkdir(parents=True, exist_ok=True)
    tag = f".{variant}" if variant else ""
    shards = {"a.weight": f"model{tag}-00001-of-00002.safetensors", "b.bias": f"model{tag}-00002-of-00002.safetensors"}
    for key, shard in shards.items():
        save_file({key: WANT[key]}, str(folder / shard))
    index = {"metadata": {"total_size": 40}, "weight_map": shards}
    (folder / f"model.safetensors.index{tag}.json").write_text(json.dumps(index))


def test_one_file(tmp_path: Path) -> None:
    """A component with one model.safetensors gives its tensors."""
    _single(tmp_path / "text_encoder")
    _assert_same(M.load_component_state_dict(tmp_path, "text_encoder", "cpu"))


def test_shards_of_the_index(tmp_path: Path) -> None:
    """A component with an index gives the tensors of each shard that the index names."""
    _sharded(tmp_path / "text_encoder")
    _assert_same(M.load_component_state_dict(tmp_path, "text_encoder", "cpu"))


def test_variant_selects_its_files(tmp_path: Path) -> None:
    """A variant reads model.<variant>.safetensors, or the index and the shards of that variant."""
    folder = tmp_path / "text_encoder"
    _single(folder, "model.fp16.safetensors")
    save_file({"other": torch.zeros(1)}, str(folder / "model.safetensors"))
    _assert_same(M.load_component_state_dict(tmp_path, "text_encoder", "cpu", variant="fp16"))
    _sharded(tmp_path / "text_encoder_2", variant="fp16")
    _assert_same(M.load_component_state_dict(tmp_path, "text_encoder_2", "cpu", variant="fp16"))


def test_a_missing_component_raises(tmp_path: Path) -> None:
    """A component with no weights file and no index raises EnvironmentError."""
    (tmp_path / "text_encoder").mkdir()
    with pytest.raises(EnvironmentError, match="model.safetensors"):
        M.load_component_state_dict(tmp_path, "text_encoder", "cpu")


@pytest.mark.parametrize("device,backend", [("cpu", "mmap"), ("cuda:0", "pread"), ("cuda", "pread")])
def test_backend_follows_the_device(monkeypatch: pytest.MonkeyPatch, device: str, backend: str) -> None:
    """A read to the CPU uses mmap. A read to an accelerator uses pread."""
    calls = []
    monkeypatch.setattr(M.safetensors.torch, "load_file",
                        lambda path, device, backend: calls.append((device, backend)) or {})
    M.load_safetensors_file("model.safetensors", device)
    assert calls == [(device, backend)]
