# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for a text-encoder LoRA on the two module layouts of the CLIP text encoders.

transformers 4 wraps the layers of `CLIPTextModel` in a `text_model` attribute. transformers 5 removed that wrapper
from `CLIPTextModel`, but `CLIPTextModelWithProjection` keeps it. Kohya SDXL LoRAs name the text-encoder weights with
the wrapper, for example `lora_te1_text_model_encoder_layers_0_self_attn_q_proj`. The tests make sure that such a LoRA
changes the correct modules of the two text encoders of SDXL, with the correct scale, and no other module.
"""

import pytest
import torch
from torch import nn

from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.loaders.lora_base import _adapt_lora_config_names, _text_model_prefix_mapper
from diffusers.models.lora import adjust_lora_scale_text_encoder, text_encoder_attn_modules, text_encoder_mlp_modules
from diffusers.utils.import_utils import is_peft_available, is_transformers_available


if is_transformers_available():
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

pytestmark = pytest.mark.skipif(
    not (is_peft_available() and is_transformers_available()), reason="The tests need peft and transformers."
)

RANK = 4
HIDDEN = 32
INTERMEDIATE = 37
NUM_LAYERS = 3
ATTN_ALPHA = 2.0
MLP_ALPHA = 8.0
# The LoRA changes these modules. The names are the transformers 4 module paths, as in the kohya keys.
TARGETS = {
    "text_encoder": [
        "text_model.encoder.layers.0.self_attn.q_proj",
        "text_model.encoder.layers.0.self_attn.k_proj",
        "text_model.encoder.layers.0.self_attn.v_proj",
        "text_model.encoder.layers.0.self_attn.out_proj",
        "text_model.encoder.layers.1.mlp.fc1",
        "text_model.encoder.layers.1.mlp.fc2",
    ],
    "text_encoder_2": [
        "text_model.encoder.layers.2.self_attn.q_proj",
        "text_model.encoder.layers.2.self_attn.k_proj",
        "text_model.encoder.layers.2.self_attn.v_proj",
        "text_model.encoder.layers.2.self_attn.out_proj",
        "text_model.encoder.layers.0.mlp.fc1",
        "text_model.encoder.layers.0.mlp.fc2",
    ],
}
KOHYA_PREFIX = {"text_encoder": "lora_te1_", "text_encoder_2": "lora_te2_"}


def _clip_config() -> "CLIPTextConfig":
    """Give the configuration of a tiny CLIP text encoder."""
    return CLIPTextConfig(
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=1,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        layer_norm_eps=1e-05,
        num_attention_heads=4,
        num_hidden_layers=NUM_LAYERS,
        vocab_size=1000,
        max_position_embeddings=77,
        hidden_act="gelu",
        projection_dim=HIDDEN,
    )


def _tiny_sdxl_pipeline() -> StableDiffusionXLPipeline:
    """Give a tiny SDXL pipeline with no tokenizers. The LoRA loaders do not use the tokenizers."""
    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        block_out_channels=(32, 64),
        layers_per_block=1,
        sample_size=32,
        in_channels=4,
        out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        attention_head_dim=(2, 4),
        use_linear_projection=True,
        addition_embed_type="text_time",
        addition_time_embed_dim=8,
        transformer_layers_per_block=(1, 2),
        projection_class_embeddings_input_dim=80,
        cross_attention_dim=64,
        norm_num_groups=1,
    )
    vae = AutoencoderKL(
        block_out_channels=[32, 64],
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
        up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
        latent_channels=4,
        norm_num_groups=1,
    )
    text_encoder = CLIPTextModel(_clip_config()).eval()
    text_encoder_2 = CLIPTextModelWithProjection(_clip_config()).eval()
    return StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=None,
        tokenizer_2=None,
        unet=unet,
        scheduler=EulerDiscreteScheduler(),
    )


def _kohya_te_lora() -> dict[str, torch.Tensor]:
    """
    Give a kohya SDXL LoRA with keys for the two text encoders only.

    The attention modules have alpha `ATTN_ALPHA` and the MLP modules have alpha `MLP_ALPHA`. The two scales are
    different, so the test also makes sure that each alpha goes to the correct module.
    """
    generator = torch.Generator().manual_seed(1)
    state_dict = {}
    for component, names in TARGETS.items():
        for name in names:
            kohya_name = KOHYA_PREFIX[component] + name.replace(".", "_")
            in_features = INTERMEDIATE if name.endswith("fc2") else HIDDEN
            out_features = INTERMEDIATE if name.endswith("fc1") else HIDDEN
            alpha = MLP_ALPHA if ".mlp." in name else ATTN_ALPHA
            state_dict[f"{kohya_name}.lora_down.weight"] = torch.randn(RANK, in_features, generator=generator)
            state_dict[f"{kohya_name}.lora_up.weight"] = torch.randn(out_features, RANK, generator=generator)
            state_dict[f"{kohya_name}.alpha"] = torch.tensor(alpha)
    return state_dict


def _module_path(text_encoder: nn.Module, checkpoint_name: str) -> str:
    """Give the module path of `checkpoint_name` in the layout of `text_encoder`."""
    return checkpoint_name if hasattr(text_encoder, "text_model") else checkpoint_name.removeprefix("text_model.")


def _linear_paths(text_encoder: nn.Module) -> list[str]:
    """Give the path of each `nn.Linear` of `text_encoder` before a LoRA loads."""
    return [name for name, module in text_encoder.named_modules() if isinstance(module, nn.Linear)]


class TestCLIPTextEncoderLayout:
    """Tests for the CLIP text-encoder helpers and the kohya LoRA loader on the two layouts."""

    def test_the_layouts_of_the_text_encoders(self):
        """Make sure that the test uses one text encoder with the wrapper and one without, as in transformers 5."""
        pipe = _tiny_sdxl_pipeline()
        assert hasattr(pipe.text_encoder_2, "text_model")
        # `CLIPTextModel` of transformers 4 has the wrapper. With transformers 4, the flat layout is not tested.
        if not hasattr(pipe.text_encoder, "text_model"):
            assert {name for name, _ in pipe.text_encoder.named_children()} == {
                "embeddings",
                "encoder",
                "final_layer_norm",
            }

    @pytest.mark.parametrize("component", ["text_encoder", "text_encoder_2"])
    def test_text_encoder_modules_give_checkpoint_names(self, component):
        """The helpers give the checkpoint names and the layer modules of the text encoder, for the two layouts."""
        text_encoder = getattr(_tiny_sdxl_pipeline(), component)
        layers = getattr(text_encoder, "text_model", text_encoder).encoder.layers

        attn = text_encoder_attn_modules(text_encoder)
        mlp = text_encoder_mlp_modules(text_encoder)

        assert [name for name, _ in attn] == [f"text_model.encoder.layers.{i}.self_attn" for i in range(NUM_LAYERS)]
        assert [name for name, _ in mlp] == [f"text_model.encoder.layers.{i}.mlp" for i in range(NUM_LAYERS)]
        assert all(module is layers[i].self_attn for i, (_, module) in enumerate(attn))
        assert all(module is layers[i].mlp for i, (_, module) in enumerate(mlp))
        adjust_lora_scale_text_encoder(text_encoder, 0.5)

    def test_prefix_mapper_removes_the_prefix_without_the_wrapper(self):
        """Without the wrapper, the mapper removes `text_model.` and keeps the other names."""
        text_encoder = _tiny_sdxl_pipeline().text_encoder
        if hasattr(text_encoder, "text_model"):
            pytest.skip("This transformers version keeps the wrapper in CLIPTextModel.")
        adapt = _text_model_prefix_mapper(text_encoder)

        assert adapt("text_model.encoder.layers.0.self_attn.q_proj.lora_A.weight") == (
            "encoder.layers.0.self_attn.q_proj.lora_A.weight"
        )
        assert adapt("encoder.layers.0.mlp.fc1.lora_B.weight") == "encoder.layers.0.mlp.fc1.lora_B.weight"
        assert adapt("text_model.unknown.lora_A.weight") == "text_model.unknown.lora_A.weight"

    def test_prefix_mapper_adds_the_prefix_with_the_wrapper(self):
        """With the wrapper, the mapper adds `text_model.` to a name of the wrapper and keeps the other names."""
        text_encoder_2 = _tiny_sdxl_pipeline().text_encoder_2
        adapt = _text_model_prefix_mapper(text_encoder_2)

        assert adapt("encoder.layers.0.self_attn.q_proj.lora_A.weight") == (
            "text_model.encoder.layers.0.self_attn.q_proj.lora_A.weight"
        )
        assert adapt("text_model.encoder.layers.0.mlp.fc1.lora_B.weight") == (
            "text_model.encoder.layers.0.mlp.fc1.lora_B.weight"
        )
        assert adapt("text_projection.lora_A.weight") == "text_projection.lora_A.weight"

    def test_lora_config_names_follow_the_layout(self):
        """The mapper also changes the module names in the LoRA metadata, but not a regular expression."""
        text_encoder_2 = _tiny_sdxl_pipeline().text_encoder_2
        adapt = _text_model_prefix_mapper(text_encoder_2)
        kwargs = {
            "r": RANK,
            "target_modules": ["encoder.layers.0.self_attn.q_proj", "q_proj"],
            "rank_pattern": {"encoder.layers.1.mlp.fc1": 8},
            "alpha_pattern": {"text_model.encoder.layers.1.mlp.fc2": 16},
        }

        adapted = _adapt_lora_config_names(kwargs, adapt)

        assert adapted["target_modules"] == ["text_model.encoder.layers.0.self_attn.q_proj", "q_proj"]
        assert adapted["rank_pattern"] == {"text_model.encoder.layers.1.mlp.fc1": 8}
        assert adapted["alpha_pattern"] == {"text_model.encoder.layers.1.mlp.fc2": 16}
        assert kwargs["target_modules"] == ["encoder.layers.0.self_attn.q_proj", "q_proj"]
        assert _adapt_lora_config_names({"target_modules": "q_proj|k_proj"}, adapt)["target_modules"] == (
            "q_proj|k_proj"
        )

    @torch.no_grad()
    def test_kohya_sdxl_lora_changes_only_the_target_modules(self):
        """
        Load a kohya SDXL LoRA with text-encoder keys, then make sure that:

        - Each target module gives `W x + b + (alpha / rank) * up(down(x))`
        - Each other linear module gives the same output as before the load
        - The output of each text encoder changes
        """
        pipe = _tiny_sdxl_pipeline()
        kohya = _kohya_te_lora()
        generator = torch.Generator().manual_seed(2)
        input_ids = torch.randint(3, 1000, (2, 16), generator=generator)

        before = {}
        for component in TARGETS:
            text_encoder = getattr(pipe, component)
            paths = _linear_paths(text_encoder)
            probes = {path: torch.randn(3, text_encoder.get_submodule(path).in_features) for path in paths}
            outputs = {path: text_encoder.get_submodule(path)(probe) for path, probe in probes.items()}
            hidden = text_encoder(input_ids, output_hidden_states=True).hidden_states[-2]
            before[component] = (paths, probes, outputs, hidden)

        pipe.load_lora_weights(kohya, adapter_name="kohya")

        for component, names in TARGETS.items():
            text_encoder = getattr(pipe, component)
            paths, probes, outputs, hidden = before[component]
            targets = {_module_path(text_encoder, name): name for name in names}

            wrapped = {name for name, module in text_encoder.named_modules() if hasattr(module, "lora_A")}
            assert wrapped == set(targets), f"{component}: LoRA layers on {sorted(wrapped)}"

            for path in paths:
                module = text_encoder.get_submodule(path)
                probe = probes[path]
                if path not in targets:
                    assert torch.equal(module(probe), outputs[path]), f"{component}: {path} changed"
                    continue
                kohya_name = KOHYA_PREFIX[component] + targets[path].replace(".", "_")
                down = kohya[f"{kohya_name}.lora_down.weight"]
                up = kohya[f"{kohya_name}.lora_up.weight"]
                scale = kohya[f"{kohya_name}.alpha"].item() / RANK
                expected = outputs[path] + scale * (probe @ down.T @ up.T)
                assert torch.allclose(module(probe), expected, atol=1e-5, rtol=1e-5), f"{component}: {path}"

            new_hidden = text_encoder(input_ids, output_hidden_states=True).hidden_states[-2]
            assert not torch.allclose(new_hidden, hidden), f"{component}: the LoRA did not change the output"

    @torch.no_grad()
    def test_unload_restores_the_text_encoders(self):
        """After `unload_lora_weights`, each text encoder gives its output from before the load."""
        pipe = _tiny_sdxl_pipeline()
        input_ids = torch.randint(3, 1000, (1, 16), generator=torch.Generator().manual_seed(3))
        reference = {c: getattr(pipe, c)(input_ids).last_hidden_state for c in TARGETS}

        pipe.load_lora_weights(_kohya_te_lora(), adapter_name="kohya")
        pipe.unload_lora_weights()

        for component in TARGETS:
            output = getattr(pipe, component)(input_ids).last_hidden_state
            assert torch.equal(output, reference[component]), component
