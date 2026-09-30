import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0_history import Pi0History


HISTORY_IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
    "goal_0_rgb",
    "previous_base_0_rgb",
    "previous_left_wrist_0_rgb",
)

HISTORY_PAIRS = (
    ("previous_base_0_rgb", "base_0_rgb"),
    ("previous_left_wrist_0_rgb", "left_wrist_0_rgb"),
)


@dataclasses.dataclass(frozen=True)
class Pi0HistoryConfig(_model.BaseModelConfig):
    """Independent config for the copied history-aware Pi0.5 model."""

    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b_lora32"
    action_expert_variant: _gemma.Variant = "gemma_300m_lora"
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 200
    pi05: bool = True
    image_keys: tuple[str, ...] = HISTORY_IMAGE_KEYS
    history_pairs: tuple[tuple[str, str], ...] = HISTORY_PAIRS
    discrete_state_input: bool = True
    pytorch_compile_mode: str | None = "max-autotune"

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0History":
        from openpi.models.pi0_history import Pi0History

        return Pi0History(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(
        self, *, batch_size: int = 1
    ) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct(
            [batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32
        )
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={key: image_spec for key in self.image_keys},
                image_masks={key: image_mask_spec for key in self.image_keys},
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct(
                    [batch_size, self.max_token_len], jnp.int32
                ),
                tokenized_prompt_mask=jax.ShapeDtypeStruct(
                    [batch_size, self.max_token_len], bool
                ),
            )
        action_spec = jax.ShapeDtypeStruct(
            [batch_size, self.action_horizon, self.action_dim], jnp.float32
        )
        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(gemma_params_filter)
            if "lora" not in self.action_expert_variant:
                filters.append(nnx.Not(action_expert_params_filter))
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(action_expert_params_filter)
            has_lora = True
        if has_lora:
            filters.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")))
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
