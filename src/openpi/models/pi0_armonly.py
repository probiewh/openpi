"""Arm-only diagnostic copy of the Pi0.5 training objective."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import pi0_armonly_config
from openpi.shared import array_typing as at


class Pi0ArmOnly(_pi0.Pi0):
    """Pi0.5 with flow-matching loss restricted to the 14 arm dimensions.

    The model keeps the original 32-D action heads, so released Pi0.5 weights
    load without reshaping. Dimensions 14:32 are ignored by the training loss
    and are also discarded by the arm-only policy output transform.
    """

    def __init__(self, config: pi0_armonly_config.Pi0ArmOnlyConfig, rngs):
        super().__init__(config, rngs)
        self.arm_loss_dim = config.arm_loss_dim

    @at.typecheck
    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(
            preprocess_rng,
            observation,
            train=train,
            image_keys=self.image_keys,
        )
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        del prefix_out
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        arm_error = v_t[..., : self.arm_loss_dim] - u_t[..., : self.arm_loss_dim]
        return jnp.mean(jnp.square(arm_error), axis=-1)
