"""Configuration for the independent arm-only Pi0.5 diagnostic model."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
from typing_extensions import override

from openpi.models import pi0_config
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    from openpi.models.pi0_armonly import Pi0ArmOnly


@dataclasses.dataclass(frozen=True)
class Pi0ArmOnlyConfig(pi0_config.Pi0Config):
    """Pi0.5 config whose loss is computed only on the first 14 actions."""

    arm_loss_dim: int = 14

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0ArmOnly":
        from openpi.models.pi0_armonly import Pi0ArmOnly

        return Pi0ArmOnly(self, rngs=nnx.Rngs(rng))
