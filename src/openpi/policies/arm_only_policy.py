"""Transforms for the arm-only diagnostic variant of the cup task."""

from __future__ import annotations

import dataclasses

import numpy as np

from openpi import transforms
from openpi.policies import aloha_policy


@dataclasses.dataclass(frozen=True)
class ArmOnlyInputs(transforms.DataTransformFn):
    """Use the normal four-camera inputs but supervise only the 14 arm joints."""

    adapt_to_pi: bool = False
    goal_camera_name: str | None = None

    def __call__(self, data: dict) -> dict:
        output = aloha_policy.AlohaInputs(
            adapt_to_pi=self.adapt_to_pi,
            goal_camera_name=self.goal_camera_name,
        )(data)
        if "actions" in output:
            actions = np.asarray(output["actions"])
            if actions.shape[-1] < 14:
                raise ValueError(f"Expected at least 14 arm action dimensions, got {actions.shape}")
            output["actions"] = actions[..., :14]
        return output


@dataclasses.dataclass(frozen=True)
class ArmOnlyOutputs(transforms.DataTransformFn):
    """Return 26-D commands while holding all 12 hand joints at their current state."""

    adapt_to_pi: bool = False
    arm_action_dim: int = 14
    full_action_dim: int = 26

    def __call__(self, data: dict) -> dict:
        arm_actions = np.asarray(data["actions"])[..., : self.arm_action_dim]
        state = np.asarray(data["state"])
        if state.shape[-1] < self.full_action_dim:
            raise ValueError(
                f"Expected at least {self.full_action_dim} state dimensions, got {state.shape}"
            )

        hand_state = state[..., self.arm_action_dim : self.full_action_dim]
        held_hands = np.broadcast_to(hand_state, (*arm_actions.shape[:-1], hand_state.shape[-1])).copy()
        full_actions = np.concatenate([arm_actions, held_hands], axis=-1)
        return aloha_policy.AlohaOutputs(
            adapt_to_pi=self.adapt_to_pi,
            output_action_dim=self.full_action_dim,
        )({**data, "actions": full_actions})
