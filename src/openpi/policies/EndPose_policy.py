import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class AlohaInputs(transforms.DataTransformFn):
    """Inputs for the Aloha policy (20D format).

    Expected inputs (20D format per arm):
    - images: dict[name, img] where img is [channel, height, width]
    - state: [20] - xyz(3) + 6d_rotation(6) + gripper(1) for each arm
    - actions: [action_horizon, 20] - xyz(3) + 6d_rotation(6) + gripper(1) for each arm
    """

    adapt_to_pi: bool = True

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            # 20D format: pass as-is, no transformation needed
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs

@dataclasses.dataclass(frozen=True)
class AlohaOutputs(transforms.DataTransformFn):
    """Outputs for the Aloha policy (20D format).
    
    Processes 20-dimensional action output (xyz + 6d rotation + gripper for each arm):
    - Left arm: xyz (3) + 6d rotation (6) + gripper (1) = 10 dims (indices 0-9)
    - Right arm: xyz (3) + 6d rotation (6) + gripper (1) = 10 dims (indices 10-19)
    
    Output format: xyz (3) + quaternion (4) + gripper (1) for each arm (8 dims per arm, 16 total)
    """

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = False

    def __call__(self, data: dict) -> dict:
        # Process 20 dims: xyz(3) + 6d_rot(6) + gripper(1) + xyz(3) + 6d_rot(6) + gripper(1)
        actions = np.asarray(data["actions"][:, :20])
        
        # Convert 6d rotation to quaternion for both arms
        actions_converted = _convert_6d_rotation_to_quaternion(actions)
        
        return {"actions": actions_converted}


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    value = value + 0.5476
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _decode_aloha(data: dict, *, adapt_to_pi: bool = False) -> dict:
    # state is [left_arm_joint_angles, left_arm_gripper, right_arm_joint_angles, right_arm_gripper]
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi)

    def convert_image(img):
        img = np.asarray(img)
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    """Decode 20D state (xyz + 6d_rotation + gripper for each arm).
    
    No transformation needed for 20D format.
    """
    return state


def _rot6d_to_quaternion(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to quaternion.
    
    The 6D representation consists of two 3D vectors that are orthonormalized to form
    a rotation matrix, which is then converted to a quaternion.
    
    Args:
        rot6d: Array of shape (..., 6) representing the 6D rotation
        
    Returns:
        Quaternion array of shape (..., 4) in format [x, y, z, w]
    """
    batch_shape = rot6d.shape[:-1]
    rot6d = rot6d.reshape(-1, 6)
    
    # Extract the two 3D vectors
    v1 = rot6d[:, :3]
    v2 = rot6d[:, 3:6]
    
    # Normalize first vector
    v1_norm = np.linalg.norm(v1, axis=1, keepdims=True)
    v1 = v1 / (v1_norm + 1e-8)
    
    # Gram-Schmidt orthogonalization for second vector
    v2 = v2 - np.sum(v1 * v2, axis=1, keepdims=True) * v1
    v2_norm = np.linalg.norm(v2, axis=1, keepdims=True)
    v2 = v2 / (v2_norm + 1e-8)
    
    # Third vector from cross product
    v3 = np.cross(v1, v2, axis=1)
    
    # Stack to form rotation matrix: [v1 | v2 | v3]
    rot_mat = np.stack([v1, v2, v3], axis=2)  # Shape: (batch, 3, 3)
    
    # Convert rotation matrix to quaternion
    quaternions = _rot_mat_to_quaternion(rot_mat)
    
    # Reshape back to original batch shape
    quaternions = quaternions.reshape(*batch_shape, 4)
    
    return quaternions


def _rot_mat_to_quaternion(rot_mat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion.
    
    Args:
        rot_mat: Rotation matrix array of shape (..., 3, 3)
        
    Returns:
        Quaternion array of shape (..., 4) in format [x, y, z, w]
    """
    batch_shape = rot_mat.shape[:-2]
    rot_mat = rot_mat.reshape(-1, 3, 3)
    
    batch_size = rot_mat.shape[0]
    quaternions = np.zeros((batch_size, 4), dtype=rot_mat.dtype)
    
    for i in range(batch_size):
        # Shepperd's method for numerical stability
        trace = np.trace(rot_mat[i])
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (rot_mat[i, 2, 1] - rot_mat[i, 1, 2]) * s
            y = (rot_mat[i, 0, 2] - rot_mat[i, 2, 0]) * s
            z = (rot_mat[i, 1, 0] - rot_mat[i, 0, 1]) * s
        elif rot_mat[i, 0, 0] > rot_mat[i, 1, 1] and rot_mat[i, 0, 0] > rot_mat[i, 2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot_mat[i, 0, 0] - rot_mat[i, 1, 1] - rot_mat[i, 2, 2])
            w = (rot_mat[i, 2, 1] - rot_mat[i, 1, 2]) / s
            x = 0.25 * s
            y = (rot_mat[i, 0, 1] + rot_mat[i, 1, 0]) / s
            z = (rot_mat[i, 0, 2] + rot_mat[i, 2, 0]) / s
        elif rot_mat[i, 1, 1] > rot_mat[i, 2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot_mat[i, 1, 1] - rot_mat[i, 0, 0] - rot_mat[i, 2, 2])
            w = (rot_mat[i, 0, 2] - rot_mat[i, 2, 0]) / s
            x = (rot_mat[i, 0, 1] + rot_mat[i, 1, 0]) / s
            y = 0.25 * s
            z = (rot_mat[i, 1, 2] + rot_mat[i, 2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rot_mat[i, 2, 2] - rot_mat[i, 0, 0] - rot_mat[i, 1, 1])
            w = (rot_mat[i, 1, 0] - rot_mat[i, 0, 1]) / s
            x = (rot_mat[i, 0, 2] + rot_mat[i, 2, 0]) / s
            y = (rot_mat[i, 1, 2] + rot_mat[i, 2, 1]) / s
            z = 0.25 * s
        
        quaternions[i] = [x, y, z, w]
    
    # Reshape back to batch shape
    quaternions = quaternions.reshape(*batch_shape, 4)
    
    return quaternions


def _convert_6d_rotation_to_quaternion(actions: np.ndarray) -> np.ndarray:
    """Convert 20D actions (xyz + 6d_rot + gripper for each arm) to 16D (xyz + quat + gripper).
    
    Input format (20 dims):
    - [0:3] - Left arm xyz
    - [3:9] - Left arm 6D rotation
    - [9] - Left arm gripper
    - [10:13] - Right arm xyz
    - [13:19] - Right arm 6D rotation
    - [19] - Right arm gripper
    
    Output format (16 dims):
    - [0:3] - Left arm xyz
    - [3:7] - Left arm quaternion (x, y, z, w)
    - [7] - Left arm gripper
    - [8:11] - Right arm xyz
    - [11:15] - Right arm quaternion (x, y, z, w)
    - [15] - Right arm gripper
    
    Args:
        actions: Array of shape (horizon, 20)
        
    Returns:
        Array of shape (horizon, 16)
    """
    horizon = actions.shape[0]
    output_actions = np.zeros((horizon, 16), dtype=actions.dtype)
    
    # Process left arm
    left_xyz = actions[:, 0:3]  # [horizon, 3]
    left_6d = actions[:, 3:9]    # [horizon, 6]
    left_gripper = actions[:, 9]  # [horizon]
    
    # Process right arm
    right_xyz = actions[:, 10:13]  # [horizon, 3]
    right_6d = actions[:, 13:19]   # [horizon, 6]
    right_gripper = actions[:, 19] # [horizon]
    
    # Convert 6D rotations to quaternions
    left_quat = _rot6d_to_quaternion(left_6d)   # [horizon, 4]
    right_quat = _rot6d_to_quaternion(right_6d) # [horizon, 4]
    
    # Assemble output
    output_actions[:, 0:3] = left_xyz
    output_actions[:, 3:7] = left_quat
    output_actions[:, 7] = left_gripper
    output_actions[:, 8:11] = right_xyz
    output_actions[:, 11:15] = right_quat
    output_actions[:, 15] = right_gripper
    
    return output_actions



