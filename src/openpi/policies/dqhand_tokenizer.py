from __future__ import annotations

import dataclasses
import pathlib

import numpy as np


@dataclasses.dataclass(frozen=True)
class _HandCodec:
    mean: np.ndarray
    std: np.ndarray
    encoder_w1: np.ndarray
    encoder_b1: np.ndarray
    encoder_w2: np.ndarray
    encoder_b2: np.ndarray
    codebook1: np.ndarray
    codebook2: np.ndarray
    decoder_w1: np.ndarray
    decoder_b1: np.ndarray
    decoder_w2: np.ndarray
    decoder_b2: np.ndarray
    combined_to_order: np.ndarray
    order_to_combined: np.ndarray

    def encode(self, hand: np.ndarray) -> np.ndarray:
        shape = hand.shape[:-1]
        x = (hand.reshape(-1, hand.shape[-1]) - self.mean) / self.std
        hidden = np.tanh(x @ self.encoder_w1.T + self.encoder_b1)
        latent = hidden @ self.encoder_w2.T + self.encoder_b2
        first = np.argmin(
            np.sum((latent[:, None, :] - self.codebook1[None, :, :]) ** 2, axis=-1),
            axis=1,
        )
        residual = latent - self.codebook1[first]
        second = np.argmin(
            np.sum((residual[:, None, :] - self.codebook2[None, :, :]) ** 2, axis=-1),
            axis=1,
        )
        combined = first * self.codebook2.shape[0] + second
        ordered = self.combined_to_order[combined]
        return ordered.reshape(shape).astype(np.float32)

    def decode(self, ordered_code: np.ndarray) -> np.ndarray:
        shape = ordered_code.shape
        ordered = np.clip(np.rint(ordered_code), 0, len(self.order_to_combined) - 1).astype(np.int64)
        combined = self.order_to_combined[ordered.reshape(-1)]
        second_size = self.codebook2.shape[0]
        first = combined // second_size
        second = combined % second_size
        latent = self.codebook1[first] + self.codebook2[second]
        hidden = np.tanh(latent @ self.decoder_w1.T + self.decoder_b1)
        normalized = hidden @ self.decoder_w2.T + self.decoder_b2
        hand = normalized * self.std + self.mean
        return hand.reshape(*shape, self.mean.shape[0]).astype(np.float32)


@dataclasses.dataclass(frozen=True)
class DQHandTokenizer:
    """NumPy runtime for the frozen left/right residual-VQ hand codecs."""

    left: _HandCodec
    right: _HandCodec

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "DQHandTokenizer":
        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"DQ-Hand tokenizer not found at {path}. Run scripts/train_dqhand_tokenizer.py first."
            )
        with np.load(path, allow_pickle=False) as arrays:
            return cls(
                left=_codec_from_npz(arrays, "left"),
                right=_codec_from_npz(arrays, "right"),
            )

    def encode(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape[-1] < 26:
            raise ValueError(f"Expected at least 26 action dimensions, got {actions.shape}")
        left_code = self.left.encode(actions[..., 14:20])
        right_code = self.right.encode(actions[..., 20:26])
        return np.concatenate(
            [actions[..., :14], left_code[..., None], right_code[..., None]], axis=-1
        ).astype(np.float32)

    def decode(self, compact_actions: np.ndarray) -> np.ndarray:
        compact_actions = np.asarray(compact_actions, dtype=np.float32)
        if compact_actions.shape[-1] < 16:
            raise ValueError(f"Expected at least 16 compact action dimensions, got {compact_actions.shape}")
        left = self.left.decode(compact_actions[..., 14])
        right = self.right.decode(compact_actions[..., 15])
        return np.concatenate([compact_actions[..., :14], left, right], axis=-1).astype(np.float32)


def _codec_from_npz(arrays, prefix: str) -> _HandCodec:
    def get(name: str) -> np.ndarray:
        return np.asarray(arrays[f"{prefix}_{name}"])

    return _HandCodec(
        mean=get("mean"),
        std=get("std"),
        encoder_w1=get("encoder_w1"),
        encoder_b1=get("encoder_b1"),
        encoder_w2=get("encoder_w2"),
        encoder_b2=get("encoder_b2"),
        codebook1=get("codebook1"),
        codebook2=get("codebook2"),
        decoder_w1=get("decoder_w1"),
        decoder_b1=get("decoder_b1"),
        decoder_w2=get("decoder_w2"),
        decoder_b2=get("decoder_b2"),
        combined_to_order=get("combined_to_order").astype(np.int64),
        order_to_combined=get("order_to_combined").astype(np.int64),
    )
