"""Train the frozen left/right residual-VQ codecs used by the DQ-Hand config."""

from __future__ import annotations

import dataclasses
import pathlib

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import tyro


class ResidualVQAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, latent_dim: int = 16, codebook_size: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, latent_dim))
        self.codebook1 = nn.Parameter(torch.randn(codebook_size, latent_dim) * 0.05)
        self.codebook2 = nn.Parameter(torch.randn(codebook_size, latent_dim) * 0.05)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, input_dim))

    @staticmethod
    def nearest(x: torch.Tensor, codebook: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distance = torch.sum((x[:, None, :] - codebook[None, :, :]) ** 2, dim=-1)
        index = torch.argmin(distance, dim=1)
        return codebook[index], index

    def forward(self, x: torch.Tensor):
        latent = self.encoder(x)
        first, first_index = self.nearest(latent.detach(), self.codebook1)
        residual = latent.detach() - first.detach()
        second, second_index = self.nearest(residual, self.codebook2)
        quantized = first + second
        straight_through = latent + (quantized - latent).detach()
        reconstruction = self.decoder(straight_through)
        return reconstruction, latent, quantized, first_index, second_index


@dataclasses.dataclass(frozen=True)
class Args:
    repo_id: str = "competition/cup_four_cameras"
    output_path: str = (
        "/root/data1/xxy/openpi/assets/first_task_4cam_dqhand/"
        "competition/cup_four_cameras/dqhand_tokenizer.npz"
    )
    steps: int = 5000
    batch_size: int = 4096
    learning_rate: float = 1e-3
    commitment: float = 0.25
    seed: int = 42
    device: str = "cuda"


def load_actions(repo_id: str) -> np.ndarray:
    dataset = LeRobotDataset(repo_id)
    try:
        actions = np.asarray(dataset.hf_dataset["action"], dtype=np.float32)
    except (AttributeError, KeyError, TypeError, ValueError):
        actions = np.stack([np.asarray(dataset[index]["action"]) for index in range(len(dataset))]).astype(np.float32)
    if actions.ndim != 2 or actions.shape[1] < 26:
        raise ValueError(f"Expected [N, >=26] actions, got {actions.shape}")
    return actions


def train_codec(data: np.ndarray, args: Args, name: str):
    mean = data.mean(axis=0).astype(np.float32)
    std = np.maximum(data.std(axis=0), 1e-4).astype(np.float32)
    normalized = ((data - mean) / std).astype(np.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(normalized)),
        batch_size=min(args.batch_size, len(normalized)),
        shuffle=True,
        drop_last=False,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = ResidualVQAutoencoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    iterator = iter(loader)
    model.train()
    for step in range(args.steps):
        try:
            (batch,) = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            (batch,) = next(iterator)
        batch = batch.to(device)
        reconstruction, latent, quantized, _, _ = model(batch)
        reconstruction_loss = torch.mean((reconstruction - batch) ** 2)
        codebook_loss = torch.mean((quantized - latent.detach()) ** 2)
        commitment_loss = torch.mean((latent - quantized.detach()) ** 2)
        loss = reconstruction_loss + codebook_loss + args.commitment * commitment_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 500 == 0 or step == args.steps - 1:
            print(f"{name} step={step} loss={loss.item():.6f} recon={reconstruction_loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        all_input = torch.from_numpy(normalized).to(device)
        reconstructed, _, _, first, second = model(all_input)
        physical = reconstructed.cpu().numpy() * std + mean
        mae = np.mean(np.abs(physical - data), axis=0)
        combined = (first * model.codebook2.shape[0] + second).cpu().numpy()
        usage = np.bincount(combined, minlength=16)

        codebook1 = model.codebook1.detach().cpu().numpy()
        codebook2 = model.codebook2.detach().cpu().numpy()
        latent_prototypes = np.stack(
            [codebook1[i] + codebook2[j] for i in range(4) for j in range(4)]
        )
        prototype_tensor = torch.from_numpy(latent_prototypes).to(device)
        decoded_prototypes = model.decoder(prototype_tensor).cpu().numpy() * std + mean
        centered = decoded_prototypes - decoded_prototypes.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        score = centered @ vh[0]
        order_to_combined = np.argsort(score).astype(np.int64)
        combined_to_order = np.empty_like(order_to_combined)
        combined_to_order[order_to_combined] = np.arange(16)

    print(f"{name} per-dimension MAE: {mae.tolist()}")
    print(f"{name} code usage: {usage.tolist()} (used={(usage > 0).sum()}/16)")
    if (usage > 0).sum() < 8:
        print(f"WARNING: {name} uses fewer than 8 codes; inspect reconstruction before Pi0.5 training")

    encoder1, encoder2 = model.encoder[0], model.encoder[2]
    decoder1, decoder2 = model.decoder[0], model.decoder[2]
    return {
        "mean": mean,
        "std": std,
        "encoder_w1": encoder1.weight.detach().cpu().numpy(),
        "encoder_b1": encoder1.bias.detach().cpu().numpy(),
        "encoder_w2": encoder2.weight.detach().cpu().numpy(),
        "encoder_b2": encoder2.bias.detach().cpu().numpy(),
        "codebook1": codebook1,
        "codebook2": codebook2,
        "decoder_w1": decoder1.weight.detach().cpu().numpy(),
        "decoder_b1": decoder1.bias.detach().cpu().numpy(),
        "decoder_w2": decoder2.weight.detach().cpu().numpy(),
        "decoder_b2": decoder2.bias.detach().cpu().numpy(),
        "combined_to_order": combined_to_order,
        "order_to_combined": order_to_combined,
        "mae": mae,
        "usage": usage,
    }


def main(args: Args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    actions = load_actions(args.repo_id)
    print(f"Loaded {len(actions)} action frames from {args.repo_id}")
    left = train_codec(actions[:, 14:20], args, "left")
    right = train_codec(actions[:, 20:26], args, "right")
    arrays = {f"left_{key}": value for key, value in left.items()}
    arrays.update({f"right_{key}": value for key, value in right.items()})
    output_path = pathlib.Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **arrays)
    print(f"Saved DQ-Hand tokenizer to {output_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
