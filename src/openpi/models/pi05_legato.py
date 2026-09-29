"""π₀.₅ + Legato continuation.

Based on pi0.py, modified to support Legato-style action chunk continuation.
Key additions over vanilla π₀.₅:
- `compute_loss`: kappa-corrected flow matching with per-sample warmup/weight curve
- `sample_actions`: guided sampling that blends previous action chunk with current flow
- `embed_suffix`: accepts an extra weight channel per action token

Convention: t=1 → noise, t=0 → action (same as π₀.₅, opposite of Legato).
"""

import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


# ──────────────────────────────────────────────
# Legato utility functions
# ──────────────────────────────────────────────


def _sample_bell_curve(rng: jax.Array, min_val: int, max_val: int, peak: int = 0) -> int:
    """Sample an integer in [min_val, max_val] biased toward `peak` using a beta distribution."""

    range_size = max_val - min_val

    def sample_from_beta():
        normalized_peak = (peak - min_val) / jnp.maximum(range_size, 1.0)
        concentration = 3.0
        alpha_left = concentration * normalized_peak + 1
        beta_left = concentration * (1 - normalized_peak) + 1
        alpha_right = concentration * (1.0 - normalized_peak) + 1.0
        beta_right = concentration * normalized_peak + 1.0

        alpha = jnp.where(normalized_peak < 0.5, alpha_left, alpha_right)
        beta = jnp.where(normalized_peak < 0.5, beta_left, beta_right)

        beta_sample = jax.random.beta(rng, alpha, beta)
        continuous_value = min_val + beta_sample * (range_size + 1)
        sampled_value = jnp.floor(continuous_value).astype(jnp.int32)
        sampled_value = jnp.clip(sampled_value, min_val, max_val)
        return sampled_value

    return jnp.where(range_size == 0, min_val, sample_from_beta())


def _sample_exp_decay(rng: jax.Array, min_val: int, max_val: int) -> int:
    """Sample an integer in [min_val, max_val] with exponential decay toward min_val."""

    range_size = max_val - min_val

    def sample_from_exp():
        candidates = jnp.arange(min_val, max_val + 1)
        weights = jnp.exp(-(candidates - min_val).astype(jnp.float32))
        probs = weights / jnp.sum(weights)
        return jax.random.choice(rng, candidates, p=probs)

    return jnp.where(range_size == 0, min_val, sample_from_exp())


def get_prefix_weights(
    start: int,
    end: int,
    total: int,
    schedule: str,
) -> jax.Array:
    """Build RTC-style reference guidance strength.

    Returns:
        omega: shape [total]

        omega=1:
            completely follow the reference action.

        omega=0:
            completely free generation.

        0 < omega < 1:
            ramp transition between reference and free generation.
    """
    start = jnp.minimum(start, end)
    indices = jnp.arange(total)

    if schedule == "ones":
        omega = jnp.ones(total, dtype=jnp.float32)

    elif schedule == "zeros":
        omega = (indices < start).astype(jnp.float32)

    elif schedule in ("linear", "exp"):
        omega = jnp.clip(
            (start - 1 - indices) / (end - start + 1) + 1,
            0.0,
            1.0,
        ).astype(jnp.float32)

        if schedule == "exp":
            omega = omega * jnp.expm1(omega) / (jnp.e - 1.0)

    else:
        raise ValueError(
            f"Invalid ramp schedule: {schedule!r}. Expected one of "
            "{'linear', 'exp', 'ones', 'zeros'}."
        )

    return jnp.where(indices >= end, 0.0, omega)


def build_weight_curve(
    rng: jax.Array,
    action_horizon: int,
    warmup_min: int,
    warmup_max: int,
    warmup_sampling: str,
    prefix_attention_horizon: int,
    ramp_schedule: str,
    warmup_peak: int | None = None
) -> jax.Array:
    """Build per-token Legato continuation weight curve with RTC-style ramp.

    Returns w with shape [action_horizon].
    - w = 0: this position copies previous action (prefix, no flow).
    - w = 1: this position is generated via normal flow matching.

    w is derived from RTC guidance strength omega via w = 1 - omega.
    """
    rng, key_warmup = jax.random.split(rng, 2)
    if warmup_sampling == "bell":
        warmup = _sample_bell_curve(key_warmup, warmup_min, warmup_max, peak=warmup_peak if warmup_peak is not None else warmup_min)
    elif warmup_sampling == "exp":
        warmup = _sample_exp_decay(key_warmup, warmup_min, warmup_max)
    else:
        warmup = jax.random.randint(key_warmup, (), warmup_min, warmup_max + 1)

    omega = get_prefix_weights(
        start=warmup,
        end=prefix_attention_horizon,
        total=action_horizon,
        schedule=ramp_schedule,
    )

    return 1.0 - omega


# ──────────────────────────────────────────────
# Legato continuation parameters
# Modify these directly to tune Legato behavior.
# ──────────────────────────────────────────────

_warmup_min: int = 2
_warmup_max: int = 4
_warmup_peak: int = 3
_warmup_sampling: str = "bell"
_inference_num_steps: int = 10
_prefix_attention_horizon: int = 30
_ramp_schedule: str = "exp"


# ──────────────────────────────────────────────
# Pi0Legato model
# ──────────────────────────────────────────────


class Pi05Legato(_model.BaseModel):
    """π₀.₅ with Legato action-chunk continuation.

    Same architecture as π₀.₅ (PaliGemma prefix + Gemma action expert with adaRMS),
    but the action head accepts an extra weight channel per token and the training /
    inference procedures use Legato-style guided flow matching.
    """

    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # Legato continuation parameters
        # Read from config if present, otherwise use module-level defaults.
        self.warmup_min = getattr(config, "warmup_min", _warmup_min)
        self.warmup_max = getattr(config, "warmup_max", _warmup_max)
        self.warmup_peak = getattr(config, "warmup_peak", _warmup_peak)
        self.warmup_sampling = getattr(config, "warmup_sampling", _warmup_sampling)
        self.inference_num_steps = getattr(config, "inference_num_steps", _inference_num_steps)
        self.prefix_attention_horizon = getattr(config, "prefix_attention_horizon", _prefix_attention_horizon)
        self.ramp_schedule = getattr(config, "ramp_schedule", _ramp_schedule)

        if not (
            0
            <= self.warmup_min
            <= self.warmup_max
            <= self.prefix_attention_horizon
            <= self.action_horizon
        ):
            raise ValueError(
                "Expected 0 <= warmup_min <= warmup_max "
                "<= prefix_attention_horizon <= action_horizon, but got "
                f"warmup_min={self.warmup_min}, "
                f"warmup_max={self.warmup_max}, "
                f"prefix_attention_horizon={self.prefix_attention_horizon}, "
                f"action_horizon={self.action_horizon}."
            )

        if self.ramp_schedule not in {"linear", "exp", "ones", "zeros"}:
            raise ValueError(
                f"Invalid ramp_schedule: {self.ramp_schedule!r}."
            )

        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.legato_weight_proj = nnx.Linear(1, action_expert_config.width, use_bias=False, kernel_init=jax.nn.initializers.zeros, rngs=rngs)

        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []

        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1])
            )
            ar_mask += [False] * image_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: at.Float[at.Array, "b ah ad"],
        timestep: at.Float[at.Array, " b"],
        *,
        legato_weight: at.Float[at.Array, "b ah 1"] | None = None,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []

        # Legato: append weight channel. Default to all-ones (vanilla flow).
        if legato_weight is None:
            legato_weight = jnp.ones((*noisy_actions.shape[:-1], 1), dtype=noisy_actions.dtype)

        if not self.pi05:
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        action_tokens = action_tokens + self.legato_weight_proj(legato_weight)
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)

        if self.pi05:
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None

        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng, weight_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_size = actions.shape[0]

        # ── Legato: sample warmup / weight curve per sample ──
        def _build_weight(rng_i):
            return build_weight_curve(
                rng=rng_i,
                action_horizon=self.action_horizon,
                warmup_min=self.warmup_min,
                warmup_max=self.warmup_max,
                warmup_sampling=self.warmup_sampling,
                prefix_attention_horizon=self.prefix_attention_horizon,
                ramp_schedule=self.ramp_schedule,
                warmup_peak=self.warmup_peak
            )

        weight_rngs = jax.random.split(weight_rng, batch_size)
        w = jax.vmap(_build_weight)(weight_rngs)  # [B, H]
        w = w[:, :, None]  # [B, H, 1]

        # ── Flow matching with Legato kappa correction ──
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, (batch_size,)) * 0.999 + 0.001
        time_expanded = time[..., None, None]  # [B, 1, 1]

        # Standard π₀.₅ interpolation: t=1 → noise, t=0 → action
        x_t = time_expanded * noise + (1 - time_expanded) * actions

        # Legato: blend previous action (ground truth) into prefix positions
        guided_x_t = (1.0 - w) * actions + w * x_t

        # Legato kappa correction: reshapes the velocity field so that training is
        # consistent with per-step, schedule-shaped guidance at inference time.
        # omega = 1 - w  (omega=1: full guidance, omega=0: free generation)
        # dt_abs = 1 / N (always positive)
        omega = 1.0 - w
        dt_abs = 1.0 / float(self.inference_num_steps)
        kappa = omega / dt_abs
        # π₀.₅ convention: time flows from 1→0.
        # v_target = [1 - κ * t] * (noise - actions)
        u_t = (1.0 - kappa * time_expanded) * (noise - actions)

        # ── Forward pass ──
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, guided_x_t, time, legato_weight=w
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] | None = None,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        rtc_args: dict | None = None,
    ) -> _model.Actions:
        """Sample actions with Legato continuation.

        Args:
            rng: JAX random key.
            observation: Model inputs (images, state, prompt).
            num_steps: Number of Euler integration steps.
            noise: Optional initial noise. If None, sampled from N(0, 1).
            rtc_args: RTC-style dict with keys ``prev_action_chunk``,
                ``inference_delay``, ``prefix_attention_horizon``.
                If None or if ``prev_action_chunk`` is None, falls back to
                standard (no-prefix) flow sampling.
        """
        observation = _model.preprocess_observation(None, observation, train=False)

        if num_steps is None:
            num_steps = self.inference_num_steps
        elif isinstance(num_steps, int) and num_steps != self.inference_num_steps:
            raise ValueError(
                "Legato requires training and inference denoising steps to match: "
                f"trained with {self.inference_num_steps}, but got {num_steps}."
            )

        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # ── Read rtc_args ──
        if rtc_args is not None:
            prev_action_chunk = rtc_args["prev_action_chunk"]
            inference_delay = rtc_args["inference_delay"]
            prefix_attention_horizon = rtc_args["prefix_attention_horizon"]

            if not (0 <= inference_delay <= prefix_attention_horizon <= self.action_horizon):
                raise ValueError(
                    "Expected 0 <= inference_delay <= prefix_attention_horizon "
                    f"<= action_horizon, but got inference_delay={inference_delay}, "
                    f"prefix_attention_horizon={prefix_attention_horizon}, "
                    f"action_horizon={self.action_horizon}."
                )
        else:
            prev_action_chunk = None

        # ── Pad prev_action_chunk to match action_horizon (reuses RTC logic) ──
        has_prev = prev_action_chunk is not None
        if has_prev:
            prev = jnp.asarray(prev_action_chunk)
            if prev.ndim == 1:
                prev = prev[None, :]  # [ad] → [1, ad]
            old_length = prev.shape[-2]
            new_length = self.action_horizon
            if old_length != new_length:
                pad_length = new_length - old_length
                # PADLAST: repeat the last action to fill the remaining horizon.
                last_action = prev[..., -1:, :]
                prev = jnp.concatenate([prev, jnp.repeat(last_action, pad_length, axis=-2)], axis=-2)
            if prev.ndim == 2:
                prev = prev[None, :, :]  # [H, ad] → [1, H, ad]
        else:
            prev = jnp.zeros_like(noise)

        # ── Legato ramp weight curve ──
        if has_prev:
            ramp_schedule = rtc_args.get("ramp_schedule", self.ramp_schedule)
            omega = get_prefix_weights(
                start=inference_delay,
                end=prefix_attention_horizon,
                total=self.action_horizon,
                schedule=ramp_schedule,
            ).astype(noise.dtype)
            w_infer = (1.0 - omega)[None, :, None]  # [1, H, 1]
        else:
            w_infer = jnp.ones((1, self.action_horizon, 1), dtype=noise.dtype)

        # ── KV cache: one forward pass over the prefix ──
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry

            # Legato: blend previous chunk into prefix positions
            guided_x_t = (1.0 - w_infer) * prev + w_infer * x_t

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, guided_x_t, jnp.broadcast_to(time, batch_size), legato_weight=w_infer
            )

            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return guided_x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
