"""Feed-forward (memory-free) counterpart to SquashedMlpLstmPolicy.

ABLATION RATIONALE
------------------
The earlier no-LSTM ablation zeroed the LSTM hidden state at every step. That
leaves a *handicapped recurrent network* -- the recurrent weights W_hh still
exist, still receive gradient, and still contribute parameters, but can never
carry information. It is a conservative test (it biases against the LSTM), but a
reviewer can legitimately object that it does not measure "recurrent vs
feed-forward"; it measures "recurrent vs crippled recurrent".

This policy is the clean comparison. The LSTM cell is replaced by a
`nn.Linear(input_size, hidden_size)` followed by the same activation, so:

  * the algorithm is identical (TimeAwareRecurrentPPOv2, untouched),
  * the time-aware discount and GAE are identical,
  * every downstream dimension is identical (the extractor still sees a
    `hidden_size`-wide vector, `net_arch` is unchanged),
  * the rollout-buffer machinery is identical -- state tensors of the same shape
    are still allocated and carried, they simply have no effect,

and the ONLY difference is that no information crosses a timestep boundary.

The paper can therefore state, without qualification: *"identical algorithm,
identical discounting, identical network widths; the recurrent cell was replaced
by a feed-forward layer of the same output width."*

Parameter counts differ (an LSTM cell has ~4x the weights of a linear layer of
the same width). `build_model` logs both counts so the difference is reported
rather than hidden. Matching parameter counts instead of widths would require
changing `net_arch`, which would confound the comparison with a capacity change
elsewhere in the network -- width-matching is the cleaner control.
"""
from __future__ import annotations

from typing import Any

import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from custom_rl.ppo_recurrent.squashed_recurrent_policy import SquashedMlpLstmPolicy


class FeedForwardCell(nn.Module):
    """Drop-in replacement for `nn.LSTM` that carries no state.

    Exposes `input_size`, `hidden_size` and `num_layers` so code that
    introspects the cell keeps working -- `_process_sequence` reshapes using
    `lstm.input_size`, and `TimeAwareRecurrentPPOv2` sizes its state tensors
    from `lstm.num_layers` and `lstm.hidden_size`.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        activation_fn: type[nn.Module] = nn.Tanh,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        # Kept so the state tensors allocated by the algorithm have the same
        # shape as in the recurrent case (they are carried but never used).
        self.num_layers = int(num_layers)
        self.linear = nn.Linear(self.input_size, self.hidden_size)
        self.activation = activation_fn()

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.activation(self.linear(x))


class SquashedMlpPolicy(SquashedMlpLstmPolicy):
    """Memory-free policy with the same interface as SquashedMlpLstmPolicy.

    State tensors are still accepted and returned (unchanged) so that
    `TimeAwareRecurrentPPOv2`, the recurrent rollout buffers, and
    `model.predict(state=..., episode_start=...)` all work without modification.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[nn.Module] = FlattenExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            lstm_hidden_size=lstm_hidden_size,
            n_lstm_layers=n_lstm_layers,
            shared_lstm=shared_lstm,
            enable_critic_lstm=enable_critic_lstm,
            lstm_kwargs=lstm_kwargs,
        )

        # Swap the recurrent cells for width-matched feed-forward cells.
        # `lstm_actor` always exists; `lstm_critic` exists only when
        # enable_critic_lstm=True and shared_lstm=False.
        self.lstm_actor = FeedForwardCell(
            self.lstm_actor.input_size,
            self.lstm_actor.hidden_size,
            num_layers=self.lstm_actor.num_layers,
            activation_fn=activation_fn,
        )
        if getattr(self, "lstm_critic", None) is not None:
            self.lstm_critic = FeedForwardCell(
                self.lstm_critic.input_size,
                self.lstm_critic.hidden_size,
                num_layers=self.lstm_critic.num_layers,
                activation_fn=activation_fn,
            )

        # Rebuild the optimizer so it tracks the new parameters and drops the
        # discarded LSTM weights (otherwise Adam would hold stale param refs).
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    # ------------------------------------------------------------------
    # Recurrence bypass
    # ------------------------------------------------------------------
    # Base class defines this as a @staticmethod called as
    #   self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)
    # Overriding it as an instance method keeps every call site working.
    def _process_sequence(  # type: ignore[override]
        self,
        features: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        lstm: nn.Module,
    ) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        """Apply the cell independently per timestep; pass states through untouched.

        `episode_starts` is deliberately ignored: with no state to reset, episode
        boundaries have no effect on the forward pass. States are returned
        unchanged so the buffer keeps seeing tensors of the expected shape.
        """
        return lstm(features), lstm_states
