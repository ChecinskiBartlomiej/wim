"""
Exponential Moving Average (EMA) for model parameters.

EMA maintains a moving average of model parameters during training,
which often leads to better and more stable generation quality.

From DDPM paper: "We used EMA on model parameters with a decay factor of 0.9999."
"""

import torch


class EMA:
    """
    Exponential Moving Average of model parameters.

    Args:
        model: PyTorch model to track
        decay: Decay factor for EMA (default: 0.9999 as in DDPM paper)

    The EMA update rule:
        ema_param = decay * ema_param + (1 - decay) * model_param
    """

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters with model's current parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        """
        Update EMA parameters after an optimizer step.

        Call this after optimizer.step() in your training loop.

        Args:
            model: The model being trained
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                # EMA update: new_ema = decay * old_ema + (1 - decay) * current
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data

    def apply_shadow(self, model):
        """
        Replace model parameters with EMA parameters (for evaluation/generation).
        Also saves current parameters to backup for later restoration.

        Use this before generating images or evaluating:
            ema.apply_shadow(model)
            # ... generate images ...
            ema.restore(model)

        Args:
            model: The model to apply EMA to
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        """
        Restore original model parameters after using EMA.

        Args:
            model: The model to restore
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        """
        Return EMA state dictionary for saving checkpoints.
        """
        return {
            'decay': self.decay,
            'shadow': self.shadow
        }

    def load_state_dict(self, state_dict):
        """
        Load EMA state from checkpoint.

        Args:
            state_dict: Dictionary containing EMA state
        """
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']
