import torch
from torch import nn
from typing import Dict, List, Optional, Callable, Union

class LossAgreggator(nn.Module):
    def __init__(self, params, num_losses, weights=None):
        super().__init__()

        self.params: List[torch.Tensor] = list(params)
        self.num_losses: int = num_losses
        self.weights: Optional[Dict[str, float]] = weights
        self.device: torch.device
        self.device = list(set(p.device for p in self.params))[0]
        self.init_loss: torch.Tensor = torch.tensor(0.0, device=self.device)
        
        def weigh_losses_initialize(weights: Optional[Dict[str, float]]) -> Callable[
            [Dict[str, torch.Tensor], 
             Optional[Dict[str, float]]], 
             Dict[str, torch.Tensor],
        ]:
            
            if weights is None:

                def weigh_losses(losses: Dict[str, torch.Tensor], weights: None) -> Dict[str, torch.Tensor]:
                    return losses

            else:

                def weigh_losses(losses: Dict[str, torch.Tensor], weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
                    
                    for key in losses.keys():
                        if key not in weights.keys():
                            weights.update({key: 1.0})
                    
                    losses = {key: weights[key] * losses[key] for key in losses.keys()}
                    
                    return losses

            return weigh_losses
        
        self.weigh_losses = weigh_losses_initialize(self.weights)

class Relobralo(LossAgreggator):
    """
    Relative loss balancing with random lookback
    Reference: "Bischof, R. and Kraus, M., 2021.
    Multi-Objective Loss Balancing for Physics-Informed Deep Learning.
    arXiv preprint arXiv:2110.09813."
    """

    def __init__(
        # self, params, num_losses, alpha=0.999, beta=1.0, tau=0.01, eps=1e-12, weights=None):
        self, params, num_losses, alpha=0.9, beta=1.0, tau=0.01, eps=1e-12, weights=None):
        super().__init__(params, num_losses, weights)
        
        self.alpha: float = alpha
        self.beta: float = beta
        self.tau: float = tau
        self.eps: float = eps

        self.register_buffer(
            "init_losses", torch.zeros(self.num_losses, device=self.device)
        )
        self.register_buffer(
            "prev_losses", torch.zeros(self.num_losses, device=self.device)
        )
        self.register_buffer(
            "lmbda_ema", torch.ones(self.num_losses, device=self.device)
        )

    def forward(self, losses: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        """
        Weights and aggregates the losses using the ReLoBRaLo algorithm

        Parameters
        ----------
        losses : Dict[str, torch.Tensor]
            A dictionary of losses.
        step : int
            Optimizer step.

        Returns
        -------
        loss : torch.Tensor
            Aggregated loss.
        """

        # weigh losses
        losses = self.weigh_losses(losses, self.weights)

        # Initialize loss
        loss: torch.Tensor = torch.zeros_like(self.init_loss)

        # Aggregate losses by summation at step 0
        if step == 0:
            for i, key in enumerate(losses.keys()):
                loss += losses[key]
                self.init_losses[i] = losses[key].clone().detach()
                self.prev_losses[i] = losses[key].clone().detach()

        # Aggregate losses using ReLoBRaLo for step > 0
        else:
            losses_stacked: torch.Tensor = torch.stack(list(losses.values()))
            normalizer_prev: torch.Tensor = (
                losses_stacked / (self.tau * self.prev_losses + self.eps)
            ).max()
            normalizer_init: torch.Tensor = (
                losses_stacked / (self.tau * self.init_losses + self.eps)
            ).max()
            rho: torch.Tensor = torch.bernoulli(torch.tensor(self.beta))
            with torch.no_grad():
                lmbda_prev: torch.Tensor = torch.exp(
                    losses_stacked / (self.tau * self.prev_losses + self.eps)
                    - normalizer_prev
                )
                lmbda_init: torch.Tensor = torch.exp(
                    losses_stacked / (self.tau * self.init_losses + self.eps)
                    - normalizer_init
                )
                lmbda_prev *= self.num_losses / (lmbda_prev.sum())
                lmbda_init *= self.num_losses / (lmbda_init.sum())

            # Compute the exponential moving average of weights and aggregate losses
            for i, key in enumerate(losses.keys()):
                with torch.no_grad():
                    self.lmbda_ema[i] = self.alpha * (
                        rho * self.lmbda_ema[i].clone() + (1.0 - rho) * lmbda_init[i]
                    )
                    self.lmbda_ema[i] += (1.0 - self.alpha) * lmbda_prev[i]
                
                loss += self.lmbda_ema[i].clone() * losses[key]
                self.prev_losses[i] = losses[key].clone().detach()

        return loss     
        
