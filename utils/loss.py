from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegLoss(nn.Module):
    """Copy from: https://github.com/jolibrain/vicreg-loss/blob/master/vicreg_loss/vicreg.py
    """
    def __init__(
        self,
        inv_coeff: float = 25.0,
        var_coeff: float = 15.0,
        cov_coeff: float = 1.0,
        gamma: float = 1.0,
        details: bool = False
    ):
        super().__init__()
        self.inv_coeff = inv_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma
        self.details = details

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Computes the VICReg loss.

        ---
        Args:
            x: Features map.
                Shape of [batch_size, representation_size].
            y: Features map.
                Shape of [batch_size, representation_size].

        ---
        Returns:
            The VICReg loss.
                Dictionary where values are of shape of [1,].
        """
        metrics = dict()
        metrics["inv-loss"] = self.inv_coeff * self.representation_loss(x, y)
        metrics["var-loss"] = (
            self.var_coeff
            * (self.variance_loss(x, self.gamma) + self.variance_loss(y, self.gamma))
            / 2
        )
        metrics["cov-loss"] = (
            self.cov_coeff * (self.covariance_loss(x) + self.covariance_loss(y)) / 2
        )
        metrics["loss"] = sum(metrics.values())
        
        return metrics if self.details else metrics["loss"]

    @staticmethod
    def representation_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the representation loss.
        Force the representations of the same object to be similar.

        ---
        Args:
            x: Features map.
                Shape of [batch_size, representation_size].
            y: Features map.
                Shape of [batch_size, representation_size].

        ---
        Returns:
            The representation loss.
                Shape of [1,].
        """
        return F.mse_loss(x, y)

    @staticmethod
    def variance_loss(x: torch.Tensor, gamma: float) -> torch.Tensor:
        """Computes the variance loss.
        Push the representations across the batch
        to be different between each other.
        Avoid the model to collapse to a single point.

        The gamma parameter is used as a threshold so that
        the model is no longer penalized if its std is above
        that threshold.

        ---
        Args:
            x: Features map.
                Shape of [batch_size, representation_size].

        ---
        Returns:
            The variance loss.
                Shape of [1,].
        """
        x = x - x.mean(dim=0)
        std = x.std(dim=0)
        var_loss = F.relu(gamma - std).mean()
        return var_loss

    @staticmethod
    def covariance_loss(x: torch.Tensor) -> torch.Tensor:
        """Computes the covariance loss.
        Decorrelates the embeddings' dimensions, which pushes
        the model to capture more information per dimension.

        ---
        Args:
            x: Features map.
                Shape of [batch_size, representation_size].

        ---
        Returns:
            The covariance loss.
                Shape of [1,].
        """
        x = x - x.mean(dim=0)
        cov = (x.T @ x) / (x.shape[0] - 1)
        cov_loss = cov.fill_diagonal_(0.0).pow(2).sum() / x.shape[1]
        return cov_loss
    

class InfoNCELoss(nn.Module):
    """ SimCLR loss @SimCLR
    Adapted from:
    https://github.com/ysharma1126/ssl_identifiability/blob/master/main_3dident.py
    """
    def __init__(self, tau: float = 0.5) -> None:
        super().__init__()
        self._tau = tau
        assert self._tau != 0
        self._metric = nn.CosineSimilarity(dim=-1)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sim_xx = self._metric(x.unsqueeze(-2), x.unsqueeze(-3)) / self._tau
        sim_yy = self._metric(y.unsqueeze(-2), y.unsqueeze(-3)) / self._tau
        sim_xy = self._metric(x.unsqueeze(-2), y.unsqueeze(-3)) / self._tau

        n = sim_xy.shape[-1]
        sim_xx[..., range(n), range(n)] = float("-inf")
        sim_yy[..., range(n), range(n)] = float("-inf")
        scores1 = torch.cat([sim_xy, sim_xx], dim=-1)    
        scores2 = torch.cat([sim_yy, sim_xy.transpose(-1,-2)], dim=-1)     
        scores = torch.cat([scores1, scores2], dim=-2)  
        targets = torch.arange(2 * n, dtype=torch.long, device=scores.device)
        total_loss = self.criterion(scores, targets)
        return total_loss


class InfoNCELossBasic(nn.Module):
    def __init__(self, tau: float = 0.5) -> None:
        super().__init__()
        self._tau = tau
        assert self._tau != 0
        self._metric = nn.CosineSimilarity(dim=-1)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        scores = self._metric(x.unsqueeze(-2), y.unsqueeze(-3)) / self._tau
        n = scores.shape[-1]
        targets = torch.arange(n, dtype=torch.long, device=scores.device)
        total_loss = self.criterion(scores, targets)
        return total_loss
    

class OrthogonalLoss(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self._metric = nn.CosineSimilarity(dim=-1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sim = self._metric(x.unsqueeze(-2), x.unsqueeze(-3))
        n = sim.shape[-1]
        if n == 1:
            return 0
        loss =torch.sum(torch.abs(sim - torch.eye(n).type(torch.float32).to(sim.device))) / (n*n -n)
        return loss


class SequentialOrthogonalLoss(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self._metric = nn.CosineSimilarity(dim=-1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        if n == 1:
            return 0
        
        loss = 0
        for i in range(0, n-1):
            loss += torch.abs(self._metric(x[n-1], x[i]))
        return loss / (n-1)


class SymmetricLoss(nn.Module):
    """ Symmetric Contrastive loss @CLIP
    """
    def __init__(self, tau: float = 0.5) -> None:
        super().__init__()
        self._tau = tau
        assert self._tau != 0
        self._metric = nn.CosineSimilarity(dim=-1)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sim_xy = self._metric(x.unsqueeze(-2), y.unsqueeze(-3)) / self._tau
        sim_yx = self._metric(y.unsqueeze(-2), x.unsqueeze(-3)) / self._tau
        n = sim_xy.shape[-1]
        targets = torch.arange(n, dtype=torch.long, device=sim_xy.device)
        total_loss = (self.criterion(sim_xy, targets) + self.criterion(sim_yx, targets))/2.
        return total_loss
