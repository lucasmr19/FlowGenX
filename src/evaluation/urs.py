import torch
import torch.nn as nn
from scipy.linalg import sqrtm
import numpy as np

from .base import BaseEvaluator, EvaluationReport, EvaluationResult

class URSEvaluator(BaseEvaluator):
    """
    Unified Representation Similarity Evaluator.

    Calcula simultáneamente:
        - FID (Fréchet distance en espacio latente)
        - MMD (Maximum Mean Discrepancy con kernel RBF)
    """

    def __init__(
        self,
        encoder: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(name="URS", category="fidelity")

        self.device = device
        self.encoder = encoder.to(self.device).eval()

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _embed(self, X: torch.Tensor) -> torch.Tensor:
        return self.encoder(X.to(self.device))

    # ------------------------------------------------------------------
    # Stats (para FID)
    # ------------------------------------------------------------------

    def _compute_stats(self, Z: torch.Tensor):
        Z = Z.float()
        mu = Z.mean(dim=0)
        sigma = torch.cov(Z.T)
        return mu, sigma

    # ------------------------------------------------------------------
    # FID
    # ------------------------------------------------------------------

    def _fid(self, mu_r, sigma_r, mu_g, sigma_g, eps=1e-6):
        mu_r = mu_r.cpu().numpy()
        mu_g = mu_g.cpu().numpy()
        sigma_r = sigma_r.cpu().numpy()
        sigma_g = sigma_g.cpu().numpy()

        # Regularización tipo Tikhonov
        sigma_r = sigma_r + eps * np.eye(sigma_r.shape[0])
        sigma_g = sigma_g + eps * np.eye(sigma_g.shape[0])

        diff = mu_r - mu_g

        covmean, _ = sqrtm(sigma_r @ sigma_g, disp=False)

        # Manejo de complejos por error numérico
        if np.iscomplexobj(covmean):
            if np.max(np.abs(covmean.imag)) > 1e-3:
                return float("nan")
            covmean = covmean.real

        fid = diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)
        return float(np.real(fid))

    # ------------------------------------------------------------------
    # MMD (RBF kernel)
    # ------------------------------------------------------------------

    def _mmd(self, Zr, Zg, sigma=1.0):
        Zr = Zr.float()
        Zg = Zg.float()

        def k(x, y):
            """
            Kernel RBF vectorizado.
            """
            return torch.exp(-((x - y) ** 2).sum(-1) / (2 * sigma**2))

        K_rr = k(Zr[:, None], Zr[None, :]).mean()
        K_gg = k(Zg[:, None], Zg[None, :]).mean()
        K_rg = k(Zr[:, None], Zg[None, :]).mean()

        return float((K_rr + K_gg - 2 * K_rg).item())

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(self, real, synthetic, **kwargs):
        """
        Devuelve simultáneamente FID y MMD.
        """

        Zr = self._embed(real)
        Zg = self._embed(synthetic)

        # --- FID ---
        mu_r, sigma_r = self._compute_stats(Zr)
        mu_g, sigma_g = self._compute_stats(Zg)
        fid_score = self._fid(mu_r, sigma_r, mu_g, sigma_g)

        # --- MMD ---
        mmd_score = self._mmd(Zr, Zg)

        return EvaluationReport(
            evaluator_name=self.name,
            category=self.category,
            results=[
                EvaluationResult(
                    metric_name="fid",
                    value=fid_score,
                ),
                EvaluationResult(
                    metric_name="mmd",
                    value=mmd_score,
                ),
            ],
        )