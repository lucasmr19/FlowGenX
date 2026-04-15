import torch
import torch.nn as nn
import scipy.linalg

from .base import BaseEvaluator, EvaluationReport, EvaluationResult

class SimpleSequenceEncoder(nn.Module):
    def forward(self, x):
        return x.float().view(x.size(0), -1)


class URSEvaluator(BaseEvaluator):
    def __init__(
        self,
        encoder: nn.Module,
        metric: str = "fid",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(name=f"URS_{metric}", category="fidelity")
        self.metric = metric
        self.device = device
        self.encoder = encoder.to(self.device)
        self.encoder = encoder.eval()

    @torch.no_grad()
    def _embed(self, X: torch.Tensor) -> torch.Tensor:
        return self.encoder(X.to(self.device))

    def _compute_stats(self, Z: torch.Tensor):
        mu = Z.mean(dim=0)
        sigma = torch.cov(Z.T)
        return mu, sigma

    def _fid(self, mu_r, sigma_r, mu_g, sigma_g):
        diff = mu_r - mu_g

        covmean = scipy.linalg.sqrtm((sigma_r @ sigma_g).cpu().numpy())
        covmean = torch.from_numpy(covmean).to(mu_r.device)

        if torch.is_complex(covmean):
            covmean = covmean.real

        return diff @ diff + torch.trace(sigma_r + sigma_g - 2 * covmean)

    def _mmd(self, Zr, Zg, sigma=1.0):
        def k(x, y):
            return torch.exp(-((x - y) ** 2).sum(-1) / (2 * sigma**2))

        K_rr = k(Zr[:, None], Zr[None, :]).mean()
        K_gg = k(Zg[:, None], Zg[None, :]).mean()
        K_rg = k(Zr[:, None], Zg[None, :]).mean()

        return K_rr + K_gg - 2 * K_rg

    def evaluate(self, real, synthetic, **kwargs):
        Zr = self._embed(real)
        Zg = self._embed(synthetic)

        if self.metric == "fid":
            mu_r, sigma_r = self._compute_stats(Zr)
            mu_g, sigma_g = self._compute_stats(Zg)
            score = self._fid(mu_r, sigma_r, mu_g, sigma_g)

        elif self.metric == "mmd":
            score = self._mmd(Zr, Zg)

        else:
            raise ValueError(f"Unknown URS metric: {self.metric}")

        return EvaluationReport(
            evaluator_name=self.name,
            category=self.category,
            results=[
                EvaluationResult(
                    metric_name=self.metric,
                    value=float(score),
                )
            ],
        )