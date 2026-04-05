import torch

from src.models_ml.base import (
    GenerativeModel,
    GenerativeModelConfig,
    ModelType,
    InputDomain,
)


class DummyModel(GenerativeModel):
    """
    Minimal implementation of GenerativeModel used to validate
    the base contract (train_step output, device handling, generation API).
    """

    def __init__(self):
        config = GenerativeModelConfig(name="dummy")
        super().__init__(config)

    @property
    def model_type(self):
        return ModelType.AUTOREGRESSIVE

    @property
    def input_domain(self):
        return InputDomain.CONTINUOUS_SEQUENCE

    def build(self):

        self.linear = torch.nn.Linear(4, 1)

        self._networks = {
            "linear": self.linear,
        }

        # move to configured device
        self.to(self.device)

        self._built = True
        return self

    def forward(self, x):
        return self.linear(x)

    def train_step(self, batch):

        x = batch.to(self.device)

        y = self.forward(x)

        loss = y.mean()

        return {
            "loss": loss,
            "dummy_metric": loss.detach(),
        }

    def generate(self, n_samples: int, **kwargs):
        return torch.zeros(n_samples, 4, device=self.device)


def test_train_step_returns_dict():
    model = DummyModel().build()

    batch = torch.randn(8, 4)

    out = model.train_step(batch)

    assert isinstance(out, dict)
    assert "loss" in out


def test_loss_is_tensor():
    model = DummyModel().build()

    batch = torch.randn(8, 4)

    out = model.train_step(batch)

    assert torch.is_tensor(out["loss"])
    assert out["loss"].ndim == 0


def test_generate_shape():
    model = DummyModel().build()

    samples = model.generate(5)

    assert samples.shape[0] == 5
    assert samples.shape[1] == 4