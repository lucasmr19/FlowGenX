from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union, TYPE_CHECKING

from torch import Tensor
from torch.utils.data import Dataset

from ..preprocessing import TrafficSample
from ..utils.logger_config import LOGGER

if TYPE_CHECKING:
    from ..representations.base import TrafficRepresentation

class RepresentationDataset(Dataset):
    """
    Dataset que aplica una TrafficRepresentation en __getitem__.

    La representación debe haber sido ajustada (fit()) antes de pasarla.
    El resultado de encode() se devuelve como tensor.

    Parameters
    ----------
    samples        : lista de TrafficSample (Flow | PacketWindow)
    representation : TrafficRepresentation ya ajustada
    label_fn       : callable opcional TrafficSample → int
    cache          : si True, precalcula y cachea todos los tensores en RAM
    """

    def __init__(
        self,
        samples:          List[TrafficSample],
        representation: "TrafficRepresentation",
        label_fn:       Optional[Callable[[TrafficSample], int]] = None,
        cache:          bool = False,
    ) -> None:
        self.samples        = samples
        self.representation = representation
        self.label_fn       = label_fn
        self._cache: Optional[List[Tensor]] = None

        if cache:
            LOGGER.info("Precalculando representaciones en caché (%d muestras)...", len(samples))
            self._cache = [representation.encode(s) for s in samples]
            LOGGER.info("Caché completada.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Union[Tensor, Tuple[Tensor, int]]:
        tensor = (
            self._cache[idx]
            if self._cache is not None
            else self.representation.encode(self.samples[idx])
        )
        if self.label_fn is not None:
            return tensor, self.label_fn(self.samples[idx])
        return tensor

    def get_sample(self, idx: int) -> TrafficSample:
        return self.samples[idx]
