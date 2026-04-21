from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple, Union

from torch.utils.data import Dataset

from ..preprocessing import TrafficSample

    
# ---------------------------------------------------------------------------
# Dataset base — trabaja sobre flujos ya parseados
# ---------------------------------------------------------------------------

class TrafficDataset(Dataset):
    """
    Dataset PyTorch sobre una lista de objetos TrafficSample.

    Opcionalmente aplica un transform (callable) que convierte
    un TrafficSample en cualquier estructura (tensor, dict, etc.).

    Parameters
    ----------
    samples     : lista de TrafficSample producida por PCAPPipeline.
    transform : callable opcional TrafficSample → Any (p.ej. una TrafficRepresentation)
    label_fn  : callable opcional TrafficSample → int para clasificación downstream
    """

    def __init__(
        self,
        samples:     List[TrafficSample],
        transform: Optional[Callable[[TrafficSample], Any]] = None,
        label_fn:  Optional[Callable[[TrafficSample], int]]  = None,
    ) -> None:
        self.samples     = samples
        self.transform = transform
        self.label_fn  = label_fn

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Union[Any, Tuple[Any, int]]:
        sample = self.samples[idx]
        item = self.transform(sample) if self.transform else sample
        if self.label_fn is not None:
            return item, self.label_fn(sample)
        return item

    def get_sample(self, idx: int) -> TrafficSample:
        """Acceso directo al TrafficSample subyacente (sin transform)."""
        return self.samples[idx]