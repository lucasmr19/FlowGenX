"""
src/reconstruction/build.py
============================
Punto de entrada único para instanciar reconstructores.

Delega completamente en ReconstructorRegistry, que mantiene el mapa
nombre → clase y gestiona aliases. Para añadir un nuevo reconstructor
basta con registrarlo en registry.py (o en tiempo de ejecución); no hay
que tocar este fichero.

Uso típico
----------
    from src.reconstruction.build import build_reconstructor

    reconstructor = build_reconstructor(
        rep_name="gasf",
        model_name="ddpm",
        verbose=True,
    )
    flows = reconstructor.reconstruct(samples_gen, labels=y_gen)

Aliases soportados (ver ReconstructorRegistry.list_available()):
    flat_tokenizer / flat
    semantic_byte  / semantic / protocol_aware
    gasf / gadf / gaf
    nprint_image / nprint-image / netdiffusion
"""

from __future__ import annotations

from typing import Any

from src.reconstruction.registry import ReconstructorRegistry
from src.reconstruction.base import BaseReconstructor


def build_reconstructor(
    rep_name: str,
    model_name: str = "",
    **kwargs: Any,
) -> BaseReconstructor:
    """
    Instancia el reconstructor adecuado para la representación indicada.

    Parameters
    ----------
    rep_name   : nombre o alias de la representación (e.g. "gasf", "flat_tokenizer").
    model_name : nombre del modelo generativo (informativo, se almacena en meta).
    **kwargs   : parámetros adicionales pasados al constructor del reconstructor
                 (e.g. vocab_size, seed, verbose, inter_packet_gap…).

    Returns
    -------
    Instancia de BaseReconstructor lista para llamar a .reconstruct().

    Raises
    ------
    KeyError  si rep_name (ni ningún alias suyo) está registrado.
    """
    return ReconstructorRegistry.get_for_representation(
        representation_name=rep_name,
        model_name=model_name,
        **kwargs,
    )
