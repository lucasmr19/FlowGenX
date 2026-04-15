"""
src/reconstruction/registry.py
================================
Fábrica de reconstructores con resolución de aliases.

Uso típico
----------
    # En experiment_builder.py, sin ningún if/elif de representaciones:
    reconstructor = ReconstructorRegistry.get_for_representation(
        representation=rep,          # instancia de TrafficRepresentation
        representation_name="gasf",  # o el nombre del config
        model_name="ddpm",
        verbose=True,
    )
    flows = reconstructor.reconstruct(samples_gen, labels=y_gen)

    # También permite registro de reconstructores personalizados:
    ReconstructorRegistry.register("my_rep", MyCustomReconstructor)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type

from src.reconstruction.base import BaseReconstructor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Aliases canónicos
# Cada clave es un nombre de representación (como aparece en los configs/tests).
# Cada valor es la clase de reconstructor correspondiente.
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Type[BaseReconstructor]] = {}


def _lazy_register_defaults() -> None:
    """
    Registro diferido para evitar imports circulares en el arranque.
    Se ejecuta en el primer acceso al registry.
    """
    from src.reconstruction.sequential import (
        FlatTokenizerReconstructor,
        SemanticByteReconstructor,
        ProtocolAwareReconstructor,
    )
    from src.reconstruction.vision import (
        GAFReconstructor,
        NprintImageReconstructor,
    )

    defaults = {
        # Sequential
        "flat_tokenizer":   FlatTokenizerReconstructor,
        "flat":             FlatTokenizerReconstructor,  # alias
        "semantic_byte":    SemanticByteReconstructor,
        "semantic":         SemanticByteReconstructor,   # alias
        "protocol_aware":   ProtocolAwareReconstructor,   # heurísticas similares
        # Visual
        "gasf":             GAFReconstructor,
        "gadf":             GAFReconstructor,            # alias (mismo pipeline)
        "gaf":              GAFReconstructor,            # alias
        "nprint_image":     NprintImageReconstructor,
        "netdiffusion":     NprintImageReconstructor,    # alias semántico
    }

    for name, cls in defaults.items():
        if name not in _REGISTRY:
            _REGISTRY[name] = cls


class ReconstructorRegistry:
    """
    Fábrica estática de reconstructores.

    Métodos
    -------
    get(name, **kwargs)
        Instancia un reconstructor por nombre.

    get_for_representation(representation, representation_name, **kwargs)
        Resuelve el reconstructor adecuado a partir de una instancia
        de TrafficRepresentation o directamente de su nombre de cadena.

    register(name, cls)
        Registra un reconstructor personalizado.

    list_available()
        Devuelve los nombres registrados.
    """

    @classmethod
    def _ensure_registered(cls) -> None:
        if not _REGISTRY:
            _lazy_register_defaults()

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    @classmethod
    def get(cls, name: str, **kwargs: Any) -> BaseReconstructor:
        """
        Instancia el reconstructor registrado bajo `name`.

        Parameters
        ----------
        name   : nombre o alias de la representación.
        kwargs : argumentos pasados al constructor del reconstructor.

        Raises
        ------
        KeyError si el nombre no está registrado.
        """
        cls._ensure_registered()
        key = name.lower().strip()
        if key not in _REGISTRY:
            available = ", ".join(sorted(_REGISTRY.keys()))
            raise KeyError(
                f"Reconstructor '{name}' no registrado. "
                f"Disponibles: {available}"
            )
        reconstructor_cls = _REGISTRY[key]
        logger.debug("[Registry] Instanciando %s para '%s'", reconstructor_cls.__name__, name)
        return reconstructor_cls(**kwargs)

    @classmethod
    def get_for_representation(
        cls,
        representation: Any = None,
        representation_name: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseReconstructor:
        """
        Resuelve el reconstructor adecuado para una representación dada.

        Acepta:
          - Una instancia de TrafficRepresentation (usa repr.__class__.__name__).
          - Un string con el nombre de la representación.
          - Ambos a la vez (representation_name tiene prioridad).

        Parameters
        ----------
        representation      : instancia de TrafficRepresentation (opcional).
        representation_name : nombre explícito (opcional, tiene prioridad).
        kwargs              : argumentos adicionales para el constructor.

        Returns
        -------
        Instancia de BaseReconstructor lista para usar.
        """
        cls._ensure_registered()

        # Resolver el nombre a usar para la búsqueda
        if representation_name is not None:
            name = representation_name
        elif representation is not None:
            name = cls._name_from_instance(representation)
        else:
            raise ValueError(
                "Se requiere 'representation' o 'representation_name'."
            )

        # Inyectar representation_name en kwargs si no está ya
        kwargs.setdefault("representation_name", name)

        return cls.get(name, **kwargs)

    @classmethod
    def register(cls, name: str, reconstructor_cls: Type[BaseReconstructor]) -> None:
        """
        Registra un reconstructor personalizado.

        Permite extender el registry sin modificar este archivo:

            from src.reconstruction.registry import ReconstructorRegistry
            ReconstructorRegistry.register("my_repr", MyReconstructor)
        """
        cls._ensure_registered()
        _REGISTRY[name.lower().strip()] = reconstructor_cls
        logger.info("[Registry] Registrado reconstructor '%s' → %s", name, reconstructor_cls.__name__)

    @classmethod
    def list_available(cls) -> list[str]:
        """Devuelve la lista de nombres/aliases registrados."""
        cls._ensure_registered()
        return sorted(_REGISTRY.keys())

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    @classmethod
    def _name_from_instance(cls, representation: Any) -> str:
        """
        Intenta inferir el nombre canónico de la representación desde
        el nombre de su clase.

        Mapeos conocidos de clase → alias canónico:
          FlatTokenizer          → flat_tokenizer
          SemanticByteTokenizer  → semantic_byte
          ProtocolAwareTokenizer → protocol_aware
          GAFRepresentation      → gasf
          NprintImageRepresentation → nprint_image
        """
        class_name = type(representation).__name__.lower()

        # Orden: más específico primero
        patterns = [
            ("nprint_image",    "nprint_image"),
            ("nprint",          "nprint_image"),
            ("gaf",             "gasf"),
            ("semantic_byte",   "semantic_byte"),
            ("semanticbyte",    "semantic_byte"),
            ("protocol_aware",  "protocol_aware"),
            ("flat",            "flat_tokenizer"),
        ]

        for pattern, canonical in patterns:
            if pattern in class_name:
                return canonical

        # Último recurso: intentar usar el nombre de clase directamente
        logger.warning(
            "[Registry] No se pudo inferir el nombre canónico para '%s'. "
            "Usando nombre de clase como clave.",
            type(representation).__name__,
        )
        return class_name