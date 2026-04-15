# core/builders.py

from ..models_ml import (
    REGISTRY_MODELS,
    CONFIG_REGISTRY_MODELS,
    get_model,
)

from ..representations import (
    REGISTRY_REPRESENTATIONS,
    CONFIG_REGISTRY_REPRESENTATIONS,
    get_representation,
)

from .config_loader import load_yaml


def build_model_from_dict(cfg: dict):
    if "model_type" not in cfg:
        raise KeyError("Falta 'model_type' en config de modelo")

    model_type = cfg["model_type"]

    if model_type not in CONFIG_REGISTRY_MODELS:
        raise ValueError(f"Modelo desconocido: {model_type}")

    config_cls = CONFIG_REGISTRY_MODELS[model_type]

    # construir config tipada
    config = config_cls(**cfg)

    # instanciar modelo
    model = get_model(model_type, config)

    return model

def build_model_from_yaml(path: str):
    cfg = load_yaml(path)
    return build_model_from_dict(cfg)

def build_representation_from_dict(cfg: dict):
    if "representation_type" not in cfg:
        raise KeyError("Falta 'representation_type' en config de representación")

    rep_type = cfg["representation_type"]

    if rep_type not in CONFIG_REGISTRY_REPRESENTATIONS:
        raise ValueError(f"Representación desconocida: {rep_type}")

    config_cls = CONFIG_REGISTRY_REPRESENTATIONS[rep_type]

    config = config_cls(**cfg)

    rep = get_representation(rep_type, config)

    return rep

def build_representation_from_yaml(path: str):
    cfg = load_yaml(path)
    return build_representation_from_dict(cfg)