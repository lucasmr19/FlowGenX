# core/experiment_builder.py

from .config_loader import load_yaml
from .builders import (
    build_model_from_dict,
    build_representation_from_dict,
)


def _validate_compatibility(rep, model):
    rep_shape = rep.output_shape
    model_input = model.config.input_shape

    if model_input is not None and rep_shape != model_input:
        raise ValueError(
            f"Incompatibilidad:\n"
            f"  Representation output_shape = {rep_shape}\n"
            f"  Model input_shape         = {model_input}"
        )

def build_experiment_from_yaml(path: str):
    cfg = load_yaml(path)

    if "model" not in cfg or "representation" not in cfg:
        raise KeyError("El YAML debe contener 'model' y 'representation'")

    # construir componentes
    rep = build_representation_from_dict(cfg["representation"])
    model = build_model_from_dict(cfg["model"])

    # validación crítica (MUY importante)
    _validate_compatibility(rep, model)

    return {
        "representation": rep,
        "model": model,
        "training": cfg.get("training", {}),
        "data": cfg.get("data", {}),
    }