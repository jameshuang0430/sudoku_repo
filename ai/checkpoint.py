from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .model import create_model


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(Path(checkpoint_path), map_location=device)
    model_type = payload.get("model_type", "mlp")
    model_config = payload.get("model_config", {})
    model = create_model(model_type, **model_config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload
