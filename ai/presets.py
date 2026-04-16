from __future__ import annotations

from dataclasses import dataclass

from .decode import DecodeMode, ITERATIVE_CONFIDENCE_THRESHOLD


@dataclass(frozen=True)
class DecodePreset:
    name: str
    decode_mode: DecodeMode
    iterative_threshold: float = ITERATIVE_CONFIDENCE_THRESHOLD
    iterative_max_fills_per_round: int | None = None


DECODE_PRESETS: dict[str, DecodePreset] = {
    "argmax": DecodePreset(name="argmax", decode_mode="argmax"),
    "iterative": DecodePreset(name="iterative", decode_mode="iterative"),
    "iterative_strict": DecodePreset(
        name="iterative_strict",
        decode_mode="iterative",
        iterative_threshold=0.75,
        iterative_max_fills_per_round=2,
    ),
    "solver_guided": DecodePreset(name="solver_guided", decode_mode="solver_guided"),
}


def apply_decode_preset(
    decode_preset: str | None,
    decode_mode: DecodeMode,
    iterative_threshold: float,
    iterative_max_fills_per_round: int | None,
) -> tuple[DecodeMode, float, int | None]:
    if decode_preset is None:
        return decode_mode, iterative_threshold, iterative_max_fills_per_round

    preset = DECODE_PRESETS[decode_preset]
    return preset.decode_mode, preset.iterative_threshold, preset.iterative_max_fills_per_round
