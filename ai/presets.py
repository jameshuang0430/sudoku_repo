from __future__ import annotations

from dataclasses import dataclass

from .decode import DecodeMode, ITERATIVE_CONFIDENCE_THRESHOLD


@dataclass(frozen=True)
class DecodePreset:
    name: str
    decode_mode: DecodeMode
    iterative_threshold: float = ITERATIVE_CONFIDENCE_THRESHOLD
    iterative_max_fills_per_round: int | None = None
    summary: str = ""


DECODE_PRESETS: dict[str, DecodePreset] = {
    "research_raw": DecodePreset(
        name="research_raw",
        decode_mode="argmax",
        summary="Raw model output with no repair.",
    ),
    "research_iterative": DecodePreset(
        name="research_iterative",
        decode_mode="iterative",
        summary="Unrestricted iterative refinement for research comparison.",
    ),
    "production_pure": DecodePreset(
        name="production_pure",
        decode_mode="iterative",
        iterative_threshold=0.75,
        iterative_max_fills_per_round=2,
        summary="Strict non-solver iterative decoding tuned for accuracy.",
    ),
    "production_fast": DecodePreset(
        name="production_fast",
        decode_mode="solver_guided",
        summary="Solver-guided production path optimized for exact repair and lower CPU latency.",
    ),
    "argmax": DecodePreset(
        name="argmax",
        decode_mode="argmax",
        summary="Back-compat alias for research_raw.",
    ),
    "iterative": DecodePreset(
        name="iterative",
        decode_mode="iterative",
        summary="Back-compat alias for research_iterative.",
    ),
    "iterative_strict": DecodePreset(
        name="iterative_strict",
        decode_mode="iterative",
        iterative_threshold=0.75,
        iterative_max_fills_per_round=2,
        summary="Back-compat alias for production_pure.",
    ),
    "solver_guided": DecodePreset(
        name="solver_guided",
        decode_mode="solver_guided",
        summary="Back-compat alias for production_fast.",
    ),
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
