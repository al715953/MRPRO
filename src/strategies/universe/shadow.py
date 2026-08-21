"""Prospective universe-reduction shadows for Melate Retro."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.data_access.config import BEST_SETTINGS
from src.domain.dtos import PredictionConfigDTO
from src.strategies.universe_reduction import UniverseReductionStrategy


PROFILE_OOS_SET = (
    "2-1-2-1",
    "2-2-1-1",
    "2-1-1-2",
    "1-3-1-1",
    "1-2-3-0",
    "2-2-0-2",
    "3-0-1-2",
    "3-1-0-2",
)


@dataclass(frozen=True)
class UniverseShadowSpec:
    key: str
    label: str
    overrides: dict[str, Any] = field(default_factory=dict)


PROMOTED_UNIVERSE_SHADOWS = (
    UniverseShadowSpec(
        key="profile_oos_43k",
        label="Sombra universo perfiles OOS / 43K",
        overrides={
            "shadow_family": "universe_reduction",
            "promotion_reference_key": "principal_ai_adaptive",
            "universe_variant": "profile_oos_43k",
            "valid_decade_profiles": list(PROFILE_OOS_SET),
            "universe_ticket_limit": 45000,
        },
    ),
    UniverseShadowSpec(
        key="profile_same_budget_40k",
        label="Sombra universo perfiles OOS / mismo tamaño",
        overrides={
            "shadow_family": "universe_reduction",
            "promotion_reference_key": "principal_ai_adaptive",
            "universe_variant": "profile_same_budget_40k",
            "valid_decade_profiles": list(PROFILE_OOS_SET),
            "universe_ticket_limit": 39864,
        },
    ),
    UniverseShadowSpec(
        key="sniper_soft_veto",
        label="Sombra Sniper veto suave / reserva 10%",
        overrides={
            "shadow_family": "universe_reduction",
            "promotion_reference_key": "principal_ai_adaptive",
            "universe_variant": "sniper_soft_veto",
            "sniper_mode": "soft",
            "sniper_soft_penalty": 0.15,
            "sniper_soft_reserve_fraction": 0.10,
        },
    ),
)


def build_universe_shadow_variant(
    history,
    selector,
    spec: UniverseShadowSpec,
    *,
    total_balls: int,
    ticket_size: int,
    ticket_count: int,
) -> dict[str, Any]:
    """Generate one non-official portfolio with its own reduced universe."""

    effective_settings = dict(BEST_SETTINGS)
    effective_settings.update(spec.overrides)
    config = PredictionConfigDTO(
        total_balls=int(total_balls),
        ticket_size=int(ticket_size),
        num_tickets=int(ticket_count),
        filter_overrides=effective_settings,
    )
    reduction = UniverseReductionStrategy().predict(history, config, verbose=False)
    config.raw_universe_ptr = reduction.metadata.get("raw_ndarray")
    prediction = selector.predict(history, config)
    reduction_stats = dict(reduction.metadata.get("reduction_stage_stats") or {})
    metadata = dict(prediction.metadata or {})
    metadata.update(
        {
            **spec.overrides,
            "raw_universe_size": int(reduction.metadata.get("final_size", 0)),
            "sniper_mode": reduction.metadata.get("sniper_mode", "hard"),
            "sniper_candidates": reduction.metadata.get("sniper_candidates", []),
            "hard_excluded_numbers": reduction.metadata.get(
                "hard_excluded_numbers", []
            ),
            "universe_ticket_limit": reduction.metadata.get(
                "universe_ticket_limit"
            ),
            "reduction_stage_stats": reduction_stats,
        }
    )
    return {
        "key": spec.key,
        "label": spec.label,
        "official": False,
        "settings": dict(spec.overrides),
        "tickets": prediction.tickets,
        "metadata": metadata,
    }


def build_promoted_universe_shadows(
    history,
    selector,
    *,
    total_balls: int,
    ticket_size: int,
    ticket_count: int,
) -> list[dict[str, Any]]:
    return [
        build_universe_shadow_variant(
            history,
            selector,
            spec,
            total_balls=total_balls,
            ticket_size=ticket_size,
            ticket_count=ticket_count,
        )
        for spec in PROMOTED_UNIVERSE_SHADOWS
    ]
