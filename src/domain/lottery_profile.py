from dataclasses import dataclass


@dataclass(frozen=True)
class LotteryProfile:
    """Configuración mínima para describir un juego soportado."""

    code: str
    display_name: str
    csv_filename: str
    source_url: str
    total_balls: int
    ticket_size: int
    includes_additional_ball: bool = False

