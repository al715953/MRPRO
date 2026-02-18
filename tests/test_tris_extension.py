from pathlib import Path

from src.core.rules import TrisMultiplicadorRules
from src.data_access.loader import TrisMultiplicadorLoader
from src.data_access.config import get_lottery_profile


def test_tris_loader_reads_digit_columns(tmp_path: Path):
    csv = tmp_path / "tris.csv"
    csv.write_text(
        "CONCURSO,FECHA,DIGITO1,DIGITO2,DIGITO3,DIGITO4,DIGITO5\n"
        "1001,01/01/2025,1,2,3,4,5\n",
        encoding="utf-8",
    )

    history = TrisMultiplicadorLoader(str(csv)).load_data()
    assert history.concursos == [1001]
    assert history.winning_numbers == [[1, 2, 3, 4, 5]]


def test_tris_rules_exact_match_prize_and_profile_exists():
    rules = TrisMultiplicadorRules(base_prize=500)
    assert rules.validate_ticket([1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 2]) == (5, True)
    assert rules.calculate_prize(5, True) == 1000
    assert get_lottery_profile("tris_multiplicador").ticket_size == 5
