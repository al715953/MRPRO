from collections import Counter
from src.core.filters.base import BaseFilter
from src.domain.dtos import CandidateCombination


class ConsecutiveFilter(BaseFilter):
    """
    Evita escaleras excesivas (Ej: 1,2,3,4...).
    En 6/39, más de 2 pares consecutivos es rarísimo.
    """

    def __init__(self, max_consecutive_pairs: int = 2):
        super().__init__(name="ConsecutiveFilter")
        self.max_pairs = max_consecutive_pairs

    def validate(self, candidate: CandidateCombination) -> bool:
        nums = sorted(candidate.numbers)
        pairs = 0
        for i in range(len(nums) - 1):
            if nums[i + 1] == nums[i] + 1:
                pairs += 1
        return pairs <= self.max_pairs


class QuadrantFilter(BaseFilter):
    """
    Fuerza la distribución espacial.
    Divide el boleto en 4 zonas y exige que se usen al menos 3 cuadrantes diferentes.
    Q1: 1-9, Q2: 10-19, Q3: 20-29, Q4: 30-39
    """

    def __init__(self):
        # CORRECCIÓN: Inicializamos el padre con el nombre del filtro
        super().__init__(name="QuadrantFilter")

    def validate(self, candidate: CandidateCombination) -> bool:
        # Mapeo simple de cuadrantes
        quadrants = set()
        for n in candidate.numbers:
            if n <= 9:
                quadrants.add(1)
            elif n <= 19:
                quadrants.add(2)
            elif n <= 29:
                quadrants.add(3)
            else:
                quadrants.add(4)

        # Exigimos ocupación de territorio: al menos 2 cuadrantes tocados
        # (Nota: Bajé la exigencia a 2 para no ser tan restrictivo en 39 números,
        # pero idealmente buscamos 3. Si ves pocos resultados, mantén >= 2)
        return len(quadrants) >= 2


class LastDigitFilter(BaseFilter):
    """
    Análisis de Terminales.
    Evita que demasiados números terminen en el mismo dígito (Ej: 3, 13, 23, 33).
    Máximo permitido usualmente es 3.
    """

    def __init__(self, max_same_ending: int = 3):
        super().__init__(name="LastDigitFilter")
        self.max_same = max_same_ending

    def validate(self, candidate: CandidateCombination) -> bool:
        endings = [n % 10 for n in candidate.numbers]
        counts = Counter(endings)
        # Si algún dígito se repite más de lo permitido, descartar
        if any(count > self.max_same for count in counts.values()):
            return False
        return True
