# Evaluación del selector híbrido V17.3

Fecha: 2026-09-08
Presupuesto: 24 tickets por sorteo
Último concurso evaluado: 1666

## Objetivo

Preservar la mayor cobertura 6/6 observada en el universo topológico antiguo
sin perder las vecindades 5/6 y el hit rate del universo suave V17.2.

## Diseño promovido

- Unión sin duplicados del universo V17.2 de 45k y los supervivientes de los
  filtros topológicos antiguos con Sniper suave.
- Un único scorer de IA contextual sobre toda la unión.
- 12 tickets del carril suave y 12 tickets de estratos profundos del carril
  topológico.
- Duplicados entre carriles excluidos antes de completar la segunda cuota.
- Radar abierto y sin reemplazo Sniper posterior.

Se probaron asignaciones 18+6, 16+8 y 12+12. La selección se hizo en desarrollo
y el candidato 12+12 se conservó sin ajustes para validación y holdout.

## Ventanas temporales

| Ventana | Concursos | V17.2 45k | Híbrido 12+12 |
|---|---:|---:|---:|
| Desarrollo | 1343–1450 | 19.95% | **22.77%** |
| Validación | 1451–1594 | **23.32%** | 22.84% |
| Holdout | 1595–1666 | 22.50% | **25.37%** |

Cada ventana usa un modelo fixed-origin entrenado únicamente con concursos
anteriores a su inicio.

## Resultado agregado, 324 sorteos

| Métrica | V17.2 45k | Híbrido 12+12 |
|---|---:|---:|
| Recuperación bruta | 22.02% | **23.38%** |
| Ganancia total | $17,120.10 | **$18,179.62** |
| Sorteos con máximo ≥4/6 | 26 | **27** |
| Sorteos con máximo ≥5/6 | 1 | 1 |
| Acierto máximo medio | 2.660 | **2.738** |
| Ganadores 6/6 dentro del universo | 2 | **10** |
| Sorteos cuyo mejor candidato tenía 5/6 | 273 | **282** |
| 6/6 entre los 24 tickets finales | 0 | 0 |

La diferencia pareada del acierto máximo medio fue `+0.077` por sorteo, con
IC bootstrap 95% `[0.000, 0.154]` y prueba de permutación bilateral `p=0.058`.
La diferencia de recuperación fue positiva pero todavía incierta
(`+$1,059.52`, `p=0.224`).

## Decisión

Se promueve V17.3 12+12 porque mejora el acierto máximo medio en las tres
ventanas, conserva el 5/6 seleccionado, no reduce los sorteos ≥4/6 y reúne la
cobertura exacta de ambos universos. La promoción no se interpreta como prueba
de capacidad predictiva para acertar 6/6 entre sólo 24 tickets: en el backtest
ninguna variante seleccionó un ganador exacto.

Las sombras históricas permanecen sobre el universo V17.2 de 45k y selector
16+8 para evitar romper su serie comparativa.

## Reproducción

```bash
.venv/bin/python run_melate_ab_experiments.py \
  --suite universe-hybrid --draws 72 --tickets 24 \
  --seed 20260908 --end-contest 1666
```
