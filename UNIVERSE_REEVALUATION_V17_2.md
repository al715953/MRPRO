# Revaluación del universo V17.2

Fecha del ejercicio: 2026-09-08
Presupuesto: 24 tickets por sorteo
Último concurso incluido: 1666

## Pregunta

Determinar si el universo productivo fijo de 45,000 candidatos redujo la
recuperación frente a la etapa de filtros Geo duros y si conviene restaurar un
universo menor o dinámico.

## Por qué producción muestra 45,000

`UniverseReductionStrategy` genera las 3,262,623 combinaciones de `C(39, 6)`,
aplica los filtros duros que estén habilitados y finalmente usa
`min(supervivientes, universe_ticket_limit)`. Por tanto, 45,000 es un tope, no
un relleno posterior. En V17.2 todos los hard gates están apagados, así que hay
más de 45,000 supervivientes en cada sorteo y el tope siempre se alcanza.

Con el pipeline topológico antiguo, las etapas reducen el universo así:

| Etapa | Antes | Después |
|---|---:|---:|
| Universo completo | 3,262,623 | 3,262,623 |
| Posicional | 3,262,623 | 2,521,519 |
| Suma | 2,521,519 | 762,782 |
| Estructura | 762,782 | 382,227 |
| Terminales | 382,227 | 381,838 |
| Espacial | 381,838 | 374,147 |
| Perfil de décadas | 374,147 | 202,389 |
| Entropía | 202,389 | 60,337 |
| Raíz digital | 60,337 | 51,210 |
| Complejidad AC | 51,210 | 39,864 |

El control con filtros topológicos y Sniper suave termina en 39,864. El perfil
V16 completo sí es dinámico: Sniper hard lo movió entre 32,984 y 39,864 en la
validación y entre 34,672 y 39,864 en el holdout.

## Diseño experimental

- Mismo modelo contextual de origen fijo en cada ventana.
- Mismo scorer productivo V17.2: IA contextual pura.
- Mismo selector productivo: 16 tickets core + 8 deep.
- Mismo presupuesto de 24 tickets.
- Comparaciones pareadas por concurso.
- Desarrollo: 108 concursos hasta el 1450.
- Validación independiente: 144 concursos, 1451–1594.
- Holdout final: 72 concursos, 1595–1666.

La exploración inicial también probó universos de 30k, 45k, 60k y 90k;
fracciones de exploración de 0%, 25%, 50%, 75% y 100%; filtros aislados de
suma, estructura, entropía y perfil de décadas; filtros topológicos completos;
y el pipeline V16 completo.

## Resultados por ventana

| Ventana | V17.2 45k | Topológicos 39,864, radar 0 | V16 duro completo |
|---|---:|---:|---:|
| Desarrollo, 108 | 19.95% | 20.41% | 23.59% |
| Validación, 144 | 23.32% | 23.64% | 21.06% |
| Holdout, 72 | 22.50% | 20.52% | 26.98% |
| Total, 324 | **22.02%** | **21.87%** | **23.22%** |

La recuperación del perfil V16 cambia de signo entre ventanas. Su ventaja
agregada depende en buena medida de un 5/6 adicional y no es estadísticamente
concluyente.

## Resultado agregado pareado

| Métrica, 324 concursos | V17.2 45k | Topológicos 39,864 | V16 duro completo |
|---|---:|---:|---:|
| Recuperación bruta | 22.02% | 21.87% | 23.22% |
| Sorteos con máximo ≥4/6 | **26** | 21 | 18 |
| Sorteos con máximo ≥5/6 | 1 | 1 | 2 |
| Acierto máximo medio | **2.660** | 2.623 | 2.608 |
| Ganadores 6/6 dentro del universo | 2 | **8** | **8** |
| 6/6 entre los 24 tickets finales | 0 | 0 | 0 |
| Diferencia de ganancias vs. V17.2 | — | -$113.60 | +$934.76 |
| p pareado de ganancias | — | 0.944 | 0.587 |
| IC 95% de diferencia media por sorteo | — | [-$9.62, $8.85] | [-$7.17, $13.36] |

En cobertura previa al selector, V17.2 produjo 273 sorteos cuyo mejor candidato
tenía 5/6, contra 107 con los filtros topológicos. A la vez, los filtros duros
incluyeron 8 ganadores exactos contra 2 de V17.2. La diferencia exacta es
sugestiva, aunque todavía no concluyente (McNemar exacto bilateral `p=0.109`).
Es una señal que debe preservarse en el siguiente diseño, no descartarse como
ruido. Ninguno de los diez ganadores presentes en ambos universos llegó a los
24 tickets finales; los ganadores del universo duro quedaron aproximadamente
entre los ranks 4,674 y 39,713.

## Separación filtro/radar

En validación, los filtros topológicos con radar abierto dieron 23.64%, 12
tickets de 4/6 y 1 de 5/6. Al aplicar radar 50 al mismo universo bajaron a
19.91%, 9 de 4/6 y ninguno de 5/6. Sin filtros, radar 50 obtuvo 22.71%, frente a
23.32% con radar abierto. El corte de radar antiguo no debe restaurarse.

## Decisión provisional

No se restaura completo el pipeline V16: su aumento agregado de recuperación no
supera la incertidumbre, no se repite en las tres ventanas y sacrifica 8
sorteos con ≥4/6. Sin embargo, tampoco se descarta su núcleo topológico: su
cobertura observada de 6/6 es cuatro veces mayor.

El siguiente candidato debe ser híbrido: conservar una vía de candidatos que
cumplan la topología antigua y otra vía suave/exploratoria que mantenga las
vecindades 5/6 de V17.2. Después, el selector debe reservar parte de los 24
tickets para ranks profundos de la vía topológica. El problema observado ya no
es sólo el tamaño del universo; es el puente entre los ganadores profundos y la
cartera final.

Seguimiento: este candidato se implementó y validó como V17.3. El resultado y
la decisión productiva están documentados en
`HYBRID_SELECTOR_REEVALUATION_V17_3.md`.

El runner conserva la suite `universe-dynamic` y telemetría explícita de tamaño
real y ganador dentro del universo. Esto permite repetir la evaluación antes de
otra promoción sin confundir cambios de universo, radar, scorer o selector.

## Reproducción

```bash
.venv/bin/python run_melate_ab_experiments.py \
  --suite universe-dynamic --draws 108 --tickets 24 \
  --seed 20260908 --end-contest 1450
```

Para una decisión futura se deben conservar ventanas temporales separadas y no
promover por una sola ocurrencia de 5/6 o por recuperación agregada sin prueba
pareada.
