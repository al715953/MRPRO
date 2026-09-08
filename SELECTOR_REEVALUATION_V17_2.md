# Reevaluación del selector Melate Retro V17.2

Fecha del análisis: 2026-09-07. Objetivo predefinido: seleccionar 24 tickets
priorizando la frecuencia de carteras con al menos 5/6; 6/6 fue secundario.

## Método

- Universo V17 fijo: 45,000 candidatos (50% núcleo suave, 50% exploración).
- Backtests de origen fijo: cada modelo sólo vio concursos anteriores a su
  ventana de prueba.
- Ledger aislado y mismo presupuesto de 24 tickets para cada variante.
- Desarrollo: concursos 1343–1450 (108 sorteos).
- Validación: concursos 1451–1594 (144 sorteos).
- Holdout final: concursos 1595–1666 (72 sorteos).
- Se probaron profundidad de ranking, distancia máxima, asignaciones
  núcleo/cobertura/profundidad, mezclas IA/Geo, IA por número y reserva Sniper.

Cada ticket cubre 199 resultados posibles con 5/6 o 6/6. El máximo para 24
tickets sin solapamiento es 4,776 resultados, apenas 0.1464% de
`C(39, 6) = 3,262,623`. Por esta baja frecuencia, además de ≥5 se usaron como
criterios secundarios: sorteos con máximo ≥4, máximo medio, rango estable,
cobertura exacta de radio uno y recuperación bruta.

## Resultado por ventana

| Configuración | Desarrollo 5/6 · 4/6 | Validación 5/6 · 4/6 | Holdout 5/6 · 4/6 |
|---|---:|---:|---:|
| Producción V17.1 | 0 · 6 | 0 · 7 | 0 · 6 |
| Distancia adaptativa | 0 · 7 | 0 · 11 | 0 · 5 |
| IA pura, núcleo 16 + profundo 8 | 0 · 8 | 1 · 11 | 0 · 7 |
| Geo puro, núcleo 16 + profundo 8 | 1 · 7 | 0 · 12 | 0 · 7 |

La distancia explícita alcanzó hasta 4,776/4,776 resultados cubiertos, pero
perdió máximo medio y no se sostuvo en holdout. Las asignaciones con sólo cuatro
élites y 16–20 tickets de cobertura también quedaron por debajo. La mezcla fija
IA/Geo 50/50 encontró un 5/6 en desarrollo, pero cayó a cinco tickets de 4/6 y
cero de 5/6 en validación. El modelo de IA por número no mejoró el control.

## Consolidado de las tres ventanas (324 sorteos)

| Configuración | Tickets 5/6 | Tickets 4/6 | Sorteos máx. ≥4 | Máximo medio | Recuperación |
|---|---:|---:|---:|---:|---:|
| Producción V17.1 | 0 | 19 | 17 | 2.509 | 20.13% |
| Distancia adaptativa | 0 | 23 | 22 | 2.448 | 20.69% |
| IA pura, 16 + 8 | 1 | 26 | 26 | 2.660 | 22.04% |
| Geo puro, 16 + 8 | 1 | 26 | 23 | 2.664 | 23.59% |

Frente a V17.1, IA pura 16+8 mejoró el máximo medio en `+0.151` por sorteo,
con bootstrap 95% `[+0.065, +0.238]` y permutación pareada `p=0.00105`.
La diferencia de eventos ≥4 no fue concluyente (`McNemar p=0.136`) y un solo
5/6 no permite afirmar una tasa estable de 5/6. El resultado debe interpretarse
como una mejora del selector bajo evidencia limitada, no como una predicción de
probabilidad física ni como garantía de premio.

## Decisión

Se promueve IA contextual pura con 16 tickets del núcleo nativo y 8 de estratos
profundos. Geo puro igualó los tickets ≥4 y ≥5, pero concentró los ≥4 en menos
sorteos; el objetivo solicitado favorece frecuencia de acierto. Se elimina la
cuota de reemplazo Sniper posterior: no mejoró 4/6 o 5/6 y puede reducir la
cobertura ya optimizada. La penalización Sniper suave permanece en el score.

La configuración de distancia queda disponible como modo experimental
`five_hit_coverage`, acompañada por telemetría exacta de quintetos y radio uno,
pero no se promueve a producción porque falló el holdout.
