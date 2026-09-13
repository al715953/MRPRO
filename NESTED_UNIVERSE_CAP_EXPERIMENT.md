# Experimento de techos anidados del universo

Fecha de ejecución: 2026-09-13.

## Pregunta

¿Podemos reducir el techo primario de 45,000 combinaciones sin perder los
ganadores 6/6 que ya sobreviven a la reducción de universo?

El muestreo `balanced_mixed` no permite responder limpiamente: al cambiar el
límite vuelve a extraer una muestra aleatoria de otro tamaño, por lo que un
universo pequeño no es subconjunto del grande. Para aislar el efecto del techo
se añadió el modo experimental `nested_balanced`, con un orden determinista de
núcleo y exploración. Con los mismos datos, configuración y semilla se cumple:

`U20k ⊂ U25k ⊂ U30k ⊂ U35k ⊂ U40k ⊂ U45k`

Durante el experimento producción no fue modificada. Después de revisar los
resultados, se adoptó por decisión operativa el modo `nested_balanced` con
techo primario de 30,000; el carril topológico continúa en 45,000.

## Barrido reciente

Configuración común: concursos 1561–1668, 108 sorteos, 24 boletos por sorteo,
semilla 20260908, selector híbrido 12+12 y carril topológico fijo en 45,000.

| Techo primario | Unión media | 6/6 en universo | Sorteos >=4 | Sorteos >=5 | 6/6 final | Recuperación bruta |
|---:|---:|---:|---:|---:|---:|---:|
| 20,000 | 59,699 | 5 | 3 | 1 | 0 | 22.40% |
| 25,000 | 64,649 | 5 | 4 | 1 | 0 | 22.09% |
| 30,000 | 69,611 | 5 | 6 | 2 | 0 | 26.23% |
| 35,000 | 74,536 | 5 | 4 | 0 | 0 | 19.08% |
| 40,000 | 79,463 | 5 | 5 | 0 | 0 | 20.05% |
| 45,000 | 84,366 | 5 | 8 | 0 | 0 | 21.96% |

30,000 fue el máximo local del barrido reciente. Frente a 20,000, su mejora de
ganancias fue de $992.92, pero no resultó estadísticamente concluyente
(permutación pareada de ganancias, p=0.233; IC bootstrap de la diferencia media
por sorteo: -$1.58 a $26.67).

## Forense del salto 30k a 35k

Los dos 5/6 de 30k provinieron del carril primario, ambos en el octavo puesto de
sus 12 boletos:

| Concurso | Ganador | Boleto 30k | Mejor con 35k |
|---:|---|---|---:|
| 1638 | 9, 12, 14, 24, 35, 36 | 9, 12, 14, 24, 25, 35 | 2/6 |
| 1657 | 1, 3, 9, 12, 22, 29 | 1, 3, 9, 11, 22, 29 | 3/6 |

Los ganadores exactos permanecieron dentro de todos los universos. Los 5,000
candidatos añadidos entre 30k y 35k alteraron la competencia del selector y
desplazaron esos vecinos 5/6. El problema ya no es cobertura del universo sino
estabilidad del ranking y de la cartera final.

## Comprobación en las tres ventanas históricas

Se congeló el candidato de 30k y se ejecutó en las ventanas previamente
definidas. Estas ventanas sirven como comprobación de robustez, pero no son una
validación independiente del barrido reciente porque existe solapamiento
temporal.

| Ventana | Sorteos | 6/6 universo 30k | Recuperación 30k | Recuperación control 45k |
|---|---:|---:|---:|---:|
| Desarrollo 1343–1450 | 108 | 2 | 23.39% | 22.77% |
| Validación 1451–1594 | 144 | 5 | 21.43% | 22.84% |
| Holdout 1595–1666 | 72 | 3 | 20.35% | 25.37% |
| Total ponderado | 324 | 10 | 21.84% | 23.38% |

En cobertura, 30k preservó los mismos 10 ganadores exactos que 45k. En la
selección final quedó por debajo del control: 23 contra 27 sorteos con al menos
4 aciertos, y 0 contra 1 con al menos 5 aciertos. Ninguna variante obtuvo un
6/6 entre los 24 boletos finales.

## Decisión

Se promovió 30k a producción por decisión operativa posterior al experimento.
La configuración oficial usa `nested_balanced` en el carril primario y conserva
45k en el topológico. El cambio se apoya en que 30k preservó los mismos 10
ganadores exactos del universo que 45k; debe tenerse presente que la cartera
final histórica quedó por debajo del control en las tres ventanas agregadas.

El siguiente experimento debe estabilizar el selector sobre prefijos anidados
—en especial sus 8 boletos núcleo y 4 profundos del carril primario— y evaluarse
en una ventana futura no usada para elegir el techo.

## Reproducción

```bash
.venv/bin/python run_melate_ab_experiments.py \
  --suite universe-nested-cap \
  --draws 108 \
  --tickets 24 \
  --seed 20260908 \
  --end-contest 1668
```
