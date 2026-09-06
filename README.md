# MRPRO

## Directorio de datos

Los archivos persistentes viven en `data/`, en la raíz del proyecto. El código de
aplicación permanece en `src/` y no contiene históricos, modelos, apuestas ni
reportes generados.

Las rutas se resuelven con `pathlib`, por lo que funcionan con separadores de
Windows y macOS/Linux. En ejecutables empaquetados se usa `data/` junto al
ejecutable cuando es escribible; si la instalación está protegida, se utiliza la
carpeta de datos del usuario. Los archivos iniciales empaquetados se copian una
sola vez y nunca reemplazan datos persistentes.

## Backtest Melate fixed-origin

La opción 6, modo 2, entrena automáticamente modelos exclusivos para la ventana
solicitada. Si el histórico termina en `#1661`, un backtest de 108 usa entrenamiento
hasta `#1553` y prueba `#1554-#1661`; uno de 218 entrena hasta `#1443` y prueba
`#1444-#1661`.

La selección de rondas usa una validación interna anterior al inicio del test.
Los sorteos evaluados no intervienen en entrenamiento ni ajuste. Los modelos se
guardan por hash del dataset, corte y tamaño en `data/backtest_models/`; repetir la
misma corrida reutiliza la caché. Estos archivos nunca reemplazan los modelos de
producción generados por la opción 4.

La reserva 80/20 mostrada durante el reentrenamiento general es un diagnóstico
temporal separado y ya no determina el corte de la opción 6.

## Laboratorio de covering designs

El menú Melate incluye la opción `C`, y también puede ejecutarse directamente:

```bash
python3 run_covering_experiment.py \
  --v 15 \
  --t k-1 \
  --budget 300 \
  --candidate-method oracle_candidate_set \
  --random-trials 100 \
  --draws 108
```

Para barridos matemáticos sin backtest:

```bash
python3 run_covering_experiment.py \
  --mode math \
  --v 10,12,15,18,20 \
  --t k-1,k-2 \
  --budget 50,100,200,300,500,1000
```

Objetivo mixto reproducible `t=5 + t=4`, con tres ventanas temporales:

```bash
python3 run_covering_experiment.py \
  --v 15 --t k-1 --secondary-t k-2 \
  --primary-weight 0.5 --secondary-weight 0.5 \
  --budget 300 --candidate-method mrpro_candidate_set \
  --draws 108 --temporal-folds 3 \
  --output data/covering_mixed_mrpro_v15_m300.json
```

Los pesos se fijan antes del backtest. Las ventanas son resúmenes walk-forward
contiguos y no deben utilizarse para reajustar el modelo que después se evalúa
en la última ventana, etiquetada `holdout_test`.

### Sombras covering prospectivas

La opción de producción registra, sin inversión real y sin cambiar los boletos
oficiales, dos carteras adicionales de 300 boletos:

- `cover_mixed_v20_m300`: objetivo mixto 50/50 para `t=5` y `t=4`.
- `cover_mixed_v18_m300`: control conservador con el mismo objetivo y presupuesto.

Ambas usan el snapshot MRPRO generado antes del sorteo, `candidate_rank_depth=500`
y se liquidan desde la opción 8 junto con las demás carteras sombra. El resumen
normaliza resultados por sorteo y por 1,000 boletos para comparar presupuestos
distintos sin confundir volumen con calidad.

La misma opción 8 muestra un segundo tablero de promoción. Cada challenger se
compara sorteo a sorteo contra una referencia MRPRO con exactamente el mismo
presupuesto. Para las carteras covering se registra también
`benchmark_mrpro_native_m300`; si falta, el gate muestra `UNMATCHED_BUDGET`.

Estados del gate:

- `INSUFFICIENT_SAMPLE`: menos de 20 sorteos pareados.
- `COLLECTING` / `PROMISING`: revisión inicial entre 20 y 49 sorteos.
- `ELIGIBLE_FOR_PILOT`: al menos 50 sorteos y evidencia pareada favorable.
- `NO_ADVANTAGE` / `REJECTED`: no mejora o presenta deterioro respaldado.

El gate considera diferencia de máximo de aciertos, tasas ≥4 y ≥5, permutation
test, intervalo bootstrap y estabilidad en tres ventanas. Nunca cambia
producción automáticamente. La fotografía reproducible se exporta en
`data/Tablero_Sombra.json`.

### Sombras prospectivas de IA, selector y reducción del universo

La opción 7 genera carteras no oficiales para aislar cambios de IA, selector y
universo. Salvo los benchmarks marcados con otro presupuesto, usan el mismo
número de boletos que la cartera principal:

- `profile_oos_43k`: perfiles de décadas seleccionados hasta el concurso 1443,
  con límite de 45,000 combinaciones.
- `profile_same_budget_40k`: los mismos perfiles con límite de 39,864
  combinaciones.
- `sniper_off_control`: apaga por completo la señal Sniper para compararla con
  el veto suave oficial.
- `legacy_hard_geo_v16`: conserva como control prospectivo los filtros duros,
  el Sniper hard y el radar por mediana de V16.
- `challenger_context50_number50`: conserva la IA activa y mezcla en partes
  iguales el modelo contextual y el modelo por número.
- `challenger_deep_rank_5000`: conserva 24 boletos, pero distribuye cobertura
  por estratos hasta rank 5000 para medir si el límite 500 es demasiado corto.

Desde V17, producción no elimina combinaciones mediante los antiguos filtros
Geo. Mantiene 45,000 candidatos con 50% de núcleo puntuado suavemente y 50% de
exploración uniforme reproducible por concurso. Sniper usa veto suave y el
radar no aplica un corte previo por mediana. Las variantes se guardan solamente
en `data/Carteras_Sombra.json`; la opción 8 las liquida y compara contra
`principal_ai_adaptive`. Ninguna sombra modifica automáticamente producción.

El reductor registra tamaños antes y después de cada etapa en
`reduction_stage_stats`. `universe_ticket_limit` controla el Top-K final; un
valor no positivo conserva el límite histórico de 45,000.

Los controles de reducción son independientes y permanecen disponibles para
reproducir experimentos legacy. Cada etapa tiene un flag `*_filter_enabled`.
La suma y la desviación estándar continúan como señales suaves; sólo podan si
sus flags correspondientes se activan explícitamente. La telemetría registra
el modo de selección, la semilla y cuántos candidatos provinieron del núcleo y
de exploración.

La comparación fixed-origin reproducible entre el control V16 y producción V17
puede ejecutarse con:

```bash
python3 run_melate_ab_experiments.py \
  --suite universe-v17 --draws 216 --tickets 24
```

Para iniciar una cartera oficial V17 desde el menú, el orden operativo es:

1. Opción 5: sincronizar el histórico oficial.
2. Opción 4: reentrenar los modelos hasta el último concurso disponible.
3. Opción 7: generar la cartera del concurso siguiente.

La opción 7 usa directamente `BEST_SETTINGS`, registra `VERSION_TAG` en el
ledger y bloquea una segunda cartera oficial para el mismo concurso. Una prueba
seca debe conservar 45,000 candidatos, sin filtros duros activos, divididos en
22,500 de núcleo y 22,500 de exploración.

La calibración de la opción 3 usa orden cronológico y separa la ventana pedida
en validación (70%) y test reservado (30%). Los parámetros se eligen únicamente
con validación; el test se reporta una sola vez. La calibración estructural se
ejecuta con Sniper apagado, porque sus pesos se optimizan por separado. Si el
Sniper no genera exclusiones en validación, se conservan los pesos vigentes y
el resultado se marca `selection_inconclusive`.

En backtest, `AIr` significa *score relativo min-max*, no probabilidad de
premio. `pNN` es su percentil dentro del universo del sorteo y `Mix` muestra
los pesos IA/Geo aplicados realmente a esa combinación. La validación temporal
del cerebro (`VALIDADA` o `NO VALIDADA`, junto con AUC y umbral) se informa una
sola vez al inicio porque es constante durante una corrida fixed-origin. Solo
se mostraría `V`/`NV` en una fila si una implementación declarara validación
variable por sorteo. Ninguno de estos indicadores debe interpretarse como
probabilidad física del sorteo.

Las sombras de IA y profundidad también pueden repetirse en un backtest
fixed-origin reproducible —el script prepara automáticamente el cerebro hasta
el sorteo anterior a la ventana— con:

```bash
python3 run_melate_ab_experiments.py --suite selector-shadows --draws 218 --tickets 24
```

Para aislar únicamente el peso entre la IA contextual y la IA por número se
puede ejecutar el barrido controlado `100/0`, `85/15` y `50/50`:

```bash
python3 run_melate_ab_experiments.py --suite controlled-weights --draws 216 --tickets 24
```

El reporte incluye resultados máximos pareados por sorteo, bootstrap, prueba
de permutación y McNemar. Los ledgers intermedios son temporales: el experimento
no sobrescribe `backtest_results.json` ni añade filas al log forense principal.

La curva marginal de boletos utiliza carteras anidadas —cada presupuesto
conserva todos los boletos del anterior— y puede repetirse con:

```bash
python3 run_melate_ticket_budget_sweep.py --draws 216 --base 24 --maximum 36 --step 2
```

Además del retorno total, el reporte separa el costo y premio de cada pareja de
boletos adicional, resultados ≥4/≥5/6 y pruebas pareadas contra el presupuesto
base. El campo `prize_breakdown` distingue `1+AD`, `2+AD`, `3`, `4`, `5`,
`5+AD`, `6` y `SIN_PREMIO`, con cantidad de tickets y ganancia por categoría.
`gross_return_ratio` representa recuperación bruta y `net_roi` el retorno neto.
También utiliza un ledger temporal para no alterar el último backtest.

La liquidación de `Mis_Apuestas.csv` conserva el mismo desglose en las columnas
`AciertosNaturales`, `Adicional` y `CategoriaPremio`. Los ledgers anteriores se
actualizan de forma compatible al agregar una predicción nueva.

### Reserva profunda 20 + 10

La sombra `challenger_core20_deep10` conserva los primeros 20 tickets del
selector nativo y añade 10 tickets de profundidad. El rango desde rank 501
hasta el final del radar se divide en diez estratos de igual población; se
elige uno por estrato priorizando novedad de pares (40%), números poco
representados (25%), baja similitud con la cartera (20%) y calidad local dentro
del estrato (15%). Las bandas son relativas al tamaño del radar y quedaron
fijadas antes de evaluar resultados, por lo que no siguen los ranks de jackpots
históricos concretos.

Producción continúa usando sus tickets oficiales. La variante de 30 se guarda
como sombra y se compara con `benchmark_mrpro_native_m30`, también de 30
tickets. El backtest aislado se ejecuta con:

```bash
python3 run_melate_ab_experiments.py \
  --suite deep-dispersion --draws 108 --tickets 30
```

La comparación de esta cartera contra boletos completamente aleatorios del
universo C(39,6), sin IA, filtros ni reducción, se ejecuta después con:

```bash
python3 run_melate_full_random_benchmark.py \
  --draws 108 --tickets 30 --trials 100000
```

El baseline genera 30 boletos uniformes y sin repetición para cada sorteo, con
el mismo presupuesto que MRPRO. Utiliza la partición hipergeométrica exacta del
universo para no materializar sus 3,262,623 combinaciones. El reporte conserva
una corrida reproducible, la distribución Monte Carlo, percentiles y la
probabilidad empírica de que random iguale o supere a MRPRO; las ganancias se
reportan también sin jackpot porque ese evento raro domina la media.

### Sombra élite + cobertura + profundidad

El selector experimental `elite_coverage_deep` divide una cartera de 30 en tres
zonas configurables. La zona élite conserva exactamente los mejores ranks; la
zona de cobertura maximiza incrementalmente pares, ternas y cuartetas nuevas;
la zona profunda repite ese objetivo dentro de estratos de igual población.
Producción oficial no cambia: la opción 7 guarda adicionalmente
`challenger_elite10_cover10_deep10` como sombra prospectiva y la compara contra
`challenger_core20_deep10`, ambas con 30 boletos.

Las variantes precomprometidas 10/10/10, 15/10/5 y 10/15/5 se ejecutan con:

```bash
python3 run_melate_ab_experiments.py \
  --suite elite-coverage-deep --draws 218 --tickets 30
```

`--variant NOMBRE` permite repetir sólo una variante diagnóstica. La telemetría
separa el rank por score —que puede empatar— del `winner_stable_rank` único que
usa el selector, e informa el máximo solapamiento ganador–cartera, faltantes y
conteos de boletos que cubrieron al menos cuatro o cinco números.

### Rotación del log forense

`data/detailed_forensic_log.csv` conserva el historial de corridas, pero rota
antes de superar 25 MiB. El archivo anterior se comprime de forma atómica en
`data/forensic_log_archive/` y la corrida nueva permanece completa en el CSV
activo. Se retienen 12 archivos comprimidos; ambos valores son configurables
mediante `FORENSIC_LOG_MAX_BYTES` y `FORENSIC_LOG_ARCHIVE_KEEP`.

`PerformanceTracker.get_summary()` sigue devolviendo el histórico retenido
completo, incluyendo los `.csv.gz`. La opción `include_archives=False` permite
leer únicamente el archivo activo. Un límite de tamaño no positivo desactiva
la rotación y una retención no positiva conserva los archivos sin poda.

`oracle_candidate_set` inserta deliberadamente el resultado ganador dentro del
conjunto candidato. Es un control matemático, no una estrategia predictiva. Los
reportes separan cobertura combinatoria, calidad del conjunto candidato y
resultados walk-forward. Todas las comparaciones random utilizan exactamente el
mismo número efectivo de boletos que el covering correspondiente.

## Extensión propuesta: **Tris con Multiplicador**

Se dejó una base para reutilizar módulos existentes del proyecto (scraper, loaders, reglas y backtest) y soportar un segundo juego además de Melate Retro.

### Cambios aplicados

1. **Perfiles de juego**
   - Se incorporó `LotteryProfile` para describir cada juego (URL, archivo histórico, tamaño de ticket, etc.).
   - Se registraron perfiles para `melate_retro` y `tris_multiplicador`.

2. **Loader para Tris con Multiplicador**
   - Nuevo `TrisMultiplicadorLoader` que carga históricos con columnas `DIGITO1..5` (o alias `D1..5` / `F1..5`).

3. **Reglas de negocio para backtest de Tris**
   - Se añadió `TrisMultiplicadorRules` con validación de acierto exacto y pago base multiplicable.

4. **Backtest reusable**
   - `BacktestEngine` ahora permite inyectar reglas (`rules`) en el constructor.
   - La distribución de aciertos ya no está fija en 0..6; se ajusta a `rules.max_hits`.

5. **Scraper multi-juego**
   - `actualizar_csv()` ahora recibe `game_code` (`melate_retro` por defecto) y guarda en el CSV del perfil correspondiente.

### Siguientes pasos recomendados para completar la funcionalidad

- **Generación de código para Tris**:
  - Crear estrategia dedicada para secuencias de 5 dígitos (orden importa).
  - Definir espacio de búsqueda (00000-99999 o universo reducido por señales históricas).
- **Backtest financiero real**:
  - Sustituir `base_prize` por tabla oficial de premios de Tris con multiplicador.
  - Persistir el multiplicador real del sorteo en el histórico para evaluar ROI correctamente.
- **Integración en CLI/UI**:
  - Selector de juego al inicio de sesión.
  - Menú de producción y liquidación que use reglas y ledger por juego.
