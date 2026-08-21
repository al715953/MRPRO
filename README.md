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
