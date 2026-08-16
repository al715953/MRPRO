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

## Laboratorio de covering designs

El menú Melate incluye la opción `C`, y también puede ejecutarse directamente:

```bash
python run_covering_experiment.py \
  --v 15 \
  --t k-1 \
  --budget 300 \
  --candidate-method oracle_candidate_set \
  --random-trials 100 \
  --draws 108
```

Para barridos matemáticos sin backtest:

```bash
python run_covering_experiment.py \
  --mode math \
  --v 10,12,15,18,20 \
  --t k-1,k-2 \
  --budget 50,100,200,300,500,1000
```

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
