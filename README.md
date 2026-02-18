# MRPRO

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

