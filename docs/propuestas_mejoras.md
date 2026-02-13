# Propuestas de mejora MRPRO (revisión técnica)

## 1) Estabilizar el optimizador ante retornos heterogéneos
- En `StrategyOptimizer.optimize_filters`, se asume que `reduce(...)` devuelve siempre una colección; sin embargo, el test de regresión inyecta un `reduce` que devuelve una tupla `(universe, debug)`.
- Esto rompe el cálculo de `u_size` y puede terminar en `KeyError` al no poblar métricas de resumen.
- Propuesta: normalizar la salida de `reduce` (aceptar `ndarray` o `tuple`) y proteger el resumen final con valores por defecto cuando no hay candidato válido.

## 2) Endurecer manejo de errores
- Hay varios `except:` genéricos en flujo principal/controlador, lo que oculta causas reales y complica soporte.
- Propuesta: capturar excepciones específicas (`ValueError`, `FileNotFoundError`, `requests.RequestException`) y registrar detalle técnico en log estructurado.

## 3) Separar UI/CLI de lógica de dominio
- `MissionController` mezcla interacción de consola, orquestación de casos de uso y lógica de negocio.
- Propuesta: extraer servicios de aplicación (casos de uso) y dejar al controlador solo como adaptador de entrada/salida.

## 4) Corregir inconsistencia de rutas de datos
- `config.py` define data bajo `BASE_DIR/data`, pero `train_static_model.py` busca primero `src/data/Melate-Retro.csv`.
- Propuesta: centralizar rutas en una sola fuente de verdad (`config.py`) y eliminar rutas duplicadas/fallbacks ambiguos.

## 5) Limpiar dependencias CUDA conflictivas
- `requirements.txt` incluye `cupy-cuda11x` y `cupy-cuda12x` a la vez, lo cual genera advertencias y posibles fallos en runtime.
- Propuesta: dividir dependencias por extras (`requirements-gpu11.txt`, `requirements-gpu12.txt`) o usar instalación condicional documentada.

## 6) Seguridad en scraping y TLS
- El scraper hace bypass SSL (`verify=False`) en fallback automático.
- Propuesta: mantener verificación activa por defecto, agregar reintentos con backoff, timeout configurable y registro explícito de cuándo se desactiva TLS.

## 7) Cobertura y estrategia de pruebas
- La suite actual es mínima y ya detecta regresiones reales del optimizador.
- Propuesta: añadir pruebas unitarias por capa (loader, filtros, optimizador, reportes) y una batería smoke para CLI no interactiva.

## 8) Documentación operativa mínima
- `README.md` está vacío.
- Propuesta: agregar guía de instalación, ejecución, arquitectura, comandos operativos y troubleshooting.

## 9) Observabilidad básica
- El proyecto usa `print`/Rich para estado operativo, pero sin trazabilidad persistente.
- Propuesta: logging estructurado con niveles (INFO/WARN/ERROR), archivo rotativo y correlación por misión/sorteo.

## 10) Calidad de configuración
- `BEST_SETTINGS` es un diccionario grande embebido en código.
- Propuesta: mover configuración a YAML/TOML versionado y validar esquema (pydantic/dataclasses) antes de ejecutar.
