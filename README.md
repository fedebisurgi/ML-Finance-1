# ML-Finance-1 — Target3 LightGBM LambdaRank pipeline

Pipeline de predicción semanal Top-20 (Target3) basado en LightGBM con blend
classifier + LambdaRank, walk-forward CV, holdout dedicado y filtros de universo.

## Workflows

El código está dividido en un módulo compartido (`target3_core.py`) y dos entry
points que se corren bajo distintas condiciones:

| Entry point | Cuándo correrlo | Duración aprox. |
|-------------|-----------------|-----------------|
| `python target3_predict.py` | **Todos los viernes** después del cierre, una vez que el Excel consolidado tiene la última semana. Entrena los 3 modelos (BASELINE / ABLATION / NEW) y exporta los 3 Excels con el Top-20 productivo. | ~15 min |
| `python target3_backtest.py` | **Bajo demanda** cuando se modifican features, hiperparámetros o filtros. Corre walk-forward sobre las fechas configuradas en `BACKTEST_DATES` y produce 3 Excels con métricas de atribución (spearman, decile spread, top-N concentration). | Proporcional a `len(BACKTEST_DATES)` (horas con 18 fechas) |

`target3_core.py` no se ejecuta directamente — solo se importa.

## Variantes que corren en cada workflow

Las tres variantes son idénticas entre `predict` y `backtest`; cambia únicamente
qué se entrena (1 vez vs N veces walk-forward):

- **BASELINE**: features históricas + filtro `Close > $5`.
- **ABLATION**: features históricas + filtros de universo completos
  (`Close > $5`, `Dollar_Volume_20d > $5M`, `Zero_Return_Days_Pct_60d < 20%`).
- **NEW**: features históricas + 4 nuevas (Amihud illiquidity, MAX5, Information
  Discreteness) + filtros completos.

## Archivos

```
target3_core.py                    # Módulo compartido (helpers, modelo, FE, run_pipeline, run_backtest_loop)
target3_predict.py                 # Entry point semanal
target3_backtest.py                # Entry point walk-forward (ajustar BACKTEST_DATES adentro)
target3_mvp_universe_filters.py    # [DEPRECATED] Script monolítico — eliminar tras sprint
```

## Configuración

Constantes principales en `target3_core.py` (sección `CONFIGURATION`):
`BASE_DIR`, `FILE_NAME`, `TOP_K`, `N_FOLDS`, `EMBARGO_DATES`, `HOLDOUT_N_WEEKS`,
`SEED`, filtros de universo, etc. Los nombres de archivo Excel se generan con
`DATE_STAMP` + `INPUT_HASH` para reproducibilidad.
