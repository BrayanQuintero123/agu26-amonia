# agu26-amonia

Pipeline semi-automático de cuantificación multi-gas para EMIT (extensión de HS_tool).

- `auto_pipeline/` — delineación semi-automática (den wavelet para localizar + crecimiento por umbral/componente conexa desde el hotspot + double-check) y cuantificación IME. Punto de entrada: `pipeline_auto.py` (`--gas ch4,co2,c2h4,c2h2,nh3`). `plot_plume_map.py` superpone la pluma sobre satélite/OSM/radiancia.
- `scripts/` — solo los 4 archivos originales de HS_tool que se modificaron (fill-value EMIT, warp norte-arriba, etc.); el resto del repo HS_tool es necesario para correr.

Uso:
```
python auto_pipeline/pipeline_auto.py "<EMIT_L1B_RAD...nc>" --gas ch4
python auto_pipeline/plot_plume_map.py --lat <lat> --lon <lon> --gas ch4 --sigma 2.0
```
