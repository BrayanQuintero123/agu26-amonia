# agu26-amonia

Pipeline semi-automático de cuantificación multi-gas para imágenes hiperespectrales
(extensión de HS_tool). Soporta **EMIT** y **Tanager**, y la infraestructura de
lectura cubre además EnMAP, PRISMA, GF5 y AVIRIS-NG.

El flujo general es: radiancia L1 → *matched filter* (enhancement en ppm·m) →
localización de la fuente → delineación de la pluma → cuantificación IME → caudal `Q` en kg/h.

## Estructura

- `auto_pipeline/` — delineación semi-automática (wavelet para localizar + crecimiento
  por umbral/componente conexa desde el hotspot + double-check) y cuantificación IME.
  Punto de entrada: `pipeline_auto.py`. `plot_plume_map.py` superpone la pluma sobre
  satélite/OSM/radiancia.
- `scripts/` — lectores por misión (`Tanager_reader.py`, `EMIT_reader.py`, …),
  retrieval L1→L2 (`hs_tool4_deltaxgas.py`), cuantificación (`quant_func_v2.py`),
  viento (`wind_v2.py`) y el flujo interactivo multi-pluma.

Gases cuantificables (tienen calibración de Ueff + masa molar): `ch4`, `co2`, `c2h4`, `c2h2`, `nh3`.

## Uso

### Pipeline principal (una pluma por gas)

```bash
python auto_pipeline/pipeline_auto.py "<EMIT_L1B_RAD...nc>" --gas ch4
python auto_pipeline/pipeline_auto.py "<...ortho_radiance_hdf5.h5>" --gas ch4 --mission Tanager --wind era5
python auto_pipeline/pipeline_auto.py "<RAD.nc>" --gas ch4,nh3,c2h4 -s "Carrasco"
```

Flags útiles:

| Flag | Descripción |
|---|---|
| `--gas` | Gas o lista separada por comas. |
| `--mission` | `EMIT`, `Tanager`, `EnMAP`, `PRISMA`, `GF5`, `AVIRIS-NG`. Si se omite, se auto-detecta del nombre del archivo. |
| `--wind {geos,era5}` | Fuente de viento. `--era5` es atajo de `--wind era5` y busca el `.nc` de ERA5 junto al archivo de entrada. |
| `--wind-value` | Viento manual en m/s, solo para esa corrida. |
| `--sigma` | Multiplicador de sigma del umbral de la pluma (default 2.0). |

Mapa de la pluma sobre imagen de fondo:

```bash
python auto_pipeline/plot_plume_map.py --lat <lat> --lon <lon> --gas ch4 --sigma 2.0
```

### Flujo interactivo multi-pluma (Tanager, CH4)

Cuando una escena trae **varias plumas**, este script las detecta todas y deja elegir
cuáles cuantificar:

```bash
python scripts/quantify_ch4_tanager_multi.py
```

Qué hace:

1. Aplica el denoiser wavelet sobre el mapa de enhancement.
2. Se queda con los máximos locales sobre el percentil **P99.8** que además superen
   **2.5σ** en el enhancement crudo, y suprime candidatos a menos de **1000 m** de uno
   más intenso — así una sola pluma no genera varios puntos.
3. Imprime una tabla numerada con un **link de Google Maps por candidato**, para revisar
   uno a uno y descartar falsos positivos antes de marcar nada.
4. Abre una ventana con los paneles *enhancement* + *wavelet* y los candidatos marcados.
5. **Un click izquierdo dentro de una pluma** hace crecer su máscara automáticamente
   (`grow_plume_from_source`: mediana 3×3, umbral a `sigma·std`, componente conexa).
   No hay que dibujar polígonos. Click derecho deshace, Enter termina.
6. Cuantifica cada pluma seleccionada (IME + Ueff + viento) y muestra una tabla resumen;
   guarda los resultados en CSV y la figura en `out_tanager/plumas_seleccionadas.png`.

Ajustes: `--pctl 99.9` (más estricto), `--n-sigma 3`, `--min-dist 1500`, `--top 15`,
`--sigma 2.0`, `--wind-value 3.2` (evita la descarga de GEOS-FP).

## Nota sobre Tanager

`Tanager_reader.py` auto-detecta el tipo de producto:

- `ortho_radiance_hdf5` → `HDFEOS/GRIDS/HYP`, geotransform y EPSG leídos del `StructMetadata`.
- `basic_radiance_hdf5` → `HDFEOS/SWATHS/HYP`, con `Latitude`/`Longitude` embebidos;
  la malla se deriva del *bounding box* lat/lon.

Tanager usa píxel de **30 m**, con los mismos coeficientes de Ueff que EnMAP/PRISMA/GF5.

## Datos

Los archivos de datos **no** se versionan: los `.h5` de Tanager pesan 580–682 MB cada uno
y el `.nc` de EMIT ~1.8 GB, muy por encima del límite de 100 MB de GitHub. El `.gitignore`
excluye `*.nc`, `*.h5`, `*.tif`, `*.npy`, `TanagerScene*/`, `out_tanager/` y `output/`.

Para correr se necesita además el repo original de HS_tool y las LUT de libRadtran.
