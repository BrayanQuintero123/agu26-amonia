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

### Comparación de matched filters (Tanager)

El repo trae dos retrievals que comparten el mismo target `t = mu · k` (con `k` de la LUT
de libRadtran), así que sus salidas están en ppm·m y son directamente comparables. Lo único
que cambia es cómo se estima el **fondo**:

| | Fondo |
|---|---|
| **LARS** (`Retrieval_methods.AT_MF`) | una covarianza por columna, `pinv`, sin regularizar |
| **ADV** (`scripts/mf_advanced.py`) | columnas agrupadas (10–30) según el perfil de striping; dentro de cada grupo PCA(3) + k-means, una covarianza por clúster con shrinkage |

Punto de entrada único, que elige solo el camino según lo que traiga cada escena:

```bash
python auto_pipeline/quantify_tanager.py tanager/*_ortho_radiance_hdf5.h5
python auto_pipeline/quantify_tanager.py tanager/*.h5 --dry-run      # solo dice qué haría
```

- **Con GT** (existe `<scene>_ql_ch4_json.geojson` *con* features) → `compare_mf.py`.
  Las plumas se siembran del GeoJSON (fuente, viento y emisión oficiales), la máscara sale
  de la huella del quicklook oficial y se cuantifica con los dos MF. Sin clics.
- **Sin GT** (no hay GeoJSON, o viene vacío) → `pick_plumes_wavelet.py`.
  Se corren los dos MF, se proponen candidatos de la **unión** de ambos sobre el mapa
  wavelet, cada uno con su link de Google Maps, y el operador elige con un clic.

Forzable con `--mode gt` / `--mode pick`.

### Convención de cuantificación

`Q = IME · 3600 · Ueff / L`. Despejando esa fórmula sobre las 20 plumas con GT de Carbon
Mapper sale, exacto y sin excepción, **`Ueff = u10`** y **`L = fetch`**. La parametrización
original del repo (`Ueff = a·u10 + b`, Guanter et al. 2021 / Roger et al. 2024) da 1.8–2.4×
menos a esos vientos, y ese factor explicaba casi todo el hueco que se veía contra el
producto oficial — no era el retrieval.

Por eso el default ahora es la convención de Carbon Mapper:

| Flag | Opciones | Default |
|---|---|---|
| `--ueff-mode` | `cm` (Ueff = u10) · `lars` (a·u10 + b) | `cm` |
| `--l-mode` | `sqrt-area` · `fetch` · `hull` | `sqrt-area` |

`sqrt-area` = `sqrt(N·gsd²)`, que es lo que ya usaba `extract_Q`, y contrastado contra el
`fetch` oficial de las 20 plumas da mediana **0.92** (el convex hull da 1.73 y se dispara
hasta 5× en plumas alargadas, así que no es comparable con el producto oficial).

`--per-plume` da una fila por pluma del GeoJSON; cuando varias comparten huella conexa, la
huella se reparte por fuente más cercana.

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
