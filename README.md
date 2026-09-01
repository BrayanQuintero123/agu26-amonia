#Cuantificación semi-automática de plumas de CH₄ en Tanager

Detectar una pluma de metano ya es rutina; convertirla en un kg/h defendible no lo es.
Este repo es un pipeline abierto que va del cubo de radiancia de **Tanager** a un caudal
por pluma, dejando al humano exactamente donde hace falta criterio y en ningún otro sitio.

**El algoritmo delimita, el operador confirma.**

```
radiancia L1  →  matched filter (ppm·m)  →  anomalía wavelet  →  candidatos k·σ
                                                                      ↓
                              Q (kg/h)  ←  IME · Ueff / L  ←  máscara por componente conexa
```

## Empezar por aquí

Un solo comando, que elige solo el camino según lo que traiga cada escena:

```bash
python auto_pipeline/quantify_tanager.py tanager/*_ortho_radiance_hdf5.h5
```

| La escena trae… | Camino | Qué pasa |
|---|---|---|
| `ql_ch4_json` **con** plumas | automático | Se siembra de las fuentes oficiales, sin clics. |
| sin GeoJSON, o vacío | picker | Propone candidatos sobre el mapa wavelet; tú eliges con un clic. |

`--dry-run` dice qué haría sin correr nada. `--mode gt` / `--mode pick` lo fuerzan.

### El picker, paso a paso

```bash
python auto_pipeline/pick_plumes_wavelet.py <escena_ortho_radiance_hdf5.h5>
```

1. Corre **los dos** matched filters y saca 4 paneles: anomalía wavelet y realce físico, para cada uno.
2. Propone como candidato **todo cúmulo** sobre `k·σ` con ≥ `--min-pix` píxeles, tomando la
   **unión de los dos** MF para que ninguna pluma se pierda porque un retrieval no la vio.
   Cada candidato sale numerado, coloreado por quién lo detectó (amarillo = solo MF por columna,
   azul = solo MF por grupos, verde = los dos) y con **su link de Google Maps**, para descartar
   falsos positivos antes de cuantificar nada.
3. Un clic hace crecer la máscara por componente conexa. Aceptas, ajustas σ, o rechazas.
4. Cuantifica con los dos MF sobre la **misma** máscara y el **mismo** viento (ERA5 por defecto,
   uno por pluma en su propia posición) y guarda CSV + PNG.

## Resultados

`results/` trae las 20 plumas de Carbon Mapper de dos escenas Tanager, cuantificadas por
este pipeline y contrastadas contra el producto oficial.

### Por qué comparar Q contra Q no significa lo que parece

Despejando `Q = IME·3600·Ueff/L` sobre las 20 plumas del GeoJSON sale, **exacto y sin
excepción**, `Ueff = u10` y `L = fetch`. La parametrización clásica del repo
(`Ueff = a·u10 + b`, Guanter et al. 2021 / Roger et al. 2024) es 1.8–2.4× menor a esos vientos.

Re-derivando Q bajo cuatro combinaciones defendibles de máscara y escala de longitud:

| máscara | L | LARS/oficial | ADV/oficial | ADV/LARS |
|---|---|---|---|---|
| huella quicklook | fetch | 2.01 | 2.43 | 1.26 |
| huella quicklook | √área | 2.07 | 2.37 | 1.14 |
| crecida k·σ propia | √área | 2.07 | 2.06 | 1.00 |
| crecida k·σ propia | fetch | 1.46 | 0.92 | 0.63 |

El cociente contra el oficial se mueve entre **0.92× y 2.43×** sin tocar un solo dato.
Eso no es error de medida, es convención — y ninguna comparación entre inventarios significa
nada si no se declara la máscara y el Ueff.

### Los dos matched filters

Los dos comparten el target `t = mu · k` (con `k` de la LUT de libRadtran), así que sus
salidas están en ppm·m y son comparables. Solo cambia cómo se estima el **fondo**:

| | Fondo |
|---|---|
| **por columna** (`Retrieval_methods.AT_MF`) | una covarianza por columna, `pinv`, sin regularizar |
| **por grupos** (`scripts/mf_advanced.py`) | columnas agrupadas (10–30) según el striping; dentro de cada grupo PCA(3) + k-means, una covarianza por clúster con shrinkage |

Con la máscara controlada, el de grupos recupera 14–26 % más masa, pero:

- SNR **peor** en las dos escenas (0.71 vs 1.12, y 3.10 vs 3.97)
- cola del fondo 4–9× más pesada (p99.9 = 3251 y 8171 vs 804 y 882 ppm·m)
- **no delinea 3 de 20 plumas** al umbral estándar de 2σ
- Q negativa en una pluma real

Agrupar columnas para tener más muestras por covarianza no compensa el ruido que introduce.

```bash
python auto_pipeline/compare_mf.py <escenas.h5> --per-plume --slim
```

### Convención de cuantificación

| Flag | Opciones | Default |
|---|---|---|
| `--ueff-mode` | `cm` (Ueff = u10, la de Carbon Mapper) · `lars` (a·u10 + b) | `cm` |
| `--l-mode` | `sqrt-area` · `fetch` · `hull` | `sqrt-area` |
| `--mask-mode` | `footprint` (aísla el retrieval) · `grow` (pipeline completo) | `footprint` |

`sqrt-area` = `√(N·gsd²)`, que es lo que ya usaba `extract_Q`, y contra el `fetch` oficial de
las 20 plumas da mediana **0.92**. El convex hull da 1.73 y se dispara hasta 5× en plumas
alargadas, así que no es comparable con el producto oficial.

## Por qué Tanager

- **30 m** frente a los 60 m de EMIT: 4× píxeles por pluma. La calidad de la máscara entra
  directo en Q por partida doble, vía IME y vía L.
- **Cobertura.** EMIT observa únicamente entre 52°N y 52°S — es la inclinación de la ISS.
  La escena de Yamal de este repo está a **66.5°N**, 14° fuera de ese límite: sus 8 plumas
  son inobservables desde la ISS, no por agenda sino por órbita.

  > "During its planned one-year mission, EMIT will collect data over Earth's dust-source
  > regions […] between 52° north and south latitude."
  > — Smith, J. M. (2023), *Meet EMIT, the Newest Imaging Spectrometer*, NASA Earthdata.

## Estructura

| | |
|---|---|
| `auto_pipeline/quantify_tanager.py` | **punto de entrada**; enruta GT vs picker |
| `auto_pipeline/pick_plumes_wavelet.py` | picker interactivo con los dos MF |
| `auto_pipeline/compare_mf.py` | comparación de los dos MF contra el GT |
| `auto_pipeline/plumes_from_json.py` | siembra de plumas desde `ql_ch4_json` |
| `auto_pipeline/wavelet_den.py` | anomalía wavelet |
| `auto_pipeline/auto_plume.py` | delineación semi-automática (EMIT y genérico) |
| `scripts/mf_advanced.py` | matched filter por grupos de columnas |
| `scripts/Retrieval_methods.py` | matched filter por columna |
| `scripts/Tanager_reader.py` | lector Tanager (ortho y basic) |
| `scripts/quant_func_v2.py` | IME, Ueff, conversión a kg |
| `results/` | las 20 plumas cuantificadas, 4 configuraciones |

Otras misiones (EMIT, EnMAP, PRISMA, GF5, AVIRIS-NG) siguen soportadas vía
`auto_pipeline/pipeline_auto.py`; ver la sección al final.

Gases cuantificables: `ch4`, `co2`, `c2h4`, `c2h2`, `nh3`.

## Instalación

Entorno conda con GDAL, rasterio, h5py, pyproj, scikit-image, scikit-learn, PyWavelets,
scipy, matplotlib. En Windows hay que **activar el entorno**, no basta con invocar el
`python.exe`: sin el `PATH` del entorno, LAPACK no carga sus DLL y `np.linalg.lstsq`
revienta con `0xc06d007f`.

```bash
conda activate <tu-entorno>
python auto_pipeline/quantify_tanager.py --help
```

Hacen falta además las LUT de libRadtran (ruta en `scripts/variable_definition.py`).

## Otras misiones (EMIT, EnMAP, PRISMA, GF5, AVIRIS-NG)

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
