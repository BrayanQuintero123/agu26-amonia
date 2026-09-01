"""Matched filter NH3 sobre la escena Tanager completa."""
import time
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

from Tanager_reader import read_Tanager, read_Tanager_bands, Tanager_geometry, swath_angle
from RT_functions import read_luts_libradtran, get_k_libradtran
from Retrieval_methods import mf_retrieval
import variable_definition

MISSION = 'Tanager'
p = "C:/Users/jefer/OneDrive/Documentos/GitHub/agu26-amonia2/TanagerScene 20241004_081921_28_4001 Core Imagery/"
n = "20241004_081921_28_4001_ortho_radiance_hdf5"
OUT = "C:/Users/jefer/OneDrive/Documentos/GitHub/agu26-amonia2/out_tanager/"

import os
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

# 1) Ventana espectral NH3 -> leemos SOLO esas bandas (el cubo entero no cabe en RAM)
wvl_all, fwhm_all = read_Tanager_bands(p, n)
window_full = np.where((wvl_all >= 1400) & (wvl_all <= 2495))[0]
print(f"[{time.time()-t0:6.1f}s] ventana: {len(window_full)} bandas "
      f"({wvl_all[window_full][0]:.1f}-{wvl_all[window_full][-1]:.1f} nm)", flush=True)

img, wvl, fwhm, sza, vza = read_Tanager(p, n, mask_clouds=True, bands=window_full)
geom = Tanager_geometry(p, n)
window = np.arange(len(window_full))   # el cubo ya viene recortado
print(f"[{time.time()-t0:6.1f}s] cubo {img.shape} {img.dtype} "
      f"({img.nbytes/1e9:.2f} GB) | SZA={sza:.1f} VZA={vza:.1f} "
      f"| tilt={swath_angle(geom['nodata']):.1f} deg", flush=True)

# 2) k desde la LUT de libRadtran
fn_lut = variable_definition.p_lut + 'LUT_ssd_100pm_wvl_1399_2500_nm_nh3.nc'
wvl_hr, rad_hr_arr, delta_x_arr = read_luts_libradtran(fn_lut, sza, vza, 3, 0)
wvl_ret, k_arr = get_k_libradtran(img, wvl, fwhm, wvl_hr, rad_hr_arr,
                                  delta_x_arr, window, mission=MISSION)
print(f"[{time.time()-t0:6.1f}s] k listo: {np.asarray(k_arr).shape}", flush=True)

# 3) Matched filter (dos pasadas: la segunda corrige la hipotesis de sparsity)
ret = mf_retrieval(img, k_arr, window, mission_name=MISSION, bool_libradtran=True)
ret[~np.isfinite(img[:, :, 0])] = np.nan
print(f"[{time.time()-t0:6.1f}s] MF hecho", flush=True)

valid = ret[np.isfinite(ret)]
std = np.nanstd(ret)
print(f"  std={std:.1f} ppm.m | p99.9={np.percentile(valid,99.9):.1f} | max={valid.max():.1f}")
print(f"  pixeles validos: {valid.size} ({valid.size/ret.size:.1%})")

np.save(OUT + 'mf_nh3.npy', ret)

# 4) GeoTIFF georreferenciado (EPSG de la escena)
try:
    import rasterio
    from rasterio.transform import from_origin
    ulx, px, _, uly, _, mpy = geom['transform']
    tr = from_origin(ulx, uly, px, -mpy)
    with rasterio.open(OUT + 'mf_nh3.tif', 'w', driver='GTiff',
                       height=ret.shape[0], width=ret.shape[1], count=1,
                       dtype='float32', crs=f"EPSG:{geom['epsg']}",
                       transform=tr, nodata=np.nan, compress='deflate') as dst:
        dst.write(ret.astype('float32'), 1)
    print(f"[{time.time()-t0:6.1f}s] GeoTIFF escrito", flush=True)
except Exception as e:
    print('  GeoTIFF omitido:', e)

# 5) Quicklook
fig, ax = plt.subplots(figsize=(9, 11))
im = ax.imshow(ret, cmap='plasma', vmin=0, vmax=2*std, extent=geom['extent'])
plt.colorbar(im, ax=ax, label=r'$\Delta$XNH$_3$ (ppm$\cdot$m)', shrink=0.7)
ax.set_xlabel(f"Easting (m, EPSG:{geom['epsg']})"); ax.set_ylabel('Northing (m)')
ax.set_title(f'NH3 matched filter - {n}')
fig.tight_layout()
fig.savefig(OUT + 'mf_nh3.png', dpi=140)
print(f"[{time.time()-t0:6.1f}s] LISTO -> {OUT}", flush=True)
