"""Matched filter NH3 sobre Tanager en geometria de sensor (basic_radiance_hdf5).

A diferencia del ortho, aqui las columnas de la imagen son columnas reales del
detector, que es lo que AT_MF asume. El resultado se reproyecta al final a la
misma malla UTM del producto ortho para poder comparar los dos.
"""
import os, time
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

from Tanager_reader import (read_Tanager_basic, read_Tanager_basic_bands,
                            Tanager_basic_geometry)
from RT_functions import read_luts_libradtran, get_k_libradtran
from Retrieval_methods import mf_retrieval
import variable_definition

MISSION = 'Tanager'
p = "C:/Users/jefer/OneDrive/Documentos/GitHub/agu26-amonia2/TanagerScene 20241004_081921_28_4001 Core Imagery/"
n = "20241004_081921_28_4001_basic_radiance_hdf5"
OUT = "C:/Users/jefer/OneDrive/Documentos/GitHub/agu26-amonia2/out_tanager/"
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

# 1) Ventana espectral NH3
wvl_all, _ = read_Tanager_basic_bands(p, n)
window_full = np.where((wvl_all >= 1400) & (wvl_all <= 2495))[0]
img, wvl, fwhm, sza, vza = read_Tanager_basic(p, n, mask_clouds=True, bands=window_full)
geom = Tanager_basic_geometry(p, n)
window = np.arange(len(window_full))
print(f"[{time.time()-t0:6.1f}s] cubo {img.shape} ({img.nbytes/1e9:.2f} GB) "
      f"| SZA={sza:.1f} VZA={vza:.1f} | validos {np.isfinite(img[:,:,0]).mean():.1%}", flush=True)

# 2) k desde la LUT
fn_lut = variable_definition.p_lut + 'LUT_ssd_100pm_wvl_1399_2500_nm_nh3.nc'
wvl_hr, rad_hr_arr, delta_x_arr = read_luts_libradtran(fn_lut, sza, vza, 3, 0)
_, k_arr = get_k_libradtran(img, wvl, fwhm, wvl_hr, rad_hr_arr, delta_x_arr,
                            window, mission=MISSION)

# 3) Matched filter en geometria de sensor
ret = mf_retrieval(img, k_arr, window, mission_name=MISSION, bool_libradtran=True)
ret[~np.isfinite(img[:, :, 0])] = np.nan
v = ret[np.isfinite(ret)]
print(f"[{time.time()-t0:6.1f}s] MF sensor-geometry: std={v.std():.1f} ppm.m "
      f"| p99.9={np.percentile(v,99.9):.1f} | max={v.max():.1f}", flush=True)
np.save(OUT + 'mf_nh3_basic_sensorgeom.npy', ret)

# 4) Reproyeccion a la malla ortho que Planet declara en el propio fichero
import pyproj
from scipy.interpolate import griddata

fr = geom['ortho_framing']
gt, rows, cols, epsg = fr['geotransform'], fr['rows'], fr['cols'], fr['epsg_code']
tr = pyproj.Transformer.from_crs('EPSG:4326', f"EPSG:{epsg}", always_xy=True)
xs, ys = tr.transform(geom['lon'], geom['lat'])

ok = np.isfinite(ret) & np.isfinite(xs) & np.isfinite(ys)
gx = gt[0] + (np.arange(cols) + 0.5) * gt[1]
gy = gt[3] + (np.arange(rows) + 0.5) * gt[5]
GX, GY = np.meshgrid(gx, gy)
ret_ortho = griddata((xs[ok], ys[ok]), ret[ok], (GX, GY), method='linear')
print(f"[{time.time()-t0:6.1f}s] reproyectado a {ret_ortho.shape} EPSG:{epsg}", flush=True)
np.save(OUT + 'mf_nh3_basic.npy', ret_ortho)

# 5) GeoTIFF + quicklook
import rasterio
from rasterio.transform import Affine
aff = Affine(gt[1], gt[2], gt[0], gt[4], gt[5], gt[3])
with rasterio.open(OUT + 'mf_nh3_basic.tif', 'w', driver='GTiff', height=rows,
                   width=cols, count=1, dtype='float32', crs=f"EPSG:{epsg}",
                   transform=aff, nodata=np.nan, compress='deflate') as dst:
    dst.write(ret_ortho.astype('float32'), 1)

std = np.nanstd(ret_ortho)
extent = [gt[0], gt[0] + cols*gt[1], gt[3] + rows*gt[5], gt[3]]
fig, ax = plt.subplots(figsize=(9, 11))
im = ax.imshow(ret_ortho, cmap='plasma', vmin=0, vmax=2*std, extent=extent)
plt.colorbar(im, ax=ax, label=r'$\Delta$XNH$_3$ (ppm$\cdot$m)', shrink=0.7)
ax.set_xlabel(f'Easting (m, EPSG:{epsg})'); ax.set_ylabel('Northing (m)')
ax.set_title('NH3 MF - basic_radiance (geometria de sensor)')
fig.tight_layout(); fig.savefig(OUT + 'mf_nh3_basic.png', dpi=140)
print(f"[{time.time()-t0:6.1f}s] LISTO", flush=True)
