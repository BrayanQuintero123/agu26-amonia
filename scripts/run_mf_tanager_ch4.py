"""Matched filter CH4 sobre Tanager ortho_radiance_hdf5."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

from Tanager_reader import read_Tanager, get_ortho_framing
from RT_functions import read_luts_libradtran, get_k_libradtran
from Retrieval_methods import mf_retrieval
import variable_definition

MISSION = 'Tanager'
p = "C:/Users/jefer/OneDrive/Documentos/GitHub/agu26-amonia2/TanagerScene 20241004_081921_28_4001 Core Imagery/"
n = "20241004_081921_28_4001_ortho_radiance_hdf5"
OUT = "C:/Users/jefer/OneDrive/Documentos/GitHub/agu26-amonia2/out_tanager/"
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

# 1) Leer cubo y recortar a ventana CH4 (LUT: 2049-2500 nm)
img_full, wvl_all, fwhm_all, sza, vza = read_Tanager(p, n)
window = np.where((wvl_all >= 2050) & (wvl_all <= 2495))[0]
img  = img_full[:, :, window]
wvl  = wvl_all[window]
fwhm = fwhm_all[window]
del img_full
print(f"[{time.time()-t0:6.1f}s] ventana CH4: {len(window)} bandas "
      f"({wvl[0]:.1f}-{wvl[-1]:.1f} nm) | cubo {img.shape} "
      f"| SZA={sza:.1f} VZA={vza:.1f}", flush=True)

# 2) k desde LUT libRadtran CH4
fn_lut = variable_definition.p_lut + 'LUT_ssd_100pm_wvl_2049_2500_nm_ch4.nc'
wvl_hr, rad_hr_arr, delta_x_arr = read_luts_libradtran(fn_lut, sza, vza, 3, 0)
win_idx = np.arange(len(window))
_, k_arr = get_k_libradtran(img, wvl, fwhm, wvl_hr, rad_hr_arr, delta_x_arr,
                             win_idx, mission=MISSION)
print(f"[{time.time()-t0:6.1f}s] k listo: {np.asarray(k_arr).shape}", flush=True)

# 3) Matched filter
ret = mf_retrieval(img, k_arr, win_idx, mission_name=MISSION, bool_libradtran=True)
ret[~np.isfinite(img[:, :, 0])] = np.nan
v   = ret[np.isfinite(ret)]
std = float(np.nanstd(ret))
print(f"[{time.time()-t0:6.1f}s] MF CH4: std={std:.1f} ppm.m "
      f"| p99={np.percentile(v,99):.1f} | p99.9={np.percentile(v,99.9):.1f} "
      f"| max={v.max():.1f}", flush=True)
np.save(OUT + 'mf_ch4.npy', ret)

# 4) GeoTIFF directo (ya orto-rectificado)
geo = get_ortho_framing(p, n)
gt, epsg = geo['geotransform'], geo['epsg_code']
try:
    import rasterio
    from rasterio.transform import Affine
    aff = Affine(gt[1], gt[2], gt[0], gt[4], gt[5], gt[3])
    with rasterio.open(OUT + 'mf_ch4.tif', 'w', driver='GTiff',
                       height=ret.shape[0], width=ret.shape[1], count=1,
                       dtype='float32', crs=f'EPSG:{epsg}', transform=aff,
                       nodata=float('nan'), compress='deflate') as dst:
        dst.write(ret.astype('float32'), 1)
    print(f"[{time.time()-t0:6.1f}s] GeoTIFF -> {OUT}mf_ch4.tif", flush=True)
except Exception as e:
    print(f"  GeoTIFF omitido: {e}")

# 5) Quicklook
rows, cols = ret.shape
extent = [gt[0], gt[0] + cols*gt[1], gt[3] + rows*gt[5], gt[3]]
fig, ax = plt.subplots(figsize=(10, 12))
im = ax.imshow(ret, cmap='inferno', vmin=0, vmax=3*std, extent=extent)
plt.colorbar(im, ax=ax, label=r'$\Delta$XCH$_4$ (ppm$\cdot$m)', shrink=0.7)
ax.set_xlabel(f'Easting (m, EPSG:{epsg})'); ax.set_ylabel('Northing (m)')
ax.set_title(f'CH4 matched filter - Tanager\n{n}')
fig.tight_layout()
fig.savefig(OUT + 'mf_ch4.png', dpi=140)
print(f"[{time.time()-t0:6.1f}s] LISTO -> {OUT}", flush=True)
