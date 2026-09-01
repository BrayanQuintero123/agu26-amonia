import numpy as np
import matplotlib.pyplot as plt

from Tanager_reader import read_Tanager, Tanager_geometry, swath_angle
from RT_functions import read_luts_libradtran, get_k_libradtran
from Retrieval_methods import mf_retrieval
import variable_definition

MISSION = 'Tanager'

# ---- 1) Ruta a tu radiancia Tanager (ortho_radiance_hdf5) ----
p = "C:/Users/jefer/OneDrive/Documentos/GitHub/agu26-amonia2/TanagerScene 20241004_081921_28_4001 Core Imagery/"
n = "20241004_081921_28_4001_ortho_radiance_hdf5"  # nombre sin extension

img, wvl, fwhm, sza, vza = read_Tanager(p, n, mask_clouds=True)
geom = Tanager_geometry(p, n)
print(f"Cubo: {img.shape}, SZA={sza:.1f}deg, VZA={vza:.1f}deg")
print(f"Pixel: {geom['pix_size_m']} m | EPSG:{geom['epsg']}")
print(f"Inclinacion del swath respecto a las columnas: {swath_angle(geom['nodata']):.1f} deg")

# ---- 2) Ventana espectral para NH3 ----
wvl_inf, wvl_sup = 1400, 2495  # Tanager llega a 2499 nm: dejamos margen para la convolucion
window = np.where((wvl >= wvl_inf) & (wvl <= wvl_sup))[0]
print(f"Bandas en ventana: {len(window)} ({wvl[window][0]:.1f}-{wvl[window][-1]:.1f} nm)")

# ---- 3) LUT de NH3 y calculo de k ----
fn_lut_mf = variable_definition.p_lut + 'LUT_ssd_100pm_wvl_1399_2500_nm_nh3.nc'

h, d_h2o = 0, 3
wvl_hr, rad_hr_arr, delta_x_arr = read_luts_libradtran(fn_lut_mf, sza, vza, d_h2o, h)
wvl_ret, k_arr = get_k_libradtran(img, wvl, fwhm, wvl_hr, rad_hr_arr,
                                  delta_x_arr, window, mission=MISSION)

# ---- 4) Matched filter ----
ret = mf_retrieval(img, k_arr, window, mission_name=MISSION, bool_libradtran=True)  # ppm.m
ret[~np.isfinite(img[:, :, window[0]])] = np.nan  # no pintamos el relleno del swath

# ---- 5) Plot en la malla UTM nativa ----
plt.figure(figsize=(8, 10))
std = np.nanstd(ret)
im = plt.imshow(ret, cmap='plasma', vmin=0, vmax=2*std, extent=geom['extent'])
plt.colorbar(im, label=r'$\Delta$XNH$_3$ (ppm$\cdot$m)')
plt.xlabel(f"Easting (m, EPSG:{geom['epsg']})"); plt.ylabel('Northing (m)')
plt.title(f'NH3 enhancement - {n}')
plt.tight_layout()
plt.show()
