import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

from EMIT_reader import read_EMIT
from RT_functions import read_luts_libradtran, get_k_libradtran
from Retrieval_methods import mf_retrieval  # o lmf_retrieval si prefieres la versión log
import variable_definition

# ---- 1) Ruta a tu radiancia EMIT ----
p = "C:/Users/jefer/OneDrive/Documentos/GitHub/HS_tool/"         # con barra final
n = "EMIT_L1B_RAD_001_20241216T085215_2435106_004"  # nombre sin extensión

img, wvl, fwhm, sza, vza = read_EMIT(p, n)
print(f"Cubo: {img.shape}, SZA={sza:.1f}°, VZA={vza:.1f}°")

# ---- 2) Ventana espectral para NH3 ----

wvl_inf, wvl_sup = 1400, 2500
window = np.where((wvl >= wvl_inf) & (wvl <= wvl_sup))[0]

# ---- 3) LUT de NH3 y cálculo de k ----
p_lut_mf = variable_definition.p_lut  # ajusta si es distinto en tu variable_definition.py
fn_lut_mf = p_lut_mf + 'LUT_ssd_100pm_wvl_1399_2500_nm_nh3.nc'

h, d_h2o = 0, 3  # mismos defaults que usa tu propio código
wvl_hr, rad_hr_arr, delta_x_arr = read_luts_libradtran(fn_lut_mf, sza, vza, d_h2o, h)
wvl_ret, k_arr = get_k_libradtran(img, wvl, fwhm, wvl_hr, rad_hr_arr, delta_x_arr, window, mission='EMIT')

# ---- 4) Matched filter ----
ret = mf_retrieval(img, k_arr, window, mission_name='EMIT', bool_libradtran=True)  # salida en ppm·m

# ---- 5) Lat/lon por pixel (directo del .nc, sin escribir placemark) ----
ds = nc.Dataset(p + n + '.nc')
lat = np.array(ds['location']['lat'])
lon = np.array(ds['location']['lon'])
ds.close()

# ---- 6) Plot georreferenciado ----
plt.figure(figsize=(8, 10))
std = np.nanstd(ret)
im = plt.pcolormesh(lon, lat, ret, cmap='plasma', vmin=0, vmax=2*std, shading='auto')
plt.colorbar(im, label=r'$\Delta$XNH$_3$ (ppm·m)')
plt.xlabel('Longitud'); plt.ylabel('Latitud')
plt.title(f'NH3 enhancement - {n}')
plt.tight_layout()
plt.show()