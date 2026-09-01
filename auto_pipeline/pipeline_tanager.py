#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline interactivo Tanager: MF + seleccion de multiples plumas + cuantificacion IME.

Uso:
  python auto_pipeline/pipeline_tanager.py "<.h5>" --gas ch4
  python auto_pipeline/pipeline_tanager.py "<.h5>" --gas ch4 --wind era5 --era5-file "<.nc>"
  python auto_pipeline/pipeline_tanager.py "<.h5>" --gas nh3 --wind-value 3.5 -s "Deir ez-Zor"

Modo AUTOMATICO (sin clics), sembrando desde el asset ql_ch4_json de la escena:
  python auto_pipeline/pipeline_tanager.py "<.h5>" --gas ch4 --auto-json
  python auto_pipeline/pipeline_tanager.py "<.h5>" --gas ch4 --auto-json --wind json --quality good

El archivo .h5 puede ser ortho_radiance_hdf5 o basic_radiance_hdf5; se auto-detecta.
Si el MF ya fue calculado (existe mf_<gas>.npy en --output-dir), se carga directamente.

Gases: ch4, co2, c2h4, c2h2, nh3.

Flujo por pluma:
  1. Enter para salir del zoom
  2. 1 click = fuente  -> imprime Google Maps en consola
  3. Clicks = poligono, Enter para cerrar
  -> pregunta si hay otra pluma (s/n)
Al terminar: cuantifica todas y guarda CSV.
"""

import os, sys, argparse, time

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(os.path.dirname(_HERE), 'scripts')
for _p in (_SCRIPTS, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import rasterio
from rasterio.warp import transform as rio_transform
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.path
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
import pyproj

from Tanager_reader import read_Tanager, get_ortho_framing
from RT_functions import read_luts_libradtran, get_k_libradtran
from Retrieval_methods import mf_retrieval
from quant_func_v2 import ppmm_to_kg, extract_ueff
from store_results import excel_info
from wind_v2 import wind_speed_manual
import variable_definition

MISSION = 'Tanager'

GAS_LUT = {
    'ch4':  ('LUT_ssd_100pm_wvl_2049_2500_nm_ch4.nc',  2050, 2495),
    'nh3':  ('LUT_ssd_100pm_wvl_1399_2500_nm_nh3.nc',  1400, 2495),
    'co2':  ('LUT_ssd_100pm_wvl_2049_2500_nm_ch4.nc',  2050, 2495),  # placeholder
    'c2h4': ('LUT_ssd_100pm_wvl_2049_2500_nm_ch4.nc',  2050, 2495),  # placeholder
    'c2h2': ('LUT_ssd_100pm_wvl_2049_2500_nm_ch4.nc',  2050, 2495),  # placeholder
}

# ------------------------------------------------------------------
# MF (calcula o carga de cache)
# ------------------------------------------------------------------

def run_or_load_mf(p, n, gas, psave):
    cache = os.path.join(psave, f'mf_{gas}.npy')
    geo = get_ortho_framing(p, n)

    if os.path.exists(cache):
        print(f'[MF] Cargando cache: {cache}')
        ret = np.load(cache)
        return ret, geo

    print(f'[MF] Calculando matched filter para {gas.upper()}...')
    t0 = time.time()
    img_full, wvl_all, fwhm_all, sza, vza = read_Tanager(p, n)

    lut_file, wvl_inf, wvl_sup = GAS_LUT[gas]
    window = np.where((wvl_all >= wvl_inf) & (wvl_all <= wvl_sup))[0]
    img  = img_full[:, :, window]
    wvl  = wvl_all[window]
    fwhm = fwhm_all[window]
    del img_full
    print(f'  cubo {img.shape} | SZA={sza:.1f} VZA={vza:.1f} | {len(window)} bandas', flush=True)

    fn_lut = variable_definition.p_lut + lut_file
    wvl_hr, rad_hr_arr, delta_x_arr = read_luts_libradtran(fn_lut, sza, vza, 3, 0)
    win_idx = np.arange(len(window))
    _, k_arr = get_k_libradtran(img, wvl, fwhm, wvl_hr, rad_hr_arr,
                                 delta_x_arr, win_idx, mission=MISSION)
    ret = mf_retrieval(img, k_arr, win_idx, mission_name=MISSION, bool_libradtran=True)
    ret[~np.isfinite(img[:, :, 0])] = np.nan
    np.save(cache, ret)

    # GeoTIFF
    gt, epsg = geo['geotransform'], geo['epsg_code']
    try:
        from rasterio.transform import Affine
        aff = Affine(gt[1], gt[2], gt[0], gt[4], gt[5], gt[3])
        with rasterio.open(os.path.join(psave, f'mf_{gas}.tif'), 'w', driver='GTiff',
                           height=ret.shape[0], width=ret.shape[1], count=1,
                           dtype='float32', crs=f'EPSG:{epsg}', transform=aff,
                           nodata=float('nan'), compress='deflate') as dst:
            dst.write(ret.astype('float32'), 1)
    except Exception as e:
        print(f'  GeoTIFF omitido: {e}')

    print(f'  MF listo en {time.time()-t0:.1f}s | std={np.nanstd(ret):.1f} ppm.m', flush=True)
    return ret, geo

# ------------------------------------------------------------------
# Helpers geometricos
# ------------------------------------------------------------------

def build_latlon_grid(geo):
    gt, epsg = geo['geotransform'], geo['epsg_code']
    rows_n, cols_n = geo['rows'], geo['cols']
    gx = gt[0] + (np.arange(cols_n) + 0.5) * gt[1]
    gy = gt[3] + (np.arange(rows_n) + 0.5) * gt[5]
    GX, GY = np.meshgrid(gx, gy)
    tr = pyproj.Transformer.from_crs(f'EPSG:{epsg}', 'EPSG:4326', always_xy=True)
    lon_g, lat_g = tr.transform(GX.ravel(), GY.ravel())
    return lat_g.reshape(GX.shape), lon_g.reshape(GX.shape), gt, epsg


def xy_to_lonlat(x, y, epsg):
    tr = pyproj.Transformer.from_crs(f'EPSG:{epsg}', 'EPSG:4326', always_xy=True)
    lon, lat = tr.transform(x, y)
    return float(lon), float(lat)


def mask_from_polygon(polygon, data, extent):
    ny, nx = data.shape
    x0, x1, y0, y1 = extent
    xv, yv = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y1, y0, ny))
    pts = np.column_stack((xv.ravel(), yv.ravel()))
    return matplotlib.path.Path(polygon).contains_points(pts).reshape(data.shape)


def convex_hull_L(mask, pix_res_m):
    r, c = np.where(mask)
    if len(r) < 3:
        return np.sqrt(max(len(r), 1)) * pix_res_m
    pts = np.column_stack((c, r)).astype(float) * pix_res_m
    return float(np.max(pdist(pts[ConvexHull(pts).vertices])))

# ------------------------------------------------------------------
# Cuantificacion de una pluma
# ------------------------------------------------------------------

def quantify_plume(data, mask, geo, gas, ts, lon_s, lat_s, label,
                   psave, site, wind_source, era5_file, wind_value, wind_from):
    pix_res = 30.0
    pix_area_cm2 = pix_res**2 * 1e4

    sum_xgas = float(np.nansum(np.where(mask, np.nan_to_num(data, nan=0), 0)))
    IME, _ = ppmm_to_kg(sum_xgas, pix_res, gas)
    L = convex_hull_L(mask, pix_res)
    N_pix = int(mask.sum())

    if wind_value is not None:
        u10 = wind_value
    else:
        u10 = wind_speed_manual(ts, lat_s, lon_s, psave, psave, label,
                                wind_source=wind_source, era5_file=era5_file,
                                wind_from=wind_from)

    if gas == 'nh3':
        ueff_1, a_1, ueff_2, a_2, err_u10, _, bool_thres = extract_ueff(u10, MISSION, gas)
        Q_1 = IME * 3600 * ueff_1 / L
        Q_2 = IME * 3600 * ueff_2 / L
        print(f'\n  [{label}] IME={IME:.2f} kg | L={L:.1f} m | N={N_pix}')
        print(f'  u10={u10:.2f} m/s | Q(tau=inf)={Q_1:.2f} kg/h | Q(tau=1h)={Q_2:.2f} kg/h')
        fields = ['Site', 'Mission', 'Timestamp', 'lat', 'lon',
                  'u10 (m/s)', 'err(u10)', 'Q_tau=inf (kg/h)', 'err(Q_inf)', 'Q_tau=1h (kg/h)', 'err(Q_1h)']
        # err simplificado
        conv = IME / max(sum_xgas, 1e-9)
        err_inf = np.sqrt((3600*IME*a_1*2/L)**2 + (3600*ueff_1*conv*np.sqrt(N_pix)*np.nanstd(data[mask])/L)**2) if bool_thres else 0
        info = [site, MISSION, ts, round(lat_s,4), round(lon_s,4),
                round(u10,2), round(err_u10,2), round(Q_1,2), round(err_inf,2), round(Q_2,2), 0]
        excel_info(fields, info, psave, gas)
        return dict(label=label, IME=IME, L=L, u10=u10, Q=Q_1, Q2=Q_2, N_pix=N_pix, lat=lat_s, lon=lon_s)
    else:
        ueff, a, err_u10, _, bool_thres = extract_ueff(u10, MISSION, gas)
        Q = IME * 3600 * ueff / L
        conv = IME / max(sum_xgas, 1e-9)
        err_Q = np.sqrt((3600*IME*a*2/L)**2 + (3600*ueff*conv*np.sqrt(N_pix)*np.nanstd(data[mask])/L)**2) if bool_thres else \
                np.sqrt((3600*IME*a*u10*0.5/L)**2 + (3600*ueff*conv*np.sqrt(N_pix)*np.nanstd(data[mask])/L)**2)
        print(f'\n  [{label}] IME={IME:.2f} kg | L={L:.1f} m | N={N_pix}')
        print(f'  u10={u10:.2f} m/s | Q({gas.upper()})={Q:.2f} ± {err_Q:.2f} kg/h')
        fields = ['Site', 'Mission', 'Timestamp', 'lat', 'lon', 'u10 (m/s)', 'err(u10)', 'Q (kg/h)', 'err(Q)']
        info = [site, MISSION, ts, round(lat_s,4), round(lon_s,4),
                round(u10,2), round(err_u10,2), round(Q,2), round(err_Q,2)]
        excel_info(fields, info, psave, gas)
        return dict(label=label, IME=IME, L=L, u10=u10, Q=Q, err_Q=err_Q, N_pix=N_pix, lat=lat_s, lon=lon_s)

# ------------------------------------------------------------------
# Modo automatico: plumas sembradas desde el GeoJSON oficial
# ------------------------------------------------------------------

def save_overview_png(ret, plumes, extent, epsg, gas, psave, vmax):
    """Una sola figura con el MF y todas las mascaras/fuentes etiquetadas."""
    try:
        fig, ax = plt.subplots(figsize=(11, 12), facecolor='black')
        ax.set_facecolor('black')
        im = ax.imshow(ret, extent=extent, cmap='inferno', vmin=0, vmax=vmax, origin='upper')
        for pl in plumes:
            ax.contour(pl['mask'], levels=[0.5], colors='cyan', linewidths=0.8,
                       extent=extent, origin='upper')
            x = extent[0] + (pl['col'] + 0.5) * (extent[1] - extent[0]) / ret.shape[1]
            y = extent[3] - (pl['row'] + 0.5) * (extent[3] - extent[2]) / ret.shape[0]
            ax.scatter(x, y, s=90, facecolor='none', edgecolor='white', linewidth=1.2, marker='o')
            ax.text(x, y, '  ' + pl['label'], color='yellow', fontsize=9)
        cb = plt.colorbar(im, ax=ax, shrink=0.75,
                          label=rf'$\Delta$X{gas.upper()} (ppm$\cdot$m)')
        cb.ax.yaxis.label.set_color('white'); cb.ax.tick_params(colors='white')
        ax.tick_params(colors='white')
        ax.set_xlabel(f'Easting (m, EPSG:{epsg})', color='white')
        ax.set_ylabel('Northing (m)', color='white')
        ax.set_title(f'{len(plumes)} plumas sembradas desde ql_ch4_json', color='white')
        plt.tight_layout()
        out = os.path.join(psave, f'plumes_auto_{gas}.png')
        fig.savefig(out, dpi=130, facecolor='black')
        plt.close(fig)
        print(f'\n[mapa] {out}')
    except Exception as e:
        print(f'\n[mapa] no se pudo generar el overview ({e}); los resultados no se ven afectados.')


def run_auto(ret, geo, gas, ts, psave, args):
    """Siembra desde el ql_ch4_json, crece cada mascara y cuantifica todas las plumas."""
    from plumes_from_json import find_plumes_json, read_plumes_json, seed_masks

    json_file = args.plumes_json or find_plumes_json(args.rad_file)
    if not json_file or not os.path.exists(json_file):
        sys.exit('--auto-json: no encontre el *_ql_ch4_json.geojson junto al .h5. '
                 'Pasalo con --plumes-json <archivo>.')
    print(f'[auto] plumas sembradas desde: {os.path.basename(json_file)}')

    plumes_json = read_plumes_json(json_file, gas)
    print(f'[auto] {len(plumes_json)} pluma(s) en el GeoJSON')

    qualities = set(q.strip() for q in args.quality.split(',')) if args.quality else None
    ladder = tuple([args.sigma] + [k for k in (1.5, 1.0, 0.75, 0.5) if k < args.sigma])

    gt, epsg = geo['geotransform'], geo['epsg_code']
    kept, skipped = seed_masks(ret, plumes_json, gt, epsg,
                               sigma_ladder=ladder, min_pix=args.min_pix,
                               qualities=qualities)

    for pl in skipped:
        print(f"  [-] {pl['label']:<3} ({pl['quality']:<12}) descartada: {pl['status']}")
    if not kept:
        print('\nNinguna pluma pudo delinearse.')
        return

    print(f'\n{"="*60}')
    print(f'Cuantificando {len(kept)} pluma(s) - gas: {gas.upper()}')
    print(f'{"="*60}')

    results = []
    for pl in kept:
        # --wind json: usamos el MISMO viento que reporta el producto oficial, para
        # comparar metodo contra metodo y no metodo+viento contra metodo+viento.
        if args.wind == 'json':
            wind_value, wind_from = pl['u10_official'], pl['wind_from_official']
            if wind_value is None:
                print(f"  [{pl['label']}] el GeoJSON no trae viento; uso --wind-value/geos")
                wind_value, wind_from = args.wind_value, args.wind_from
            wind_source = 'geos'
        else:
            wind_value, wind_from, wind_source = args.wind_value, args.wind_from, args.wind

        print(f"\n--- {pl['label']}  ({pl['quality']}, sigma={pl['sigma']:.2f}, {pl['n_pix']} px) ---")
        res = quantify_plume(ret, pl['mask'], geo, gas, ts, pl['lon'], pl['lat'],
                             pl['label'], psave, args.site, wind_source,
                             args.era5_file, wind_value, wind_from)
        res.update(quality=pl['quality'], sigma=pl['sigma'],
                   q_official=pl['q_official'], ime_official=pl['ime_official'])
        results.append(res)

    save_overview_png(ret, kept, [gt[0], gt[0] + geo['cols']*gt[1],
                                  gt[3] + geo['rows']*gt[5], gt[3]],
                      epsg, gas, psave, float(np.percentile(ret[np.isfinite(ret)], 99.5)))

    # --- Tabla resumen + comparacion contra el producto oficial ---
    print(f'\n{"="*94}')
    print(f'{"P":<4} {"qual":<12} {"lat":>9} {"lon":>10} {"N":>5} {"IME":>8} {"L(m)":>7} '
          f'{"u10":>5} {"Q(kg/h)":>9} {"Qofic":>9} {"diff":>7}')
    print('-' * 94)
    tot_q = tot_o = 0.0
    for r in results:
        qo = r.get('q_official')
        diff = f'{100*(r["Q"]-qo)/qo:+.0f}%' if qo else '   -'
        tot_q += r['Q']; tot_o += qo or 0.0
        print(f'{r["label"]:<4} {r["quality"]:<12} {r["lat"]:>9.5f} {r["lon"]:>10.5f} '
              f'{r["N_pix"]:>5} {r["IME"]:>8.1f} {r["L"]:>7.0f} {r["u10"]:>5.2f} '
              f'{r["Q"]:>9.1f} {(qo if qo else 0):>9.1f} {diff:>7}')
    print('=' * 94)
    if tot_o:
        print(f'Total Q = {tot_q:.1f} kg/h {gas.upper()}   |   total oficial = {tot_o:.1f} kg/h '
              f'({100*(tot_q-tot_o)/tot_o:+.0f}%)')
    else:
        print(f'Total Q = {tot_q:.1f} kg/h {gas.upper()}')


# ------------------------------------------------------------------
# Seleccion interactiva de una pluma
# ------------------------------------------------------------------

def select_one_plume(fig, ax, data, extent, epsg, plume_idx):
    ax.set_title(
        f'PLUMA {plume_idx}  |  '
        '1) Zoom (opcional) → Enter\n'
        '2) Click en la FUENTE\n'
        '3) Delinea la pluma → Enter para cerrar',
        color='white', fontsize=9, pad=6)
    plt.draw()

    print(f'\n[P{plume_idx}] Haz zoom si quieres y pulsa Enter...', flush=True)
    while True:
        if not plt.fignum_exists(fig.number):
            return None
        if plt.waitforbuttonpress(timeout=0.2):
            break

    if not plt.fignum_exists(fig.number):
        return None

    ax.set_title(f'PLUMA {plume_idx}  |  Click en la FUENTE', color='white', fontsize=10)
    plt.draw()
    print(f'[P{plume_idx}] Click en la fuente...', flush=True)
    source = plt.ginput(n=1, timeout=0, show_clicks=True)
    if not source or not plt.fignum_exists(fig.number):
        return None

    sx, sy = source[0]
    lon_s, lat_s = xy_to_lonlat(sx, sy, epsg)
    gm = f'https://www.google.com/maps?q={lat_s:.6f},{lon_s:.6f}&z=16'
    print(f'\n[P{plume_idx}] Fuente: lat={lat_s:.6f}  lon={lon_s:.6f}')
    print(f'  Google Maps -> {gm}\n', flush=True)

    ax.scatter(sx, sy, s=200, facecolor='white', edgecolor='red',
               linewidth=2.5, marker='*', zorder=5)
    plt.draw()

    ax.set_title(f'PLUMA {plume_idx}  |  Delinea → Enter para cerrar', color='white', fontsize=10)
    plt.draw()
    print(f'[P{plume_idx}] Dibuja el polígono (click izq=añadir, click der=deshacer, Enter=cerrar)...', flush=True)
    polygon = plt.ginput(n=-1, timeout=0, show_clicks=True, mouse_add=1, mouse_pop=3, mouse_stop=2)

    if len(polygon) < 3:
        print(f'[P{plume_idx}] Polígono insuficiente, descartado.', flush=True)
        return None

    xs_p = [pt[0] for pt in polygon] + [polygon[0][0]]
    ys_p = [pt[1] for pt in polygon] + [polygon[0][1]]
    ax.plot(xs_p, ys_p, 'y-', linewidth=1.5, zorder=4)
    ax.text(sx, sy + abs(extent[3]-extent[2])*0.005, f'P{plume_idx}',
            color='yellow', fontsize=9, zorder=6)
    plt.draw()

    mask = mask_from_polygon(polygon, data, extent)
    return mask, lon_s, lat_s

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Pipeline interactivo Tanager: MF + seleccion multi-pluma + IME.')
    ap.add_argument('rad_file',
                    help='Ruta al .h5 de Tanager (ortho_radiance_hdf5 o basic_radiance_hdf5).')
    ap.add_argument('--gas', default='ch4',
                    help='Gas a cuantificar: ch4, co2, c2h4, c2h2, nh3.')
    ap.add_argument('-o', '--output-dir', default=None,
                    help='Carpeta de salida. Por defecto: "output/" junto al .h5.')
    ap.add_argument('-s', '--site', default='site',
                    help='Nombre del sitio (etiqueta el CSV).')
    ap.add_argument('--wind', choices=['geos', 'era5', 'json'], default='geos',
                    help='Fuente de viento. "json" usa wind_speed_avg del ql_ch4_json (solo con --auto-json).')
    ap.add_argument('--era5-file', default=None,
                    help='Ruta al .nc ERA5 (u10/v10) si --wind era5.')
    ap.add_argument('--wind-value', type=float, default=None,
                    help='Velocidad de viento manual en m/s (anula geos/era5).')
    ap.add_argument('--wind-from', type=float, default=None,
                    help='Direccion del viento (grados met.) para el override manual.')
    ap.add_argument('--auto-json', action='store_true',
                    help='Modo automatico: siembra las plumas desde el asset ql_ch4_json y las cuantifica todas, sin clics.')
    ap.add_argument('--plumes-json', default=None,
                    help='Ruta al *_ql_ch4_json.geojson. Por defecto se busca junto al .h5.')
    ap.add_argument('--quality', default=None,
                    help='Filtra por plume_quality del GeoJSON, p.ej. "good" o "good,questionable". Por defecto: todas.')
    ap.add_argument('--sigma', type=float, default=2.0,
                    help='Umbral inicial (k*sigma) para crecer la mascara en modo automatico. Solo baja si la fuente no lo supera.')
    ap.add_argument('--min-pix', type=int, default=5,
                    help='Tamano minimo de mascara (px) para aceptar una pluma en modo automatico.')
    args = ap.parse_args()

    gas = args.gas.strip().lower()
    if gas not in GAS_LUT:
        sys.exit(f'Gas no soportado: {gas}. Opciones: {list(GAS_LUT)}')

    rad_file = os.path.abspath(args.rad_file)
    p = os.path.dirname(rad_file) + '/'
    n = os.path.splitext(os.path.basename(rad_file))[0]
    psave = args.output_dir if args.output_dir else os.path.join(p, 'output') + '/'
    os.makedirs(psave, exist_ok=True)

    # timestamp del nombre de archivo (YYYYMMDD_HHMMSS -> YYYYMMDDHHmmss)
    parts = n.split('_')
    ts = parts[0] + parts[1] if len(parts) >= 2 else n[:14]

    # --- MF ---
    ret, geo = run_or_load_mf(p, n, gas, psave)

    if args.auto_json:
        run_auto(ret, geo, gas, ts, psave, args)
        return

    if args.wind == 'json':
        sys.exit('--wind json solo tiene sentido con --auto-json (el viento viene del GeoJSON).')

    gt, epsg = geo['geotransform'], geo['epsg_code']
    extent = [gt[0], gt[0] + geo['cols']*gt[1],
              gt[3] + geo['rows']*gt[5], gt[3]]

    # --- Ventana ---
    valid = ret[np.isfinite(ret)]
    vmax = np.percentile(valid, 99.5)
    std  = float(np.nanstd(ret))

    fig, ax = plt.subplots(figsize=(10, 12), facecolor='black')
    ax.set_facecolor('black')
    im = ax.imshow(ret, extent=extent, cmap='inferno', vmin=0, vmax=vmax, origin='upper')
    cbar = plt.colorbar(im, ax=ax, shrink=0.75,
                        label=rf'$\Delta$X{gas.upper()} (ppm$\cdot$m)')
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(colors='white')
    ax.tick_params(colors='white')
    ax.set_xlabel(f'Easting (m, EPSG:{epsg})', color='white')
    ax.set_ylabel('Northing (m)', color='white')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)
    ax.set_autoscale_on(False)

    # --- Loop de plumas ---
    plumes = []
    idx = 1

    while plt.fignum_exists(fig.number):
        result = select_one_plume(fig, ax, ret, extent, epsg, idx)

        if result is None:
            if not plt.fignum_exists(fig.number):
                break
        else:
            mask, lon_s, lat_s = result
            plumes.append(dict(idx=idx, mask=mask, lon=lon_s, lat=lat_s))
            idx += 1

        if not plt.fignum_exists(fig.number):
            break

        print(f'¿Marcar otra pluma? [s/n]: ', end='', flush=True)
        if input().strip().lower() != 's':
            break

    if plt.fignum_exists(fig.number):
        plt.close(fig)

    if not plumes:
        print('\nNo se selecciono ninguna pluma.')
        return

    # --- Cuantificacion ---
    print(f'\n{"="*60}')
    print(f'Cuantificando {len(plumes)} pluma(s) — gas: {gas.upper()}')
    print(f'{"="*60}')

    results = []
    for pl in plumes:
        label = f'P{pl["idx"]}'
        res = quantify_plume(ret, pl['mask'], geo, gas, ts,
                             pl['lon'], pl['lat'], label,
                             psave, args.site, args.wind,
                             args.era5_file, args.wind_value, args.wind_from)
        results.append(res)

    # --- Tabla resumen ---
    print(f'\n{"="*65}')
    print(f'{"Pluma":<7} {"lat":>10} {"lon":>11} {"IME(kg)":>9} {"L(m)":>7} {"u10":>5} {"Q(kg/h)":>9}')
    print('-' * 65)
    for r in results:
        print(f'{r["label"]:<7} {r["lat"]:>10.5f} {r["lon"]:>11.5f} '
              f'{r["IME"]:>9.2f} {r["L"]:>7.1f} {r["u10"]:>5.2f} {r["Q"]:>9.2f}')
    print('=' * 65)
    print(f'Total Q = {sum(r["Q"] for r in results):.2f} kg/h  {gas.upper()}')


if __name__ == '__main__':
    main()
