#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cuantificacion asistida para escenas SIN plumas oficiales (sin ql_ch4_json),
corriendo LOS DOS matched filters.

Calcula el MF por columna (LARS) y el MF por grupos de columnas con fondo
PCA/k-means y shrinkage (HyGas, scripts/mf_advanced.py), propone candidatos a
partir de LOS DOS (union, para que ninguna pluma se pierda porque un solo
retrieval no la vio), los pinta numerados, y tu eliges con un clic cual es pluma.
Cada pluma aceptada se cuantifica con LOS DOS MF sobre la MISMA mascara y el
MISMO viento, asi que la diferencia en Q es del retrieval y de nada mas.

  1. Salen 4 paneles: anomalia wavelet (den) y realce fisico, para cada MF.
     Los candidatos van numerados; el color del numero dice quien lo detecto.
  2. Zoom si quieres -> Enter.
  3. Clic sobre cada candidato que consideres pluma -> Enter para terminar.
  4. Por cada uno: se muestra cuanto crecio la mascara y aceptas / ajustas sigma /
     rechazas.
  5. Cuantifica con los dos MF y guarda CSV + PNG.

VIENTO: aqui no hay GeoJSON, o sea que no hay viento oficial que heredar. Por
defecto se descarga ERA5 para la fecha de la escena y la posicion de CADA pluma
(--wind geos para GEOS-FP, --wind-value <m/s> para imponer uno propio). Nunca hay
un valor inventado por defecto: el viento se propaga lineal a Q.

Uso:
  python auto_pipeline/pick_plumes_wavelet.py tanager/20241121_183741_33_4001_ortho_radiance_hdf5.h5
  python auto_pipeline/pick_plumes_wavelet.py <.h5> --gas ch4 --sigma 2.5 --min-pix 15 --grow-on adv
"""

import os
import sys
import time
import argparse
import csv

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(os.path.dirname(_HERE), 'scripts')
for _p in (_SCRIPTS, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pyproj
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from skimage import measure

from Tanager_reader import read_Tanager, get_ortho_framing
from RT_functions import read_luts_libradtran, get_k_libradtran
from Retrieval_methods import mf_retrieval, masking_plumes
from quant_func_v2 import ppmm_to_kg, extract_ueff
from mf_advanced import run_advanced_mf
from wavelet_den import compute_den

MISSION = 'Tanager'
PIX_RES = 30.0

GAS_LUT = {
    'ch4':  ('LUT_ssd_100pm_wvl_2049_2500_nm_ch4.nc', 2050, 2495),
    'nh3':  ('LUT_ssd_100pm_wvl_1399_2500_nm_nh3.nc', 1400, 2495),
    'co2':  ('LUT_ssd_100pm_wvl_2049_2500_nm_ch4.nc', 2050, 2495),
    'c2h4': ('LUT_ssd_100pm_wvl_2049_2500_nm_ch4.nc', 2050, 2495),
    'c2h2': ('LUT_ssd_100pm_wvl_2049_2500_nm_ch4.nc', 2050, 2495),
}


# ------------------------------------------------------------------
# Los dos MF (con cache)
# ------------------------------------------------------------------

def run_or_load_both(p, n, gas, psave, adv_kwargs, force=False):
    """Devuelve (ret_lars, ret_adv, geo). Los dos comparten el mismo target k de
    la LUT, que es lo que hace que las dos salidas esten en ppm.m comparables."""
    import variable_definition
    scene = n.split('_ortho_')[0]
    c_l = os.path.join(psave, f'{scene}_mf_lars_{gas}.npy')
    c_a = os.path.join(psave, f'{scene}_mf_adv_{gas}.npy')
    geo = get_ortho_framing(p, n)

    if not force and os.path.exists(c_l) and os.path.exists(c_a):
        print(f'[MF] cache de los dos retrievals en {psave}')
        return np.load(c_l), np.load(c_a), geo

    print('[read] leyendo cubo...', flush=True)
    img_full, wvl_all, fwhm_all, sza, vza = read_Tanager(p, n)
    lut_file, wvl_inf, wvl_sup = GAS_LUT[gas]
    w = np.where((wvl_all >= wvl_inf) & (wvl_all <= wvl_sup))[0]
    img = img_full[:, :, w].astype(np.float32)
    wvl, fwhm = wvl_all[w], fwhm_all[w]
    del img_full
    print(f'  cubo {img.shape} | SZA={sza:.1f} VZA={vza:.1f} | {len(w)} bandas', flush=True)

    wvl_hr, rad_hr, dx = read_luts_libradtran(variable_definition.p_lut + lut_file, sza, vza, 3, 0)
    win = np.arange(len(w))
    _, k_arr = get_k_libradtran(img, wvl, fwhm, wvl_hr, rad_hr, dx, win, mission=MISSION)

    t0 = time.time()
    print('[MF-LARS] por columna...', flush=True)
    ret_l = mf_retrieval(img, k_arr, win, mission_name=MISSION, bool_libradtran=True)
    ret_l[~np.isfinite(img[:, :, 0])] = np.nan
    np.save(c_l, ret_l)
    print(f'  listo en {time.time()-t0:.0f}s | std={np.nanstd(ret_l):.1f} ppm.m', flush=True)

    k_vec = k_arr[0] if np.ndim(k_arr) == 2 else k_arr
    t0 = time.time()
    print('[MF-ADV] grupos de columnas + PCA/k-means + shrinkage...', flush=True)
    ret1 = run_advanced_mf(radiance_cube=img, targets=k_vec, wavelengths=wvl, mask=None, **adv_kwargs)
    exclude = (masking_plumes(np.nan_to_num(ret1), 10, np.nanstd(ret1)) * 1).astype(bool)
    ret2 = run_advanced_mf(radiance_cube=img, targets=k_vec, wavelengths=wvl, mask=exclude, **adv_kwargs)
    ret_a = np.where(np.isfinite(ret2), ret2, ret1)   # los px de pluma salen NaN en la 2a pasada
    ret_a[~np.isfinite(img[:, :, 0])] = np.nan
    np.save(c_a, ret_a)
    print(f'  listo en {time.time()-t0:.0f}s | std={np.nanstd(ret_a):.1f} ppm.m', flush=True)

    del img
    return ret_l, ret_a, geo


# ------------------------------------------------------------------
# Candidatos (union de los dos MF) y crecimiento
# ------------------------------------------------------------------

def detect(enh, sigma_mult, min_pix):
    """Cumulos por encima de sigma_mult*sigma con >= min_pix px. Mismo criterio
    que masking_plumes / Roger et al. 2025, pero devolviendo TODOS los candidatos."""
    filt = medfilt2d(np.nan_to_num(enh.astype(np.float32)), kernel_size=3)
    std = float(np.nanstd(enh))
    lab = measure.label((filt > sigma_mult * std) & (filt < 1e5), connectivity=1)
    out = []
    for reg in measure.regionprops(lab):
        if reg.area < min_pix:
            continue
        rr, cc = reg.coords[:, 0], reg.coords[:, 1]
        pk = int(np.argmax(filt[rr, cc]))
        out.append(dict(area=int(reg.area), row=int(rr[pk]), col=int(cc[pk]),
                        peak=float(filt[rr[pk], cc[pk]])))
    return out, std


def merge_candidates(cl, ca, tol=6):
    """Union de los candidatos de los dos MF. Dos picos a menos de `tol` px se
    consideran la misma pluma vista por los dos retrievals."""
    cands = []
    for c in cl:
        cands.append(dict(row=c['row'], col=c['col'], area_l=c['area'], area_a=0,
                          peak_l=c['peak'], peak_a=0.0, seen='L'))
    for c in ca:
        hit = None
        for d in cands:
            if np.hypot(d['col'] - c['col'], d['row'] - c['row']) <= tol:
                hit = d; break
        if hit is not None:
            hit['area_a'] = c['area']; hit['peak_a'] = c['peak']; hit['seen'] = 'LA'
        else:
            cands.append(dict(row=c['row'], col=c['col'], area_l=0, area_a=c['area'],
                              peak_l=0.0, peak_a=c['peak'], seen='A'))
    cands.sort(key=lambda d: -max(d['area_l'], d['area_a']))
    return cands


def grow_from(enh, row, col, sigma_mult):
    filt = medfilt2d(np.nan_to_num(enh.astype(np.float32)), kernel_size=3)
    std = float(np.nanstd(enh))
    lab = measure.label((filt > sigma_mult * std) & (filt < 1e5), connectivity=1)
    if lab[row, col] == 0:
        return None
    return lab == lab[row, col]


# ------------------------------------------------------------------
# Cuantificacion (nucleo LARS sin cambios)
# ------------------------------------------------------------------

def convex_hull_L(mask):
    r, c = np.where(mask)
    if len(r) < 3:
        return np.sqrt(max(len(r), 1)) * PIX_RES
    pts = np.column_stack((c, r)).astype(float) * PIX_RES
    return float(np.max(pdist(pts[ConvexHull(pts).vertices])))


def plume_L(mask, l_mode):
    """Escala de longitud. Por defecto sqrt(N*gsd^2), que es lo que usa extract_Q
    en el repo y ademas reproduce el `fetch` de Carbon Mapper con mediana 0.92
    sobre sus 20 plumas oficiales. El convex hull da 1.73 del fetch y se dispara
    en plumas alargadas, asi que no es comparable con el producto oficial."""
    if l_mode == 'hull':
        return convex_hull_L(mask)
    return float(np.sqrt(mask.sum()) * PIX_RES)


def quantify(enh, mask, u10, gas, ueff_mode='cm', l_mode='sqrt-area'):
    """IME + Q sobre una mascara. Ver compare_mf.quantify: 'cm' usa Ueff = u10,
    que es lo que hace Carbon Mapper (deducido exacto de las 20 plumas con GT);
    'lars' usa la parametrizacion a*u10+b del repo, que a estos vientos da
    1.8-2.4x menos."""
    sum_xgas = float(np.nansum(np.where(mask, np.nan_to_num(enh, nan=0.0), 0.0)))
    IME, _ = ppmm_to_kg(sum_xgas, PIX_RES, gas)
    L = plume_L(mask, l_mode)
    N = int(mask.sum())
    inside = enh[mask]; inside = inside[np.isfinite(inside)]
    noise = float(np.std(inside)) if inside.size else 0.0
    conv = IME / max(sum_xgas, 1e-9)

    if gas == 'nh3':
        ueff1, a1, ueff2, a2, err_u10, _, thres = extract_ueff(u10, MISSION, gas)
        wt = 2.0 if thres else u10 * 0.5
        if ueff_mode == 'cm':
            #Ueff = u10 no distingue tau: no hay Q(tau=1h) que reportar.
            ueff1, a1, ueff2 = float(u10), 1.0, None
        Q = IME * 3600 * ueff1 / L
        err = np.sqrt((3600*IME*a1*wt/L)**2 + (3600*ueff1*conv*np.sqrt(N)*noise/L)**2)
        Q2 = IME * 3600 * ueff2 / L if ueff2 else None
        return dict(IME=IME, L=L, N=N, Q=Q, Q2=Q2, err_Q=err, err_u10=err_u10, ueff=ueff1)

    ueff, a, err_u10, _, thres = extract_ueff(u10, MISSION, gas)
    wt = 2.0 if thres else u10 * 0.5
    if ueff_mode == 'cm':
        ueff, a = float(u10), 1.0     # dQ/du10 = Q/u10
    Q = IME * 3600 * ueff / L
    err = np.sqrt((3600*IME*a*wt/L)**2 + (3600*ueff*conv*np.sqrt(N)*noise/L)**2)
    return dict(IME=IME, L=L, N=N, Q=Q, Q2=None, err_Q=err, err_u10=err_u10, ueff=ueff)


# ------------------------------------------------------------------
# Interfaz
# ------------------------------------------------------------------

def pixel_to_lonlat(row, col, gt, epsg):
    x = gt[0] + (col + 0.5) * gt[1]
    y = gt[3] + (row + 0.5) * gt[5]
    tr = pyproj.Transformer.from_crs(f'EPSG:{epsg}', 'EPSG:4326', always_xy=True)
    lon, lat = tr.transform(x, y)
    return float(lon), float(lat)


SEEN_COLOR = {'L': 'yellow', 'A': 'deepskyblue', 'LA': 'lime'}


def show_and_pick(ret_l, ret_a, den_l, den_a, cands, gas, scene, gt, epsg):
    """2x2: den y realce fisico de cada MF, con los candidatos numerados."""
    panels = []
    for arr, kind, tag in ((den_l, 'den', 'LARS'), (den_a, 'den', 'ADV'),
                           (ret_l, 'enh', 'LARS'), (ret_a, 'enh', 'ADV')):
        if kind == 'den':
            disp = medfilt2d(arr.astype(np.float32), kernel_size=3)
            pos = arr[arr > 0]
            vm = float(np.nanpercentile(pos, 99)) if pos.size else 1.0
            lbl = 'den'
        else:
            disp = arr
            fin = arr[np.isfinite(arr)]
            vm = float(np.nanpercentile(fin, 99.5)) if fin.size else 1.0
            lbl = rf'$\Delta$X{gas.upper()} (ppm$\cdot$m)'
        panels.append((disp, vm, lbl, f'{tag} - {"anomalia wavelet" if kind=="den" else "realce fisico"}'))

    fig, axes = plt.subplots(2, 2, figsize=(19, 13), facecolor='black', sharex=True, sharey=True)
    for ax, (disp, vm, lbl, ttl) in zip(axes.ravel(), panels):
        ax.set_facecolor('black')
        im = ax.imshow(disp, cmap='inferno', vmin=0, vmax=vm)
        ax.set_title(ttl, color='white', fontsize=11)
        ax.axis('off')
        cb = plt.colorbar(im, ax=ax, shrink=0.75, label=lbl)
        cb.ax.yaxis.label.set_color('white'); cb.ax.tick_params(colors='white')
        for i, cd in enumerate(cands, 1):
            col = SEEN_COLOR[cd['seen']]
            ax.scatter(cd['col'], cd['row'], s=110, facecolor='none',
                       edgecolor=col, linewidth=1.3, zorder=5)
            ax.text(cd['col'] + 6, cd['row'] - 6, str(i), color=col,
                    fontsize=10, fontweight='bold', zorder=6)

    fig.suptitle(f'{scene}  |  {len(cands)} candidato(s)   '
                 'amarillo=solo LARS   azul=solo ADV   verde=los dos\n'
                 'Zoom si quieres -> Enter -> clic sobre cada pluma real -> Enter para terminar',
                 color='white', fontsize=12)
    plt.tight_layout()
    plt.show(block=False); plt.pause(0.5)
    for ax in axes.ravel():
        ax.set_autoscale_on(False)

    #Cada candidato con sus coordenadas y su link, para poder mirarlo en Maps ANTES
    #de decidir si es pluma o un falso positivo (una balsa, un tejado, una sombra).
    print(f'\n{len(cands)} candidatos (por tamano):')
    for i, cd in enumerate(cands, 1):
        lon, lat = pixel_to_lonlat(cd['row'], cd['col'], gt, epsg)
        cd['lat'], cd['lon'] = lat, lon
        print(f'  {i:>3} [{cd["seen"]:<2}] {cd["area_l"]:>5} px LARS / {cd["area_a"]:>5} px ADV | '
              f'pico {cd["peak_l"]:>8.1f} / {cd["peak_a"]:>8.1f} ppm.m | fila {cd["row"]}, col {cd["col"]}')
        print(f'        {lat:.6f}, {lon:.6f}  ->  https://www.google.com/maps?q={lat:.6f},{lon:.6f}&z=17')

    print('\nHaz zoom si quieres y pulsa Enter en la figura...', flush=True)
    while True:
        if not plt.fignum_exists(fig.number):
            return [], None
        if plt.waitforbuttonpress(timeout=0.2):
            break

    print('Clic sobre cada pluma que quieras cuantificar. Enter para terminar.', flush=True)
    clicks = plt.ginput(n=-1, timeout=0, show_clicks=True)
    return clicks, fig


def nearest_candidate(cands, x, y, max_dist=40):
    best, bd = None, np.inf
    for i, cd in enumerate(cands):
        d = np.hypot(cd['col'] - x, cd['row'] - y)
        if d < bd:
            bd, best = d, i
    return None if (best is None or bd > max_dist) else best


# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Corre los dos MF, propone candidatos de la union de ambos, tu eliges con un clic y cuantifica cada pluma con los dos retrievals.')
    ap.add_argument('rad_file', help='.h5 ortho_radiance de Tanager.')
    ap.add_argument('--gas', default='ch4', choices=list(GAS_LUT))
    ap.add_argument('-o', '--output-dir', default=None)
    ap.add_argument('-s', '--site', default='site')
    ap.add_argument('--sigma', type=float, default=2.0, help='Umbral k*sigma para proponer candidatos y crecer la mascara.')
    ap.add_argument('--min-pix', type=int, default=10, help='Tamano minimo de un candidato (px).')
    ap.add_argument('--ueff-mode', choices=['cm', 'lars'], default='cm',
                    help="Velocidad efectiva. 'cm' = Ueff=u10, la de Carbon Mapper (deducida exacta de sus 20 plumas con GT); es la que hace los numeros comparables con el producto oficial. 'lars' = a*u10+b, la parametrizacion original del repo.")
    ap.add_argument('--l-mode', choices=['sqrt-area', 'hull'], default='sqrt-area',
                    help="Escala de longitud. 'sqrt-area' = sqrt(N*gsd^2), reproduce el fetch oficial con mediana 0.92. 'hull' = diametro del convex hull (mediana 1.73, no comparable).")
    ap.add_argument('--grow-on', choices=['lars', 'adv'], default='lars',
                    help='Sobre que mapa se crece la mascara. La MISMA mascara se usa para cuantificar los dos MF, para que la diferencia sea solo del retrieval.')
    ap.add_argument('--wind', choices=['geos', 'era5'], default='era5',
                    help='Fuente del viento, descargado para la fecha y la posicion de CADA pluma. Por defecto ERA5.')
    ap.add_argument('--wind-value', type=float, default=None,
                    help='Viento u10 (m/s) propio, anula --wind. Sin GeoJSON no hay viento oficial que heredar.')
    ap.add_argument('--era5-file', default=None)
    ap.add_argument('--force', action='store_true', help='Recalcula los MF aunque exista cache.')
    ap.add_argument('--group-min', type=int, default=10)
    ap.add_argument('--group-max', type=int, default=30)
    ap.add_argument('--clusters', type=int, default=3)
    ap.add_argument('--shrinkage', type=float, default=0.1)
    ap.add_argument('--per-cluster-targets', action='store_true')
    ap.add_argument('--adaptive-shrinkage', action='store_true')
    args = ap.parse_args()

    gas = args.gas
    rad = os.path.abspath(args.rad_file)
    p = os.path.dirname(rad) + os.sep
    n = os.path.splitext(os.path.basename(rad))[0]
    scene = n.split('_ortho_')[0]
    psave = args.output_dir or os.path.join(p, 'out_pick')
    os.makedirs(psave, exist_ok=True)

    adv_kwargs = dict(group_min=args.group_min, group_max=args.group_max,
                      n_clusters=args.clusters, shrinkage=args.shrinkage,
                      per_cluster_targets=args.per_cluster_targets,
                      adaptive_shrinkage=args.adaptive_shrinkage)

    ret_l, ret_a, geo = run_or_load_both(p, n, gas, psave, adv_kwargs, force=args.force)
    gt, epsg = geo['geotransform'], geo['epsg_code']

    print('[wavelet] anomalia den de los dos mapas...', flush=True)
    den_l, den_a = compute_den(ret_l), compute_den(ret_a)

    cl, std_l = detect(ret_l, args.sigma, args.min_pix)
    ca, std_a = detect(ret_a, args.sigma, args.min_pix)
    cands = merge_candidates(cl, ca)
    print(f'[det] sigma_LARS={std_l:.1f} ppm.m -> {len(cl)} candidato(s) | '
          f'sigma_ADV={std_a:.1f} ppm.m -> {len(ca)} candidato(s) | union: {len(cands)}')
    if not cands:
        print('Ningun candidato. Baja --sigma o --min-pix.')
        return

    clicks, fig = show_and_pick(ret_l, ret_a, den_l, den_a, cands, gas, scene, gt, epsg)
    if not clicks:
        print('No elegiste ninguna pluma.')
        if fig is not None and plt.fignum_exists(fig.number):
            plt.close(fig)
        return

    grow_map = ret_l if args.grow_on == 'lars' else ret_a
    chosen = []
    for x, y in clicks:
        idx = nearest_candidate(cands, x, y)
        if idx is None:
            print(f'  clic en ({x:.0f},{y:.0f}) no cae cerca de ningun candidato; ignorado.')
            continue
        if any(c['idx'] == idx for c in chosen):
            continue
        cd = cands[idx]
        k = float(args.sigma)
        while True:
            mask = grow_from(grow_map, cd['row'], cd['col'], k)
            if mask is None:
                ans = input(f'  candidato {idx+1}: el pico no supera {k:.2f}sigma en el mapa {args.grow_on.upper()}. '
                            f'Nuevo sigma o "r" para descartar: ').strip().lower()
                if ans == 'r':
                    mask = None; break
                try: k = float(ans); continue
                except ValueError: print('    no es un numero.'); continue
            ans = input(f'  candidato {idx+1}: {int(mask.sum())} px a {k:.2f}sigma. '
                        f'y = aceptar / a = ajustar sigma / r = rechazar: ').strip().lower()
            if ans == 'y':
                break
            if ans == 'a':
                try: k = float(input('    nuevo sigma: '))
                except ValueError: print('    no es un numero.')
                continue
            if ans == 'r':
                mask = None; break
            print('    respuesta no valida.')
        if mask is not None:
            chosen.append(dict(idx=idx, cand=cd, mask=mask, sigma=k))

    if fig is not None and plt.fignum_exists(fig.number):
        plt.close(fig)
    if not chosen:
        print('\nNinguna pluma aceptada.')
        return

    # --- viento: el MISMO para los dos MF, pero uno por pluma en SU posicion ---
    parts = scene.split('_')
    ts = parts[0] + parts[1] if len(parts) >= 2 else scene[:14]

    def wind_for(lat, lon, tag):
        if args.wind_value is not None:
            return args.wind_value
        from wind_v2 import wind_speed_manual
        return float(wind_speed_manual(ts, lat, lon, psave, psave, f'{scene}_{tag}',
                                       wind_source=args.wind, era5_file=args.era5_file,
                                       wind_from=None))

    if args.wind_value is not None:
        print(f'\n[wind] u10 dado a mano = {args.wind_value:.2f} m/s para todas las plumas')
    else:
        print(f'\n[wind] {args.wind.upper()} para la escena {ts}, un valor por pluma en su propia posicion')

    # --- cuantificacion con LOS DOS ---
    print(f'\n{"="*110}')
    print(f'{"#":<4}{"visto":<6}{"lat":>10}{"lon":>11}{"sig":>6}{"N":>6}{"u10":>6}'
          f'{"IME_L":>9}{"IME_A":>9}{"Q_LARS":>11}{"Q_ADV":>11}{"A/L":>7}')
    print('-' * 110)
    results = []
    for j, ch in enumerate(chosen, 1):
        lon, lat = pixel_to_lonlat(ch['cand']['row'], ch['cand']['col'], gt, epsg)
        u10 = wind_for(lat, lon, f'p{j}')
        rl = quantify(ret_l, ch['mask'], u10, gas, args.ueff_mode, args.l_mode)
        ra = quantify(ret_a, ch['mask'], u10, gas, args.ueff_mode, args.l_mode)
        ratio = ra['Q'] / rl['Q'] if rl['Q'] else float('nan')
        results.append(dict(scene=scene, site=args.site, gas=gas, n_plume=j,
                            seen=ch['cand']['seen'], lat=lat, lon=lon,
                            sigma=ch['sigma'], grow_on=args.grow_on, N=rl['N'], L=rl['L'], u10=u10,
                            ueff=rl['ueff'], ueff_mode=args.ueff_mode, l_mode=args.l_mode,
                            IME_lars=rl['IME'], Q_lars=rl['Q'], err_lars=rl['err_Q'],
                            IME_adv=ra['IME'], Q_adv=ra['Q'], err_adv=ra['err_Q'],
                            ratio_adv_lars=ratio))
        print(f'{j:<4}{ch["cand"]["seen"]:<6}{lat:>10.5f}{lon:>11.5f}{ch["sigma"]:>6.2f}{rl["N"]:>6}{u10:>6.2f}'
              f'{rl["IME"]:>9.1f}{ra["IME"]:>9.1f}{rl["Q"]:>11.1f}{ra["Q"]:>11.1f}{ratio:>7.2f}')
        print(f'      +/- {rl["err_Q"]:.1f} (LARS)   +/- {ra["err_Q"]:.1f} (ADV)   '
              f'https://www.google.com/maps?q={lat:.6f},{lon:.6f}&z=16')
    print('=' * 110)
    tl = sum(r['Q_lars'] for r in results); ta = sum(r['Q_adv'] for r in results)
    print(f'TOTAL  Q_LARS={tl:.1f}   Q_ADV={ta:.1f} kg/h {gas.upper()}   ADV/LARS={ta/tl:.2f}' if tl
          else f'TOTAL  Q_LARS={tl:.1f}  Q_ADV={ta:.1f}')
    print(f'mascara crecida sobre {args.grow_on.upper()}, identica para los dos: la diferencia es solo del retrieval.')

    out_csv = os.path.join(psave, f'{scene}_pick_{gas}.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        wcsv = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        wcsv.writeheader(); wcsv.writerows(results)
    print(f'[csv] {out_csv}')

    try:
        fig2, axes = plt.subplots(1, 2, figsize=(19, 10), facecolor='black', sharex=True, sharey=True)
        for ax, arr, ttl in zip(axes, (ret_l, ret_a), ('MF LARS (por columna)', 'MF ADV (grupos+PCA/kmeans)')):
            ax.set_facecolor('black')
            fin = arr[np.isfinite(arr)]
            im = ax.imshow(arr, cmap='inferno', vmin=0, vmax=float(np.nanpercentile(fin, 99.5)))
            for j, ch in enumerate(chosen, 1):
                ax.contour(ch['mask'], levels=[0.5], colors='cyan', linewidths=0.9)
                ax.text(ch['cand']['col'] + 6, ch['cand']['row'] - 6, str(j),
                        color='yellow', fontsize=11, fontweight='bold')
            ax.set_title(f'{ttl}\nstd={np.nanstd(arr):.1f} ppm.m', color='white')
            ax.axis('off')
            cb = plt.colorbar(im, ax=ax, shrink=0.7, label=rf'$\Delta$X{gas.upper()} (ppm$\cdot$m)')
            cb.ax.yaxis.label.set_color('white'); cb.ax.tick_params(colors='white')
        fig2.suptitle(f'{scene} - {len(chosen)} pluma(s) aceptada(s)', color='white', fontsize=13)
        plt.tight_layout()
        out_png = os.path.join(psave, f'{scene}_pick_{gas}.png')
        fig2.savefig(out_png, dpi=130, facecolor='black')
        plt.close(fig2)
        print(f'[mapa] {out_png}')
    except Exception as e:
        print(f'[mapa] omitido ({e})')


if __name__ == '__main__':
    main()
