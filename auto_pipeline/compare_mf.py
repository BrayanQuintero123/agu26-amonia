#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparacion de matched filters sobre Tanager: MF por columna (LARS, el del repo)
vs MF por GRUPOS de columnas con fondo PCA/k-means y covarianza con shrinkage
(estilo HyGas, scripts/mf_advanced.py).

Por que asi: las dos implementaciones comparten el mismo target `t = mu * k`
(k sale de la LUT de libradtran), o sea que las dos salidas estan en ppm.m y son
directamente comparables. Lo unico que cambia es como se estima el FONDO:

  LARS      -> una covarianza por columna, pinv, sin regularizar.
  Avanzado  -> columnas agrupadas (10-30) segun el perfil de striping, dentro de
               cada grupo PCA(3) + k-means, una covarianza por cluster con
               shrinkage. Mas muestras por covarianza y fondo mas homogeneo.

Para que la comparacion aisle el RETRIEVAL y no la delineacion, los dos se
cuantifican sobre la MISMA mascara, que ademas no viene de ninguno de los dos:
es la huella del quicklook oficial (ortho_ql_ch4), recortada a la componente
conexa que contiene la fuente del GeoJSON. El viento tambien es el mismo (el que
reporta el GeoJSON), asi que cualquier diferencia en IME/Q es del MF.

Cuando varias plumas del GeoJSON caen en la misma huella conexa, se cuantifica
la huella UNA vez y se compara contra la SUMA de sus emisiones oficiales.

Uso:
  python auto_pipeline/compare_mf.py tanager/20241004_081921_28_4001_ortho_radiance_hdf5.h5
  python auto_pipeline/compare_mf.py tanager/*_ortho_radiance_hdf5.h5 --gas ch4
"""

import os
import sys
import glob
import time
import argparse
import csv

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(os.path.dirname(_HERE), 'scripts')
for _p in (_SCRIPTS, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import rasterio
import pyproj
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from skimage import measure

from Tanager_reader import read_Tanager, get_ortho_framing
from RT_functions import read_luts_libradtran, get_k_libradtran
from Retrieval_methods import mf_retrieval, masking_plumes, AT_MF
from quant_func_v2 import ppmm_to_kg, extract_ueff
from mf_advanced import run_advanced_mf
from plumes_from_json import read_plumes_json, find_plumes_json

MISSION = 'Tanager'
PIX_RES = 30.0

GAS_LUT = {
    'ch4': ('LUT_ssd_100pm_wvl_2049_2500_nm_ch4.nc', 2050, 2495),
    'nh3': ('LUT_ssd_100pm_wvl_1399_2500_nm_nh3.nc', 1400, 2495),
}


# ------------------------------------------------------------------
# Retrievals
# ------------------------------------------------------------------

def compute_k(img, wvl, fwhm, sza, vza, gas):
    """Target de absorcion unitaria k(lambda) desde la LUT de libradtran.
    Es el MISMO k para los dos MF: eso es lo que hace comparables las salidas."""
    import variable_definition
    lut_file, _, _ = GAS_LUT[gas]
    fn_lut = variable_definition.p_lut + lut_file
    wvl_hr, rad_hr_arr, delta_x_arr = read_luts_libradtran(fn_lut, sza, vza, 3, 0)
    win_idx = np.arange(img.shape[2])
    _, k_arr = get_k_libradtran(img, wvl, fwhm, wvl_hr, rad_hr_arr,
                                delta_x_arr, win_idx, mission=MISSION)
    return np.asarray(k_arr, dtype=np.float64)


def mf_lars(img, k_arr):
    """MF del repo, sin tocar: AT_MF por columna + segunda pasada enmascarando plumas."""
    win_idx = np.arange(img.shape[2])
    return mf_retrieval(img, k_arr, win_idx, mission_name=MISSION, bool_libradtran=True)


def mf_advanced_2pass(img, k_arr, wvl, **kwargs):
    """Igual que mf_retrieval pero con el MF avanzado: primera pasada, se detectan
    las plumas, y segunda pasada excluyendolas de la estadistica de fondo. Sin esto
    la comparacion seria injusta (LARS si corrige la hipotesis de sparsity)."""
    ret1 = run_advanced_mf(radiance_cube=img, targets=k_arr, wavelengths=wvl,
                           mask=None, **kwargs)
    plume = masking_plumes(np.nan_to_num(ret1), 10, np.nanstd(ret1)) * 1
    exclude = plume.astype(bool)                     # True = fuera del fondo
    ret2 = run_advanced_mf(radiance_cube=img, targets=k_arr, wavelengths=wvl,
                           mask=exclude, **kwargs)
    # Los pixeles enmascarados salen NaN en la 2a pasada: se rellenan con la 1a,
    # que es justo donde estan las plumas y lo que queremos cuantificar.
    out = np.where(np.isfinite(ret2), ret2, ret1)
    return out


# ------------------------------------------------------------------
# Mascara comun desde el quicklook oficial
# ------------------------------------------------------------------

def align_quicklook(ql_tif, gt, epsg, shape):
    """Remuestrea (vecino mas cercano) la huella del quicklook oficial a la malla del MF."""
    with rasterio.open(ql_tif) as ds:
        ql = ds.read(1)
        T = ds.transform
        ql_epsg = ds.crs.to_epsg()

    rows, cols = shape
    x = gt[0] + (np.arange(cols) + 0.5) * gt[1]
    y = gt[3] + (np.arange(rows) + 0.5) * gt[5]
    X, Y = np.meshgrid(x, y)

    if ql_epsg != epsg:
        tr = pyproj.Transformer.from_crs(f'EPSG:{epsg}', f'EPSG:{ql_epsg}', always_xy=True)
        X, Y = tr.transform(X, Y)

    inv = ~T
    c, r = inv * (X, Y)
    c = np.floor(c).astype(int)
    r = np.floor(r).astype(int)
    ok = (r >= 0) & (r < ql.shape[0]) & (c >= 0) & (c < ql.shape[1])

    out = np.zeros(shape, dtype=ql.dtype)
    out[ok] = ql[r[ok], c[ok]]
    return out > 0


def footprint_components(foot, min_pix=10):
    lab = measure.label(foot, connectivity=2)
    sizes = np.bincount(lab.ravel())
    for cid in range(1, len(sizes)):
        if sizes[cid] < min_pix:
            lab[lab == cid] = 0
    return lab


def source_pixel(lon, lat, gt, epsg, shape):
    tr = pyproj.Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True)
    x, y = tr.transform(lon, lat)
    col = int(np.floor((x - gt[0]) / gt[1]))
    row = int(np.floor((y - gt[3]) / gt[5]))
    if not (0 <= row < shape[0] and 0 <= col < shape[1]):
        return None
    return row, col


def component_for_source(lab, row, col, radius=4):
    """Componente de la huella oficial que contiene la fuente; si la fuente cae
    justo fuera del borde, se busca la componente mas cercana en +/-radius."""
    if lab[row, col] > 0:
        return int(lab[row, col])
    r0, r1 = max(row - radius, 0), min(row + radius + 1, lab.shape[0])
    c0, c1 = max(col - radius, 0), min(col + radius + 1, lab.shape[1])
    win = lab[r0:r1, c0:c1]
    vals = win[win > 0]
    if vals.size == 0:
        return 0
    return int(np.bincount(vals).argmax())


# ------------------------------------------------------------------
# Cuantificacion IME (nucleo de LARS, sin cambios)
# ------------------------------------------------------------------

def convex_hull_L(mask, pix_res=PIX_RES):
    r, c = np.where(mask)
    if len(r) < 3:
        return np.sqrt(max(len(r), 1)) * pix_res
    pts = np.column_stack((c, r)).astype(float) * pix_res
    return float(np.max(pdist(pts[ConvexHull(pts).vertices])))


def plume_L(mask, l_mode, fetch=None):
    """Escala de longitud de la pluma.

      sqrt-area : sqrt(N*gsd^2). Es lo que usa extract_Q en el repo, y contrastado
                  contra el `fetch` de las 20 plumas oficiales da mediana 0.92 --
                  o sea, reproduce el fetch de Carbon Mapper bastante bien.
      fetch     : el fetch del GeoJSON tal cual (solo si la escena lo trae).
      hull      : diametro del convex hull. Contra el fetch oficial da mediana 1.73
                  y se dispara en plumas alargadas (hasta 5x), asi que no es
                  comparable con el producto oficial.
    """
    if l_mode == 'fetch' and fetch:
        return float(fetch)
    if l_mode == 'hull':
        return convex_hull_L(mask)
    return float(np.sqrt(mask.sum()) * PIX_RES)


def quantify(enh, mask, u10, gas, ueff_mode='cm', l_mode='sqrt-area', fetch=None):
    """IME + Q sobre una mascara dada. La mascara y el viento son identicos para
    los dos MF, asi que Q solo depende del retrieval.

    ueff_mode:
      'cm'   -> Ueff = u10. Es lo que hace Carbon Mapper: despejando Q=IME*3600*Ueff/L
                en las 20 plumas del GeoJSON sale Ueff = wind_speed_avg exacto, con
                L = fetch. Usar esto es lo unico que permite comparar Q contra Q.
      'lars' -> Ueff = a*u10 + b (Guanter et al. 2021 / Roger et al. 2024), que es
                la parametrizacion original del repo. Da 1.8-2.4x menos a estos
                vientos, y ese factor explica casi todo el hueco contra el oficial.
    """
    vals = np.where(mask, np.nan_to_num(enh, nan=0.0), 0.0)
    sum_xgas = float(np.nansum(vals))
    IME, _ = ppmm_to_kg(sum_xgas, PIX_RES, gas)
    L = plume_L(mask, l_mode, fetch)
    N = int(mask.sum())

    ueff_lars, a, err_u10, _, bool_thres = extract_ueff(u10, MISSION, gas)
    ueff = float(u10) if ueff_mode == 'cm' else ueff_lars

    Q = IME * 3600 * ueff / L
    conv = IME / max(sum_xgas, 1e-9)
    inside = enh[mask]
    inside = inside[np.isfinite(inside)]
    noise = float(np.std(inside)) if inside.size else 0.0
    wind_term = 2.0 if bool_thres else u10 * 0.5
    #En modo 'cm' Ueff = u10, o sea dQ/du10 = Q/u10: el termino de viento es directo.
    slope = 1.0 if ueff_mode == 'cm' else a
    err_Q = np.sqrt((3600 * IME * slope * wind_term / L) ** 2 +
                    (3600 * ueff * conv * np.sqrt(N) * noise / L) ** 2)
    return dict(IME=IME, L=L, N=N, Q=Q, err_Q=err_Q, u10=u10, ueff=ueff,
                mean_enh=float(np.nanmean(inside)) if inside.size else 0.0)


# ------------------------------------------------------------------
# Una escena
# ------------------------------------------------------------------

def process_scene(rad_file, gas, psave, adv_kwargs, force=False, ueff_mode='cm', l_mode='sqrt-area', per_plume=False):
    p = os.path.dirname(os.path.abspath(rad_file)) + os.sep
    n = os.path.splitext(os.path.basename(rad_file))[0]
    scene = n.split('_ortho_')[0]
    print(f'\n{"#"*96}\n# ESCENA {scene}\n{"#"*96}', flush=True)

    geo = get_ortho_framing(p, n)
    gt, epsg = geo['geotransform'], geo['epsg_code']
    shape = (geo['rows'], geo['cols'])

    c_lars = os.path.join(psave, f'{scene}_mf_lars_{gas}.npy')
    c_adv = os.path.join(psave, f'{scene}_mf_adv_{gas}.npy')

    if not force and os.path.exists(c_lars) and os.path.exists(c_adv):
        print('[MF] cargando cache de los dos retrievals')
        ret_lars, ret_adv = np.load(c_lars), np.load(c_adv)
    else:
        print('[read] leyendo cubo...', flush=True)
        img_full, wvl_all, fwhm_all, sza, vza = read_Tanager(p, n)
        _, wvl_inf, wvl_sup = GAS_LUT[gas]
        w = np.where((wvl_all >= wvl_inf) & (wvl_all <= wvl_sup))[0]
        img = img_full[:, :, w].astype(np.float32)   # float32: el MF avanzado ya hace su propia copia en float64
        wvl, fwhm = wvl_all[w], fwhm_all[w]
        del img_full
        print(f'  cubo {img.shape} | SZA={sza:.1f} VZA={vza:.1f} | {len(w)} bandas', flush=True)

        print('[k] target desde LUT libradtran...', flush=True)
        k_arr = compute_k(img, wvl, fwhm, sza, vza, gas)

        t0 = time.time()
        print('[MF-LARS] por columna...', flush=True)
        ret_lars = mf_lars(img, k_arr)
        ret_lars[~np.isfinite(img[:, :, 0])] = np.nan
        print(f'  listo en {time.time()-t0:.0f}s | std={np.nanstd(ret_lars):.1f} ppm.m', flush=True)
        np.save(c_lars, ret_lars)

        # El MF avanzado espera k como vector 1-D de bandas (target unitario).
        k_vec = k_arr[0] if np.ndim(k_arr) == 2 else k_arr
        t0 = time.time()
        print('[MF-ADV] grupos de columnas + PCA/k-means + shrinkage...', flush=True)
        ret_adv = mf_advanced_2pass(img, k_vec, wvl, **adv_kwargs)
        ret_adv[~np.isfinite(img[:, :, 0])] = np.nan
        print(f'  listo en {time.time()-t0:.0f}s | std={np.nanstd(ret_adv):.1f} ppm.m', flush=True)
        np.save(c_adv, ret_adv)
        del img

    # --- plumas oficiales + huella comun ---
    js = find_plumes_json(rad_file)
    plumes = read_plumes_json(js, gas) if js else []
    ql = os.path.join(p, f'{scene}_ortho_ql_ch4.tif')
    foot = align_quicklook(ql, gt, epsg, shape)
    lab = footprint_components(foot)
    print(f'[huella] quicklook oficial: {int(foot.sum())} px, '
          f'{lab.max()} componente(s) | GeoJSON: {len(plumes)} pluma(s)')

    # agrupar plumas del GeoJSON por componente conexa de la huella
    groups = {}
    orphans = []
    for pl in plumes:
        rc = source_pixel(pl['lon'], pl['lat'], gt, epsg, shape)
        if rc is None:
            orphans.append((pl, 'fuera de la escena')); continue
        cid = component_for_source(lab, *rc)
        if cid == 0:
            orphans.append((pl, 'sin huella oficial en el quicklook')); continue
        groups.setdefault(cid, []).append(pl)

    for pl, why in orphans:
        print(f"  [-] {pl['label']} descartada: {why}")

    rows_out = []
    for cid, members in sorted(groups.items()):
        comp = (lab == cid)
        #Varias plumas del GeoJSON pueden caer en la MISMA huella conexa. Con
        #--per-plume la huella se reparte por fuente mas cercana (Voronoi sobre
        #los pixeles de la componente), asi cada pluma oficial recibe su propia
        #masa y se puede comparar una a una. Sin la flag se cuantifica la huella
        #entera una vez contra la SUMA de las emisiones oficiales.
        if per_plume and len(members) > 1:
            rr, cc = np.where(comp)
            srcs = [source_pixel(m['lon'], m['lat'], gt, epsg, shape) for m in members]
            d2 = np.stack([(rr - s[0]) ** 2 + (cc - s[1]) ** 2 if s else
                           np.full(rr.shape, np.inf) for s in srcs])
            owner = np.argmin(d2, axis=0)
            parts = []
            for i, m in enumerate(members):
                sub = np.zeros_like(comp)
                sub[rr[owner == i], cc[owner == i]] = True
                parts.append(([m], sub, True))
        else:
            parts = [(members, comp, False)]

        for mem, mask, was_split in parts:
            if not mask.any():
                continue
            labels = '+'.join(m['label'] for m in mem)
            u10 = float(np.mean([m['u10_official'] for m in mem]))
            wdir = float(np.mean([m['wind_from_official'] for m in mem
                                  if m['wind_from_official'] is not None] or [np.nan]))
            q_off = sum(m['q_official'] for m in mem)
            ime_off = sum(m['ime_official'] for m in mem)
            qual = ','.join(sorted({m['quality'] for m in mem}))
            #Cuando varias plumas comparten huella, la L que las cubre a todas es
            #la mayor de sus fetch.
            fetch = max([m['fetch_official'] for m in mem if m['fetch_official']] or [0]) or None

            r_l = quantify(ret_lars, mask, u10, gas, ueff_mode, l_mode, fetch)
            r_a = quantify(ret_adv, mask, u10, gas, ueff_mode, l_mode, fetch)

            rows_out.append(dict(scene=scene, plumes=labels, quality=qual,
                                 plume_id=';'.join(m['plume_id'] for m in mem),
                                 datetime=mem[0]['datetime'],
                                 lat=round(float(np.mean([m['lat'] for m in mem])), 6),
                                 lon=round(float(np.mean([m['lon'] for m in mem])), 6),
                                 shared_footprint=was_split, n_pix=int(mask.sum()),
                                 u10=u10, wind_dir=wdir,
                                 wind_source=mem[0]['wind_source_official'],
                                 ueff=r_l['ueff'], ueff_mode=ueff_mode,
                                 L=r_l['L'], l_mode=l_mode, fetch_off=fetch,
                                 ime_off=ime_off, q_off=q_off,
                                 ime_lars=r_l['IME'], q_lars=r_l['Q'], err_lars=r_l['err_Q'],
                                 mean_lars=r_l['mean_enh'],
                                 ime_adv=r_a['IME'], q_adv=r_a['Q'], err_adv=r_a['err_Q'],
                                 mean_adv=r_a['mean_enh'],
                                 q_lars_over_off=(r_l['Q'] / q_off if q_off else None),
                                 q_adv_over_off=(r_a['Q'] / q_off if q_off else None),
                                 q_adv_over_lars=(r_a['Q'] / r_l['Q'] if r_l['Q'] else None)))

    save_maps(ret_lars, ret_adv, lab, scene, gas, psave)
    return rows_out, ret_lars, ret_adv


def save_maps(ret_lars, ret_adv, lab, scene, gas, psave):
    try:
        pos = np.concatenate([ret_lars[np.isfinite(ret_lars)], ret_adv[np.isfinite(ret_adv)]])
        vmax = float(np.percentile(pos, 99.7))
        fig, axes = plt.subplots(1, 2, figsize=(19, 10), facecolor='black', sharex=True, sharey=True)
        for ax, arr, ttl in zip(axes, (ret_lars, ret_adv),
                                ('MF por columna (LARS)', 'MF grupos+PCA/kmeans+shrinkage')):
            ax.set_facecolor('black')
            im = ax.imshow(arr, cmap='inferno', vmin=0, vmax=vmax)
            ax.contour(lab > 0, levels=[0.5], colors='cyan', linewidths=0.7)
            ax.set_title(f'{ttl}\nstd={np.nanstd(arr):.1f} ppm.m', color='white')
            ax.axis('off')
            cb = plt.colorbar(im, ax=ax, shrink=0.7,
                              label=rf'$\Delta$X{gas.upper()} (ppm$\cdot$m)')
            cb.ax.yaxis.label.set_color('white'); cb.ax.tick_params(colors='white')
        fig.suptitle(f'{scene}  |  misma escala, contorno cian = huella oficial',
                     color='white', fontsize=14)
        plt.tight_layout()
        out = os.path.join(psave, f'{scene}_compare_{gas}.png')
        fig.savefig(out, dpi=120, facecolor='black')
        plt.close(fig)
        print(f'[mapa] {out}')
    except Exception as e:
        print(f'[mapa] omitido ({e})')


# ------------------------------------------------------------------

def print_table(rows, gas):
    if not rows:
        print('\nSin plumas cuantificadas.')
        return
    print(f'\n{"="*118}')
    print(f'{"escena":<22}{"plumas":<10}{"N":>5}{"u10":>6}{"IME_L":>9}{"IME_A":>9}{"IME_of":>9}'
          f'{"Q_LARS":>10}{"Q_ADV":>10}{"Q_ofic":>10}{"A/L":>7}{"L/of":>8}{"A/of":>8}')
    print('-' * 118)
    for r in rows:
        ratio = r['q_adv'] / r['q_lars'] if r['q_lars'] else float('nan')
        dl = r['q_lars'] / r['q_off'] if r['q_off'] else float('nan')
        da = r['q_adv'] / r['q_off'] if r['q_off'] else float('nan')
        print(f'{r["scene"]:<22}{r["plumes"]:<10}{r["n_pix"]:>5}{r["u10"]:>6.2f}'
              f'{r["ime_lars"]:>9.1f}{r["ime_adv"]:>9.1f}{r["ime_off"]:>9.1f}'
              f'{r["q_lars"]:>10.1f}{r["q_adv"]:>10.1f}{r["q_off"]:>10.1f}'
              f'{ratio:>7.2f}{dl:>8.2f}{da:>8.2f}')
    print('=' * 118)
    tl = sum(r['q_lars'] for r in rows); ta = sum(r['q_adv'] for r in rows)
    to = sum(r['q_off'] for r in rows)
    print(f'TOTAL   Q_LARS={tl:.0f}   Q_ADV={ta:.0f}   Q_oficial={to:.0f} kg/h {gas.upper()}')
    if to:
        print(f'        LARS/oficial={tl/to:.2f}   ADV/oficial={ta/to:.2f}   ADV/LARS={ta/tl:.2f}')
    print('  A/L = Q_ADV / Q_LARS   |   L/of y A/of = cociente contra el producto oficial')


def main():
    ap = argparse.ArgumentParser(description='Compara el MF por columna (LARS) contra el MF por grupos de columnas con fondo PCA/k-means (HyGas) sobre las mismas plumas.')
    ap.add_argument('rad_files', nargs='+', help='Uno o varios .h5 ortho_radiance de Tanager.')
    ap.add_argument('--gas', default='ch4', choices=list(GAS_LUT))
    ap.add_argument('-o', '--output-dir', default=None, help='Carpeta de salida (por defecto: out_compare/ junto al primer .h5).')
    ap.add_argument('--force', action='store_true', help='Recalcula los MF aunque exista cache .npy.')
    ap.add_argument('--slim', action='store_true',
                    help='CSV reducido: solo escena, pluma, calidad, lat/lon, u10, N y las tres Q (GT, MF normal, MF por grupos de columnas).')
    ap.add_argument('--per-plume', action='store_true',
                    help='Una fila por pluma del GeoJSON. Cuando varias comparten huella conexa, la huella se reparte por fuente mas cercana. Sin la flag, la huella compartida se cuantifica una sola vez contra la suma de sus emisiones.')
    ap.add_argument('--ueff-mode', choices=['cm', 'lars'], default='cm',
                    help="Velocidad efectiva. 'cm' = Ueff=u10, la de Carbon Mapper (deducida exacta de las 20 plumas del GeoJSON); es la unica que permite comparar Q contra Q. 'lars' = a*u10+b, la parametrizacion original del repo.")
    ap.add_argument('--l-mode', choices=['sqrt-area', 'fetch', 'hull'], default='sqrt-area',
                    help="Escala de longitud. 'sqrt-area' = sqrt(N*gsd^2), reproduce el fetch oficial con mediana 0.92. 'fetch' = el del GeoJSON tal cual. 'hull' = diametro del convex hull (mediana 1.73 del fetch, no comparable).")
    ap.add_argument('--group-min', type=int, default=10)
    ap.add_argument('--group-max', type=int, default=30)
    ap.add_argument('--clusters', type=int, default=3)
    ap.add_argument('--shrinkage', type=float, default=0.1)
    ap.add_argument('--per-cluster-targets', action='store_true',
                    help='Escala el target por la media de cada cluster (mas parecido a la formulacion de LARS).')
    ap.add_argument('--adaptive-shrinkage', action='store_true')
    args = ap.parse_args()

    files = []
    for pat in args.rad_files:
        files.extend(sorted(glob.glob(pat)) or [pat])

    psave = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(files[0])), 'out_compare')
    os.makedirs(psave, exist_ok=True)

    adv_kwargs = dict(group_min=args.group_min, group_max=args.group_max,
                      n_clusters=args.clusters, shrinkage=args.shrinkage,
                      per_cluster_targets=args.per_cluster_targets,
                      adaptive_shrinkage=args.adaptive_shrinkage)

    all_rows = []
    for f in files:
        try:
            rows, _, _ = process_scene(f, args.gas, psave, adv_kwargs, force=args.force,
                                       ueff_mode=args.ueff_mode, l_mode=args.l_mode,
                                       per_plume=args.per_plume)
            all_rows.extend(rows)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'[!] escena {f} fallo: {e}')

    print_table(all_rows, args.gas)

    if all_rows:
        out_csv = os.path.join(psave, f'compare_mf_{args.gas}.csv')
        if args.slim:
            #Lo esencial: donde esta la pluma, con que viento, y las TRES Q.
            rows = [dict(escena=r['scene'], pluma=r['plumes'], calidad=r['quality'],
                         lat=r['lat'], lon=r['lon'], u10=round(r['u10'], 2),
                         N_px=r['n_pix'],
                         Q_GT=round(r['q_off'], 1),
                         Q_MF_normal=round(r['q_lars'], 1),
                         Q_MF_columnas=round(r['q_adv'], 1))
                    for r in all_rows]
        else:
            rows = all_rows
        with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
            wcsv = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wcsv.writeheader(); wcsv.writerows(rows)
        print(f'\n[csv] {out_csv}')


if __name__ == '__main__':
    main()
