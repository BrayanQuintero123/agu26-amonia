"""
Cuantificacion interactiva multi-pluma CH4 Tanager con deteccion wavelet.

Flujo:
  1. Carga mf_ch4.npy (ya calculado)
  2. Wavelet den -> encuentra candidatos (maximos locales > P98)
  3. Imprime en consola Google Maps link de cada candidato
  4. Abre ventana: enhancement + candidatos marcados
  5. Usuario da click DENTRO de cada pluma que quiere cuantificar (boton izq)
     Boton derecho = deshacer ultimo. Enter = terminar seleccion.
  6. Cada click crece la pluma automaticamente (>sigma*std, componente conexa)
     — no hay que dibujar poligono.
  7. Cuantifica todas las plumas seleccionadas y muestra tabla resumen

Uso:
  python scripts/quantify_ch4_tanager_multi.py
  python scripts/quantify_ch4_tanager_multi.py --p98 99 --min-sep 15
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'auto_pipeline'))

import numpy as np
from scipy.ndimage import maximum_filter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.path
import pyproj
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

from wavelet_den import compute_den
from auto_plume import grow_plume_from_source
from quant_func_v2 import ppmm_to_kg, extract_ueff
from wind_v2 import wind_speed_manual
from store_results import excel_info

# ------------------------------------------------------------------
# Configuracion
# ------------------------------------------------------------------
MF_TIF  = "C:/Users/jefer/OneDrive/Documentos/GitHub/agu26-amonia2/out_tanager/mf_ch4.tif"
MF_NPY  = "C:/Users/jefer/OneDrive/Documentos/GitHub/agu26-amonia2/out_tanager/mf_ch4.npy"
PSAVE   = "C:/Users/jefer/OneDrive/Documentos/GitHub/agu26-amonia2/out_tanager/"
NAME    = "20241004_081921_28_4001"
SITE    = "Deir ez-Zor"
MISSION = 'Tanager'
PIX_M   = 30.0
EPSG    = 32637
GT      = [609150.0, 30.0, 0.0, 3922020.0, 0.0, -30.0]

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def utm_to_lonlat(x, y):
    tr = pyproj.Transformer.from_crs(f'EPSG:{EPSG}', 'EPSG:4326', always_xy=True)
    lon, lat = tr.transform(x, y)
    return float(lon), float(lat)


def pixel_to_utm(col, row):
    e = GT[0] + (col + 0.5) * GT[1]
    n = GT[3] + (row + 0.5) * GT[5]
    return e, n


def pixel_to_lonlat(col, row):
    e, n = pixel_to_utm(col, row)
    return utm_to_lonlat(e, n)


def find_candidates(den, ret, percentile=98, min_sep_px=15, n_sigma=2.0, top=25,
                    min_dist_m=600.0):
    """
    Maximos locales del wavelet den sobre el percentil dado, filtrados ademas
    por el enhancement crudo (>= n_sigma * std) para descartar ruido wavelet.

    Luego aplica supresion greedy: recorriendo de mayor a menor intensidad, se
    descarta todo candidato que caiga a menos de `min_dist_m` metros de uno ya
    aceptado. Asi no salen varios puntos pegados sobre la misma pluma.
    Finalmente recorta a los `top` mas intensos.
    """
    thr = np.percentile(den[den > 0], percentile)
    std = float(np.nanstd(ret))
    ret_f = np.nan_to_num(ret, nan=0.0)
    peak_map = (den >= thr) & (den == maximum_filter(den, size=min_sep_px)) \
               & (ret_f >= n_sigma * std)
    rows, cols = np.where(peak_map)
    vals = den[rows, cols]
    order = np.argsort(vals)[::-1]
    rows, cols, vals = rows[order], cols[order], vals[order]

    # --- supresion greedy por distancia real (metros) ---
    min_d2 = (min_dist_m / PIX_M) ** 2   # en pixeles^2
    keep_r, keep_c, keep_v = [], [], []
    for r, c, v in zip(rows, cols, vals):
        if any((r - kr)**2 + (c - kc)**2 < min_d2 for kr, kc in zip(keep_r, keep_c)):
            continue
        keep_r.append(r); keep_c.append(c); keep_v.append(v)
        if len(keep_r) >= top:
            break

    return np.array(keep_r), np.array(keep_c), np.array(keep_v)


def utm_to_pixel(x, y, shape):
    """UTM (m) -> (row, col) recortado al raster."""
    col = int((x - GT[0]) / GT[1])
    row = int((y - GT[3]) / GT[5])
    row = int(np.clip(row, 0, shape[0] - 1))
    col = int(np.clip(col, 0, shape[1] - 1))
    return row, col


def convex_hull_L(mask):
    r, c = np.where(mask)
    if len(r) < 3:
        return np.sqrt(max(len(r), 1)) * PIX_M
    pts = np.column_stack((c, r)).astype(float) * PIX_M
    return float(np.max(pdist(pts[ConvexHull(pts).vertices])))


def quantify(ret, mask, lon_s, lat_s, label, u10):
    pix_area_cm2 = PIX_M**2 * 1e4
    gas = 'ch4'
    sum_xgas = float(np.nansum(np.where(mask, np.nan_to_num(ret, nan=0), 0)))
    IME, _ = ppmm_to_kg(sum_xgas, PIX_M, gas)
    L = convex_hull_L(mask)
    N = int(mask.sum())
    ueff, a, err_u10, _, bool_thres = extract_ueff(u10, MISSION, gas)
    Q = IME * 3600 * ueff / L
    conv = IME / max(sum_xgas, 1e-9)
    err_Q = np.sqrt((3600*IME*a*2/L)**2 +
                    (3600*ueff*conv*np.sqrt(N)*np.nanstd(ret[mask])/L)**2) if bool_thres else \
            np.sqrt((3600*IME*a*u10*0.5/L)**2 +
                    (3600*ueff*conv*np.sqrt(N)*np.nanstd(ret[mask])/L)**2)
    print(f'\n  [{label}] IME={IME:.2f} kg | L={L:.1f} m | N={N}')
    print(f'  u10={u10:.2f} m/s | Q(CH4)={Q:.2f} ± {err_Q:.2f} kg/h')
    ts = NAME[:8] + NAME[9:15]
    fields = ['Site', 'Mission', 'Timestamp', 'lat', 'lon', 'u10 (m/s)', 'err(u10)', 'Q (kg/h)', 'err(Q)']
    info = [SITE, MISSION, ts, round(lat_s,4), round(lon_s,4),
            round(u10,2), round(err_u10,2), round(Q,2), round(err_Q,2)]
    excel_info(fields, info, PSAVE, 'ch4')
    return dict(label=label, IME=IME, L=L, u10=u10, Q=Q, err_Q=err_Q, N=N, lat=lat_s, lon=lon_s)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pctl', '--p98', dest='p98', type=float, default=99.8,
                    help='Percentil del wavelet den para candidatos (default: 99.8)')
    ap.add_argument('--min-sep', type=int, default=15,
                    help='Separacion minima entre candidatos en pixeles (default: 15)')
    ap.add_argument('--n-sigma', type=float, default=2.5,
                    help='Umbral en el enhancement crudo: n*std (default: 2.5)')
    ap.add_argument('--min-dist', type=float, default=1000.0,
                    help='Distancia minima real entre candidatos, en metros (default: 1000)')
    ap.add_argument('--top', type=int, default=15,
                    help='Maximo numero de candidatos a mostrar (default: 25)')
    ap.add_argument('--sigma', type=float, default=2.0,
                    help='Multiplicador de sigma para el crecimiento de la pluma (default: 2.0)')
    ap.add_argument('--wind-value', type=float, default=None,
                    help='Viento manual en m/s (si no se da, se descarga GEOS-FP)')
    args = ap.parse_args()

    # --- cargar enhancement ---
    ret = np.load(MF_NPY)
    rows_n, cols_n = ret.shape
    extent = [GT[0], GT[0] + cols_n*GT[1], GT[3] + rows_n*GT[5], GT[3]]
    std = float(np.nanstd(ret))

    # --- wavelet ---
    print('Calculando wavelet den...', flush=True)
    den = compute_den(ret)
    print(f'  den: max={den.max():.1f}  p98={np.percentile(den[den>0],98):.1f}', flush=True)

    # --- candidatos ---
    cand_rows, cand_cols, cand_vals = find_candidates(
        den, ret, args.p98, args.min_sep, args.n_sigma, args.top, args.min_dist)
    print(f'\n{len(cand_rows)} candidatos (wavelet P{args.p98:g}, '
          f'>{args.n_sigma:g}σ, sep ≥{args.min_dist:g} m):')
    print(f'  {"#":<4} {"lat":>10} {"lon":>11} {"den":>8}   Google Maps')
    print('  ' + '-'*70)
    candidates = []
    for k, (r, c, v) in enumerate(zip(cand_rows, cand_cols, cand_vals)):
        lon, lat = pixel_to_lonlat(c, r)
        gm = f'https://www.google.com/maps?q={lat:.6f},{lon:.6f}&z=16'
        print(f'  {k+1:<4} {lat:>10.5f} {lon:>11.5f} {v:>8.1f}   {gm}')
        candidates.append(dict(k=k+1, row=r, col=c, lat=lat, lon=lon, gm=gm))

    # --- ventana interactiva ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 10), facecolor='black')
    ax_enh, ax_wav = axes

    for ax in axes:
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.set_xlabel('Easting (m)', color='white')

    im1 = ax_enh.imshow(ret, extent=extent, cmap='inferno', vmin=0, vmax=3*std, origin='upper')
    ax_enh.set_title(r'Enhancement $\Delta$XCH$_4$ (ppm·m)', color='white', fontsize=10)
    plt.colorbar(im1, ax=ax_enh, shrink=0.7, label='ppm·m').ax.tick_params(colors='white')

    im2 = ax_wav.imshow(den, extent=extent, cmap='inferno',
                        vmin=0, vmax=np.percentile(den[den>0], 99.5), origin='upper')
    ax_wav.set_title('Wavelet den (anomalia de pluma)', color='white', fontsize=10)
    plt.colorbar(im2, ax=ax_wav, shrink=0.7).ax.tick_params(colors='white')

    # marcar candidatos en ambos paneles
    for cd in candidates:
        e, n = pixel_to_utm(cd['col'], cd['row'])
        for ax in axes:
            ax.plot(e, n, 'o', ms=8, mfc='none', mec='cyan', mew=1.5, zorder=5)
            ax.text(e + 200, n, str(cd['k']), color='cyan', fontsize=7, zorder=6)

    fig.suptitle(
        'Click izq DENTRO de la pluma | Click der = deshacer | Enter = terminar',
        color='white', fontsize=10)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)

    # --- seleccion de hotspots ---
    print(f'\nHaz click DENTRO de cada pluma que quieres cuantificar '
          f'(la mascara se crece sola a >{args.sigma:g}σ).')
    print('Click izquierdo = añadir, click derecho = deshacer, Enter = terminar.\n')
    selected_xy = plt.ginput(n=-1, timeout=0, show_clicks=True,
                             mouse_add=1, mouse_pop=3, mouse_stop=2)

    if not selected_xy:
        print('No se selecciono ningun punto.')
        plt.close(fig)
        return

    # se sigue trabajando sobre el panel del enhancement de la misma ventana
    ax2 = ax_enh
    ax2.set_autoscale_on(False)
    for sx, sy in selected_xy:
        ax2.scatter(sx, sy, s=180, facecolor='white', edgecolor='red',
                    linewidth=2, marker='*', zorder=6)
    fig.suptitle('Cuantificando plumas seleccionadas...', color='white', fontsize=10)
    plt.draw(); plt.pause(0.3)

    results = []
    for idx, (sx, sy) in enumerate(selected_xy):
        label_p = f'P{idx+1}'
        lon_s, lat_s = utm_to_lonlat(sx, sy)
        gm = f'https://www.google.com/maps?q={lat_s:.6f},{lon_s:.6f}&z=16'
        print(f'\n[{label_p}] Fuente: lat={lat_s:.6f} lon={lon_s:.6f}')
        print(f'  Google Maps -> {gm}')

        # --- crecimiento automatico desde el pixel clickeado ---
        src_row, src_col = utm_to_pixel(sx, sy, ret.shape)
        mask, status = grow_plume_from_source(np.nan_to_num(ret, nan=0.0),
                                              src_row, src_col, args.sigma)

        if status != 'ok':
            # reintento: buscar el maximo local en una ventana alrededor del click
            win = 5
            r0, r1 = max(0, src_row-win), min(ret.shape[0], src_row+win+1)
            c0, c1 = max(0, src_col-win), min(ret.shape[1], src_col+win+1)
            sub = np.nan_to_num(ret[r0:r1, c0:c1], nan=0.0)
            dr, dc = np.unravel_index(np.argmax(sub), sub.shape)
            src_row, src_col = r0 + dr, c0 + dc
            mask, status = grow_plume_from_source(np.nan_to_num(ret, nan=0.0),
                                                  src_row, src_col, args.sigma)

        if status != 'ok' or mask is None or mask.sum() < 5:
            print(f'[{label_p}] El pixel no supera {args.sigma:g}σ '
                  f'(o la isla es < 5 px). Descartado. Prueba --sigma menor.')
            continue

        # la fuente real es el pixel crecido, no el click
        lon_s, lat_s = pixel_to_lonlat(src_col, src_row)
        print(f'  -> fuente ajustada: lat={lat_s:.6f} lon={lon_s:.6f} | '
              f'{int(mask.sum())} px sobre {args.sigma:g}σ')

        ax2.contour(mask, levels=[0.5], colors='yellow', linewidths=1.5,
                    extent=extent, origin='upper', zorder=4)
        ax2.text(sx, sy, f' {label_p}', color='yellow', fontsize=9, zorder=7)
        plt.draw(); plt.pause(0.2)

        # viento
        ts_str = NAME[:8] + NAME[9:15]
        if args.wind_value is not None:
            u10 = args.wind_value
        else:
            u10 = wind_speed_manual(ts_str, lat_s, lon_s, PSAVE, PSAVE, label_p)

        res = quantify(ret, mask, lon_s, lat_s, label_p, u10)
        results.append(res)

    fig.suptitle('Plumas cuantificadas (contorno amarillo)', color='white', fontsize=10)
    fig.savefig(os.path.join(PSAVE, 'plumas_seleccionadas.png'),
                dpi=150, facecolor='black')
    plt.close(fig)

    if not results:
        print('\nNo se cuantifico ninguna pluma.')
        return

    # --- tabla resumen ---
    print(f'\n{"="*65}')
    print(f'{"Pluma":<7} {"lat":>10} {"lon":>11} {"IME(kg)":>9} {"L(m)":>7} {"u10":>5} {"Q(kg/h)":>9} {"err":>7}')
    print('-'*65)
    for r in results:
        print(f'{r["label"]:<7} {r["lat"]:>10.5f} {r["lon"]:>11.5f} '
              f'{r["IME"]:>9.2f} {r["L"]:>7.1f} {r["u10"]:>5.2f} '
              f'{r["Q"]:>9.2f} {r["err_Q"]:>7.2f}')
    print('='*65)
    print(f'Total Q = {sum(r["Q"] for r in results):.2f} kg/h  CH4')


if __name__ == '__main__':
    main()
