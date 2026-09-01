#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figura del pipeline: las seis etapas con los productos REALES de una pluma, no
cajas con flechas.

  L1 radiance -> matched filter (ppm.m) -> wavelet anomaly -> k*sigma candidates
              -> connected-component mask -> IME*Ueff/L -> Q (kg/h)

Reusa el cache de MF que dejan compare_mf.py / pick_plumes_wavelet.py, asi que si
ya corriste cualquiera de los dos esto tarda segundos.

Uso:
  python auto_pipeline/figure_pipeline.py tanager/20241004_081921_28_4001_ortho_radiance_hdf5.h5 --plume D
  python auto_pipeline/figure_pipeline.py <.h5> --plume E --mf-dir tanager/out_compare -o results
"""

import os
import sys
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(os.path.dirname(_HERE), 'scripts')
for _p in (_SCRIPTS, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from scipy.signal import medfilt2d
from skimage import measure

from Tanager_reader import read_Tanager, get_ortho_framing
from wavelet_den import compute_den
from plumes_from_json import read_plumes_json, find_plumes_json
from compare_mf import source_pixel, quantify, grow_mask_ksigma

PIX_RES = 30.0
RAD_BAND_NM = 2100.0          # dentro de la ventana SWIR del retrieval


def crop_around(mask, pad=28):
    r, c = np.where(mask)
    r0, r1 = max(r.min() - pad, 0), min(r.max() + pad + 1, mask.shape[0])
    c0, c1 = max(c.min() - pad, 0), min(c.max() + pad + 1, mask.shape[1])
    #cuadramos el recorte para que los seis paneles compartan aspecto
    h, w = r1 - r0, c1 - c0
    if h > w:
        g = (h - w) // 2
        c0, c1 = max(c0 - g, 0), min(c1 + g, mask.shape[1])
    elif w > h:
        g = (w - h) // 2
        r0, r1 = max(r0 - g, 0), min(r1 + g, mask.shape[0])
    return r0, r1, c0, c1


def panel(ax, arr, cmap, vmin, vmax, step, title, sub):
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor('#c8cdd6'); s.set_linewidth(0.8)
    ax.text(0.035, 0.965, step, transform=ax.transAxes, va='top', ha='left',
            fontsize=8, fontweight='bold', color='#b4571a', family='monospace',
            bbox=dict(boxstyle='square,pad=0.28', fc='white', ec='none', alpha=0.88))
    ax.set_title(title, fontsize=10.5, fontweight='bold', pad=7, color='#12161d')
    ax.set_xlabel(sub, fontsize=8.2, color='#4c5665', labelpad=5)


def main():
    ap = argparse.ArgumentParser(description='Figura de las seis etapas del pipeline sobre una pluma real.')
    ap.add_argument('rad_file')
    ap.add_argument('--plume', default=None, help='Etiqueta de la pluma del GeoJSON (A, B, ...). Por defecto, la de mayor emision.')
    ap.add_argument('--gas', default='ch4')
    ap.add_argument('--sigma', type=float, default=2.0)
    ap.add_argument('--min-pix', type=int, default=10)
    ap.add_argument('--mf-dir', default=None, help='Carpeta con el cache <scene>_mf_lars_<gas>.npy. Por defecto out_compare junto al .h5.')
    ap.add_argument('-o', '--output-dir', default='results')
    args = ap.parse_args()

    rad = os.path.abspath(args.rad_file)
    p = os.path.dirname(rad) + os.sep
    n = os.path.splitext(os.path.basename(rad))[0]
    scene = n.split('_ortho_')[0]
    mf_dir = args.mf_dir or os.path.join(p, 'out_compare')
    cache = os.path.join(mf_dir, f'{scene}_mf_lars_{args.gas}.npy')
    if not os.path.exists(cache):
        sys.exit(f'No encuentro el cache del MF: {cache}\nCorre antes compare_mf.py o pick_plumes_wavelet.py.')

    geo = get_ortho_framing(p, n)
    gt, epsg = geo['geotransform'], geo['epsg_code']
    shape = (geo['rows'], geo['cols'])

    # --- etapa 2: matched filter (del cache) ---
    mf = np.load(cache)

    # --- pluma objetivo ---
    plumes = read_plumes_json(find_plumes_json(rad), args.gas)
    if args.plume:
        sel = [q for q in plumes if q['label'] == args.plume.upper()]
        if not sel:
            sys.exit(f'No hay pluma "{args.plume}" en el GeoJSON. Hay: {[q["label"] for q in plumes]}')
    else:
        sel = [max(plumes, key=lambda q: q['q_official'] or 0)]
    pl = sel[0]
    src = source_pixel(pl['lon'], pl['lat'], gt, epsg, shape)
    if src is None:
        sys.exit('La fuente de esa pluma cae fuera de la escena.')

    # --- etapa 5: mascara ---
    mask = grow_mask_ksigma(mf, [src], args.sigma)
    if mask is None:
        sys.exit(f'La pluma no crece a {args.sigma}sigma.')
    r0, r1, c0, c1 = crop_around(mask)

    # --- etapa 1: radiancia L1 ---
    print('[read] leyendo una banda de radiancia...', flush=True)
    img, wvl, _, _, _ = read_Tanager(p, n)
    b = int(np.argmin(np.abs(wvl - RAD_BAND_NM)))
    radb = img[r0:r1, c0:c1, b].astype(float)
    del img

    # --- etapa 3: anomalia wavelet ---
    den = compute_den(mf)

    # --- etapa 4: candidatos k*sigma en el recorte ---
    filt = medfilt2d(np.nan_to_num(mf.astype(np.float32)), kernel_size=3)
    std = float(np.nanstd(mf))
    lab = measure.label((filt > args.sigma * std) & (filt < 1e5), connectivity=1)
    cands = [rg for rg in measure.regionprops(lab[r0:r1, c0:c1]) if rg.area >= args.min_pix]

    # --- etapa 6: cuantificacion ---
    res = quantify(mf, mask, pl['u10_official'], args.gas, 'cm', 'sqrt-area', None)

    # ---------------- figura ----------------
    mf_c, den_c = mf[r0:r1, c0:c1], den[r0:r1, c0:c1]
    mask_c = mask[r0:r1, c0:c1]

    fin = mf_c[np.isfinite(mf_c)]
    vmax_mf = float(np.nanpercentile(fin, 99)) if fin.size else 1.0
    pos = den_c[den_c > 0]
    vmax_den = float(np.nanpercentile(pos, 99)) if pos.size else 1.0
    rv = radb[np.isfinite(radb)]
    vmin_r, vmax_r = (np.percentile(rv, 2), np.percentile(rv, 98)) if rv.size else (0, 1)

    fig, axes = plt.subplots(2, 3, figsize=(11.6, 8.4), facecolor='white')
    fig.subplots_adjust(left=0.035, right=0.965, top=0.845, bottom=0.075,
                        wspace=0.16, hspace=0.34)
    ax = axes.ravel()

    panel(ax[0], radb, 'gray', vmin_r, vmax_r, '01',
          'L1 radiance',
          f'{RAD_BAND_NM:.0f} nm  ·  W m$^{{-2}}$ sr$^{{-1}}$ $\\mu$m$^{{-1}}$')

    panel(ax[1], mf_c, 'inferno', 0, vmax_mf, '02',
          'Matched filter',
          f'$\\Delta$X{args.gas.upper()} (ppm$\\cdot$m)  ·  $\\sigma$ = {std:.0f}')

    panel(ax[2], medfilt2d(den_c.astype(np.float32), 3), 'inferno', 0, vmax_den, '03',
          'Wavelet anomaly',
          'background $\\approx$ 0, sources highlighted')

    # 04 candidatos sobre el realce
    panel(ax[3], mf_c, 'inferno', 0, vmax_mf, '04',
          f'{args.sigma:g}$\\sigma$ candidates',
          f'{len(cands)} cluster(s) $\\geq$ {args.min_pix} px in this crop')
    for i, rg in enumerate(sorted(cands, key=lambda g: -g.area), 1):
        cy, cx = rg.centroid
        ax[3].add_patch(Circle((cx, cy), max(np.sqrt(rg.area) * 0.9, 5),
                               fill=False, ec='#4fd6e0', lw=1.4))
        ax[3].text(cx + 6, cy - 6, str(i), color='#ffe066', fontsize=9,
                   fontweight='bold')

    # 05 mascara. Fuera de la pluma hay NaN, o sea que se ve el fondo del eje: tiene
    # que ser negro como el de los otros paneles de realce.
    ax[4].set_facecolor('black')
    panel(ax[4], np.where(mask_c, mf_c, np.nan), 'inferno', 0, vmax_mf, '05',
          'Connected-component mask',
          f'{int(mask_c.sum())} px  ·  L = $\\sqrt{{N}}\\cdot$gsd = {res["L"]:.0f} m')
    ax[4].contour(mask_c, levels=[0.5], colors='#4fd6e0', linewidths=1.2)
    ax[4].scatter(src[1] - c0, src[0] - r0, s=120, marker='*',
                  facecolor='white', edgecolor='#b4571a', linewidth=1.4, zorder=5)

    # 06 resultado
    a6 = ax[5]
    a6.set_xlim(0, 1); a6.set_ylim(0, 1)   # sin imagen no hay limites de datos utiles
    a6.set_xticks([]); a6.set_yticks([])
    a6.set_facecolor('#f7f8fa')
    for s in a6.spines.values():
        s.set_edgecolor('#c8cdd6'); s.set_linewidth(0.8)
    a6.text(0.035, 0.965, '06', transform=a6.transAxes, va='top', ha='left',
            fontsize=8, fontweight='bold', color='#b4571a', family='monospace')
    a6.set_title('IME $\\cdot$ U$_{eff}$ / L', fontsize=10.5, fontweight='bold',
                 pad=7, color='#12161d')
    a6.set_xlabel('U$_{eff}$ = u$_{10}$,  L = $\\sqrt{N\\cdot gsd^2}$',
                  fontsize=8.2, color='#4c5665', labelpad=5)

    rows = [('IME',            f'{res["IME"]:.0f} kg'),
            ('L',              f'{res["L"]:.0f} m'),
            ('u$_{10}$',       f'{pl["u10_official"]:.2f} m s$^{{-1}}$'),
            ('N',              f'{res["N"]} px')]
    y = 0.80
    for k, v in rows:
        a6.text(0.10, y, k, fontsize=10, color='#4c5665', ha='left', va='center',
                transform=a6.transAxes)
        a6.text(0.90, y, v, fontsize=10.5, color='#12161d', ha='right', va='center',
                family='monospace', transform=a6.transAxes)
        y -= 0.115
    a6.plot([0.10, 0.90], [y + 0.045, y + 0.045], color='#c8cdd6', lw=0.9,
            transform=a6.transAxes, clip_on=False)
    a6.text(0.10, y - 0.075, 'Q', fontsize=13, color='#12161d', ha='left',
            va='center', fontweight='bold', transform=a6.transAxes)
    a6.text(0.90, y - 0.075, f'{res["Q"]:.0f} kg h$^{{-1}}$', fontsize=15,
            color='#b4571a', ha='right', va='center', fontweight='bold',
            family='monospace', transform=a6.transAxes)
    if pl['q_official']:
        a6.text(0.90, y - 0.195, f'Carbon Mapper: {pl["q_official"]:.0f} kg h$^{{-1}}$',
                fontsize=8.4, color='#7b8595', ha='right', va='center',
                family='monospace', transform=a6.transAxes)

    # --- flechas entre paneles ---
    def arrow(a, b, dy=0.0):
        p1 = a.get_position(); p2 = b.get_position()
        fig.patches.append(FancyArrowPatch(
            (p1.x1 + 0.008, (p1.y0 + p1.y1) / 2 + dy),
            (p2.x0 - 0.008, (p2.y0 + p2.y1) / 2 + dy),
            transform=fig.transFigure, arrowstyle='-|>', mutation_scale=13,
            lw=1.1, color='#9aa3b0', shrinkA=0, shrinkB=0))
    #Solo dentro de cada fila. El salto 03 -> 04 lo marcan los numerales: cualquier
    #flecha que lo dibuje cruza la figura entera y ensucia mas de lo que aclara.
    for i in (0, 1, 3, 4):
        arrow(ax[i], ax[i + 1])

    fig.text(0.035, 0.955,
             'From radiance cube to emission rate',
             fontsize=16, fontweight='bold', color='#12161d', ha='left', va='top')
    fig.text(0.035, 0.905,
             f'Tanager  ·  scene {scene}  ·  plume {pl["label"]}  ·  '
             f'{pl["lat"]:.4f}, {pl["lon"]:.4f}  ·  30 m',
             fontsize=9.2, color='#4c5665', ha='left', va='top', family='monospace')
    fig.text(0.965, 0.955,
             'The algorithm delineates,\nthe operator confirms at step 04',
             fontsize=8.6, color='#7b8595', ha='right', va='top', style='italic')

    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.join(args.output_dir, f'fig_pipeline_{scene}_{pl["label"]}')
    for ext, kw in (('png', dict(dpi=220)), ('pdf', {}), ('svg', {})):
        fig.savefig(f'{base}.{ext}', facecolor='white', bbox_inches='tight', **kw)
        print(f'[fig] {base}.{ext}')
    plt.close(fig)


if __name__ == '__main__':
    main()
