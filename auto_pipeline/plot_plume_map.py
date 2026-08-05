#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Superpone la pluma detectada sobre un mapa base real (satelite / calles) y sobre
la radiancia, para verificacion visual del sitio.

Misma idea que emit_ime::_draw_plume_panel: se pinta el ENHANCEMENT FISICO solo
en los pixeles de la mascara (resto transparente), escalado al p98 DENTRO de la
mascara. El basemap se trae con contextily (Esri satelite / OpenStreetMap);
los tiles de Google no se pueden usar asi (su ToS lo prohibe fuera de su API).

Dos formas de uso:
  - Integrada: el pipeline llama render_plume_map(...) con la mascara ya aceptada
    (no re-crece nada) -> el PNG se genera solo al aceptar la pluma.
  - Standalone (CLI): re-crece la mascara desde una lat/lon y la dibuja.

El render usa un canvas Agg off-screen (no toca el backend interactivo del pipeline,
asi que no aparecen ventanas espurias durante la seleccion).

CLI:
  python auto_pipeline/plot_plume_map.py --lat 7.0776 --lon -73.1427 --gas ch4 --sigma 2.0 --path-img "<carpeta del .nc>"
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
from matplotlib.figure import Figure                      # off-screen figure, no pyplot backend needed
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize
import matplotlib.cm as mcm
from scipy.signal import medfilt2d

from georreferencing import warp_array_for_display
from gas_bands import GAS_BANDS


def render_plume_map(mask, gas_enh, rad, lat, lon, list_gcp, lat_src, lon_src,
                     gas, psave, name, sigma=None, pad_deg=0.012, wind_uv=None):
    """Render the (already-accepted) plume mask over satellite / OSM / radiance basemaps.
    Renders off-screen (Agg canvas) so it never disturbs an interactive matplotlib session.
    Saves output/plume_on_map_<gas>.png and returns its path (or None if the mask is empty)."""

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        print('  [map] empty mask, nothing to plot.')
        return None

    #--- warp to north-up (regular lon/lat grid) so we can place it on a basemap by extent ---
    enh_disp = medfilt2d(np.asarray(gas_enh, dtype=np.float32), kernel_size=3)
    rad_w, gt = warp_array_for_display(list_gcp, np.where(rad > 0, rad, np.nan))
    enh_w, _ = warp_array_for_display(list_gcp, enh_disp, ref_gt=gt, ref_shape=rad_w.shape)
    mask_w, _ = warp_array_for_display(list_gcp, mask.astype(np.float32), ref_gt=gt, ref_shape=rad_w.shape)
    mask_wb = np.nan_to_num(mask_w) > 0.5
    if not mask_wb.any():
        print('  [map] mask fell outside the warped grid, nothing to plot.')
        return None

    #--- plume RGBA: physical enhancement, opaque only inside the mask (emit_ime recipe) ---
    vals_in = enh_w[mask_wb]; vals_in = vals_in[np.isfinite(vals_in)]
    vmax_p = float(np.nanpercentile(vals_in, 98)) if vals_in.size else 1.0
    norm = Normalize(vmin=0, vmax=max(vmax_p, 1e-6))
    cmap = mcm.inferno
    rgba = cmap(norm(np.clip(np.nan_to_num(enh_w), 0, None)))
    rgba[..., 3] = np.where(mask_wb, 0.85, 0.0)

    #--- zoom window around the plume ---
    rw, cw = np.where(mask_wb)
    lon_lo = gt[0] + cw.min() * gt[1] - pad_deg
    lon_hi = gt[0] + cw.max() * gt[1] + pad_deg
    lat_hi = gt[3] + rw.min() * gt[5] + pad_deg
    lat_lo = gt[3] + rw.max() * gt[5] - pad_deg
    ext = [gt[0], gt[0] + rad_w.shape[1] * gt[1], gt[3] + rad_w.shape[0] * gt[5], gt[3]]

    try:
        import contextily as ctx
        ctx_ok = True
    except ImportError:
        ctx_ok = False

    #--- wind vector (downwind direction), for false-positive screening ---
    #A real plume should extend DOWNWIND (along this arrow). Perpendicular/opposite = suspect.
    #Prefer the wind actually used for Q (wind_uv from the pipeline); else fall back to the GEOS cache.
    wind = None
    if wind_uv is not None:
        u10v, uxv, uyv = float(wind_uv[0]), float(wind_uv[1]), float(wind_uv[2])
    else:
        wa = None
        wpath = os.path.join(psave, name + '_u_arr_bilinear.npy')
        if os.path.exists(wpath):
            wa = np.load(wpath)
        u10v, uxv, uyv = (float(wa[0]), float(wa[1]), float(wa[2])) if wa is not None else (0.0, 0.0, 0.0)
    wmag = np.hypot(uxv, uyv)
    if wmag > 1e-6:
        wind = (u10v, uxv / wmag, uyv / wmag)

    fig = Figure(figsize=(26, 9))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)
    panels = [('Esri satelite', 'sat'), ('OpenStreetMap', 'osm'), ('Radiancia EMIT (misma grilla)', 'rad')]

    for ax, (title, kind) in zip(axes, panels):
        if kind == 'rad':
            ax.imshow(rad_w, extent=ext, origin='upper', cmap='gray',
                      vmin=np.nanpercentile(rad_w, 2), vmax=np.nanpercentile(rad_w, 98), zorder=1)
        ax.imshow(rgba, extent=ext, origin='upper', zorder=2, interpolation='nearest')
        ax.set_xlim(lon_lo, lon_hi); ax.set_ylim(lat_lo, lat_hi)

        if kind in ('sat', 'osm') and ctx_ok:
            src = ctx.providers.Esri.WorldImagery if kind == 'sat' else ctx.providers.OpenStreetMap.Mapnik
            try:
                ctx.add_basemap(ax, crs='EPSG:4326', source=src, zorder=1, attribution_size=5)
            except Exception as e:
                print(f'  [map] basemap {kind} failed ({e}); flat background')
                ax.set_facecolor('#d0d8e0')
        elif kind in ('sat', 'osm'):
            ax.set_facecolor('#d0d8e0')

        ax.plot(lon_src, lat_src, '*', color='#ff3333', markersize=18,
                markeredgecolor='white', markeredgewidth=1.0, zorder=5)

        if wind is not None:
            u10v, ex, en = wind
            Ldeg = 0.16 * (lat_hi - lat_lo)                       # arrow length in latitude-degrees
            dlat = en * Ldeg
            dlon = ex * Ldeg / np.cos(np.radians(lat_src))        # correct for lon compression so it points true compass
            ax.annotate('', xy=(lon_src + dlon, lat_src + dlat), xytext=(lon_src, lat_src),
                        arrowprops=dict(arrowstyle='-|>', color='cyan', lw=2.5, mutation_scale=18), zorder=6)
            ax.text(lon_src + dlon * 1.18, lat_src + dlat * 1.18, f'viento\nu10={u10v:.1f} m/s',
                    color='cyan', fontsize=8, ha='center', va='center',
                    bbox=dict(facecolor='black', alpha=0.5, pad=2, edgecolor='none'), zorder=7)

        ax.set_title(title, fontsize=13)
        ax.set_xlabel('Longitud', fontsize=9); ax.set_ylabel('Latitud', fontsize=9)
        ax.tick_params(labelsize=7)

    sm = mcm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label(r'$\Delta$X' + f'{gas} (ppm·m) — physical enhancement', fontsize=12)

    sig_txt = f', sigma={sigma}' if sigma is not None else ''
    fig.suptitle(f'Pluma {gas.upper()} detectada — {int(mask.sum())} px{sig_txt}\n'
                 f'fuente: {lat_src:.4f}, {lon_src:.4f}   |   {name}', fontsize=14)

    out = os.path.join(psave, f'plume_on_map_{gas}.png')
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print('  [map] saved', out)
    return out


def main(path_img, name, psave, lat_src, lon_src, sigma, grow_on, pad_deg, gas='ch4'):
    #Standalone: rebuild the mask by growing from the given lat/lon, then render.
    from envi_files import load_hdr_file
    from georreferencing import location_and_time, gcp_from_placemark, lonlat_to_native_pixel
    from wavelet_den import compute_den
    from auto_plume import grow_plume_from_source

    ts, lat_c, lon_c, lat, lon = location_and_time(path_img, name, 'EMIT')
    list_gcp = gcp_from_placemark(path_img + 'placemark_' + name + '.placemark')

    show_b, _ = GAS_BANDS[gas]
    cube = load_hdr_file(psave, name + '_tool4')
    gas_enh = np.array(cube[:, :, show_b], dtype=float)
    rad = np.array(cube[:, :, 0], dtype=float)

    src_x, src_y = lonlat_to_native_pixel(lon_src, lat_src, lat, lon)
    print(f'Source lat/lon ({lat_src}, {lon_src}) -> native pixel (row={src_y}, col={src_x})')

    grow_map = compute_den(gas_enh) if grow_on == 'den' else gas_enh
    mask, status = grow_plume_from_source(grow_map, src_y, src_x, sigma_mult=sigma)
    if status != 'ok':
        print(f'Growth failed: {status}. Try a lower --sigma.')
        return
    print(f'Plume ({gas}): {int(mask.sum())} px  (grown on {grow_on}, sigma={sigma})')

    render_plume_map(mask, gas_enh, rad, lat, lon, list_gcp, lat_src, lon_src,
                     gas, psave, name, sigma=sigma, pad_deg=pad_deg)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Overlay a detected plume on a real basemap (satellite/OSM) and on radiance.')
    ap.add_argument('--lat', type=float, required=True, help='Source latitude (deg)')
    ap.add_argument('--lon', type=float, required=True, help='Source longitude (deg)')
    ap.add_argument('--gas', default='ch4', help='Gas to plot: ch4, co2, c2h4, c2h2, nh3')
    ap.add_argument('--sigma', type=float, default=2.0)
    ap.add_argument('--grow', choices=['den', 'physical'], default='physical', help='Map the mask is grown on (match what you accepted in the pipeline)')
    ap.add_argument('--pad', type=float, default=0.012, help='Zoom padding around the plume, in degrees')
    ap.add_argument('--name', default='EMIT_L1B_RAD_001_20260129T142036_2602909_023')
    ap.add_argument('--path-img', default='./', help='Folder containing the EMIT .nc and its .placemark')
    ap.add_argument('--psave', default='output/')
    a = ap.parse_args()

    gas = a.gas.strip().lower()
    if gas not in GAS_BANDS:
        raise SystemExit(f'Unknown/non-quantifiable gas: {gas}. Valid: {list(GAS_BANDS)}')

    main(a.path_img, a.name, a.psave, a.lat, a.lon, a.sigma, a.grow, a.pad, gas)
