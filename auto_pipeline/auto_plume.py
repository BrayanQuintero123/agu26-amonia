#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Delineacion SEMI-AUTOMATICA de pluma (reemplazo de scripts/quant_func_v2.py::select_plume).

Filosofia acordada: el algoritmo delimita, el humano hace 'double-check'.
  1) El operador da UN clic en la fuente (indica CUAL pluma, no que pixeles).
  2) Crecimiento automatico: umbral (>k*sigma) + componentes conexas (skimage,
     flood-fill de adyacencia) y se conserva SOLO la isla que contiene el clic.
     (misma idea que scripts/Retrieval_methods.py::masking_plumes y que
      wavelet_plume_finder_public/emit_ime.py::build_data lineas 432-443).
  3) Double-check: se muestra la mascara y el operador la acepta / la rechaza /
     reajusta el umbral k. El umbral objetivo decide los pixeles -> reproducible.

Todo el nucleo de LARS (retrieval, IME, Ueff, conversion hidrostatica) se reutiliza
sin cambios; aqui solo cambia COMO se obtiene la mascara y la fuente del viento.
Los archivos originales quedan intactos.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(os.path.dirname(_HERE), 'scripts')
for _p in (_SCRIPTS, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.signal import medfilt2d
from skimage import measure

#---- reused, unchanged, from the original repo ----
from georreferencing import (location_and_time, georreference, gcp_from_placemark,
                             warp_array_for_display, warped_pixel_to_lonlat, lonlat_to_native_pixel)
from quant_func_v2 import extract_Q  # IME quantification core, untouched
#---- new wind (bilinear) ----
from wind_bilinear import wind_speed_bilinear
#---- wavelet anomaly (den) for display + geometry only ----
from wavelet_den import compute_den

DEFAULT_SIGMA_MULT = 2.0   # >2*sigma, consistent with masking_plumes / Roger et al. 2025
DEFAULT_AREA_MIN = 10      # minimum plume size in pixels (informational)


def grow_plume_from_source(ch4_map, src_row, src_col, sigma_mult=DEFAULT_SIGMA_MULT):
    """Grow a plume mask from the clicked source pixel.

    Threshold at sigma_mult*std over a 3x3 median-filtered map, label connected
    components (4-connectivity), and keep ONLY the component that contains
    (src_row, src_col). Returns (mask, status):
      status = 'ok'                   -> mask has the plume island
      status = 'source_below_threshold' -> source pixel is not above threshold; mask is None
    """
    filt = medfilt2d(ch4_map, kernel_size=3)
    std = np.nanstd(ch4_map)
    binary = (filt > sigma_mult * std) & (filt < 1e5)

    labels = measure.label(binary, connectivity=1)
    src_label = labels[src_row, src_col]

    if src_label == 0:
        return None, 'source_below_threshold'

    mask = (labels == src_label)
    return mask, 'ok'


def _draw_star(ax, wx, wy):
    ax.scatter(wx, wy, s=150, facecolor='white', edgecolor='r', linewidth=3, marker='*')


def select_plume_auto(plot_xch4, ref_rad, mission, gas, lat, lon, list_gcp,
                      sigma_mult=DEFAULT_SIGMA_MULT):
    """North-up display; source click grows the plume automatically; operator double-checks.
    Returns (mask_native, bool_det, source_coord) with the SAME contract as the original select_plume."""

    if mission == 'EMIT':
        pix_res = 60
    elif mission in ('EnMAP', 'PRISMA', 'GF5', 'ZY1'):
        pix_res = 30
    elif mission == 'AVIRIS-NG':
        pix_res = 4

    #--- wavelet anomaly: den is what we SHOW and what we GROW the mask on (geometry/detection).
    #    The physical enhancement (dxgas_quan) is quantified separately downstream, unchanged. ---
    print('Computing wavelet anomaly (den) for display/delineation...')
    den = compute_den(plot_xch4)  # raw den, used for growth; background ~= 0, sources highlighted

    #--- display: emit_ime overview recipe (inferno + vmax = p99 of POSITIVE den values) PLUS our
    #    3x3 median filter to kill single-pixel speckle (emit_ime does not median-filter, so its raw
    #    overview is actually noisier; the median filter is what makes ours read clean). ---
    den_disp = medfilt2d(den.astype(np.float32), kernel_size=3)
    _pos = den[den > 0]
    vminval_ret = 0
    vmaxval_ret = float(np.nanpercentile(_pos, 99)) if _pos.size else float(np.nanpercentile(den, 99))

    #The double-check preview shows the PHYSICAL enhancement, not den: consist_signals() inside the wavelet
    #flattens strong signals to the image max, so den has no internal gradient (plateau) and any plume renders
    #as a flat blob. The physical enhancement keeps the real core->edge gradient, and it is what gets quantified.
    #Same choice as emit_ime's plume panel (_draw_plume_panel).
    enh_disp = medfilt2d(np.asarray(plot_xch4, dtype=np.float32), kernel_size=3)

    rad = np.ones(ref_rad.shape) * np.nan
    rad[ref_rad > 0] = ref_rad[ref_rad > 0]
    mean_rad, std_rad = np.nanmean(rad), np.nanstd(rad)
    vminval_rad, vmaxval_rad = mean_rad - 3 * std_rad, mean_rad + 3 * std_rad

    #--- warp both to north-up for display; all clicks translated back to native ---
    rad_warp, gt = warp_array_for_display(list_gcp, rad)
    gas_warp, _ = warp_array_for_display(list_gcp, den_disp, ref_gt=gt, ref_shape=rad_warp.shape)   # den: selection panel
    enh_warp, _ = warp_array_for_display(list_gcp, enh_disp, ref_gt=gt, ref_shape=rad_warp.shape)   # physical: double-check panel

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True, constrained_layout=True)
    mappable1 = ax1.imshow(rad_warp, vmin=vminval_rad, vmax=vmaxval_rad, cmap='gray')
    mappable2 = ax2.imshow(gas_warp, vmin=vminval_ret, vmax=vmaxval_ret, cmap='inferno')

    fig.suptitle(
        '0) Check in your console whether you want to proceed.\n'
        '1) Zoom into plume area (magnifying glass). Press enter when done.\n'
        '2) Pinpoint the emission source (ONE click inside the plume).\n'
        '3) The plume is grown automatically; confirm in the console.',
        fontsize=18)

    cbar1 = fig.colorbar(mappable1, ax=ax1, orientation='horizontal', fraction=0.04, pad=0.02)
    cbar1.set_label(r'L (W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)', fontsize=20); cbar1.ax.tick_params(labelsize=14)
    cbar2 = fig.colorbar(mappable2, ax=ax2, orientation='horizontal', fraction=0.04, pad=0.02)
    cbar2.set_label(f'wavelet anomaly (den) — {gas}', fontsize=20); cbar2.ax.tick_params(labelsize=14)

    pix_res_display = abs(gt[5]) * 111320  # deg -> m, warped grid resolution
    ax1.add_artist(ScaleBar(pix_res_display, "m", length_fraction=0.25))
    ax2.add_artist(ScaleBar(pix_res_display, "m", length_fraction=0.25))
    ax1.axis('off'); ax2.axis('off')
    plt.rc('legend', fontsize=14)

    plt.show(block=False); plt.pause(0.1)
    ax1.set_autoscale_on(False); ax2.set_autoscale_on(False)

    #--- 0) proceed? ---
    while True:
        decision = input('Check the map. Do you still want to proceed? y/n: ')
        if decision in ('y', 'n'):
            break
        print('Not valid answer. Repeat again.')

    if decision == 'n':
        plt.close()
        return None, False, None, sigma_mult

    #--- 1) zoom + enter ---
    plt.show(block=False); plt.pause(0.1)
    while True:
        if plt.waitforbuttonpress():
            break

    #--- 2) one source click -> native pixel ---
    plt.draw()
    source = plt.ginput(n=1, timeout=0, show_clicks=True)
    source_wx, source_wy = source[0][0], source[0][1]
    source_lon, source_lat = warped_pixel_to_lonlat(source_wx, source_wy, gt)
    source_x, source_y = lonlat_to_native_pixel(source_lon, source_lat, lat, lon)
    source_coord = [source_x, source_y]
    _draw_star(ax1, source_wx, source_wy); _draw_star(ax2, source_wx, source_wy)
    plt.draw()

    #--- 3) automatic growth + double-check loop ---
    #Growth runs on the PHYSICAL enhancement, not on den: consist_signals() flattens strong den signals to a
    #plateau, so a threshold on den is quasi-binary and the sigma knob does nothing (786->781 px from 2 to 5 sigma).
    #The physical map has a real gradient, so sigma actually controls the mask, and the mask edge means something
    #physical ("2-sigma of the enhancement") -- consistent with what extract_Q then integrates.
    #den is still what you SEE (it locates the source better); the click only picks which anomaly.
    k = float(sigma_mult)
    mask = None
    bool_det = False
    while True:
        grown, status = grow_plume_from_source(plot_xch4, source_y, source_x, sigma_mult=k)

        if status == 'source_below_threshold':
            #You clicked where den is bright, but growth runs on the physical map, where that pixel may be weaker.
            print(f'The clicked pixel is below the {k:.2f}-sigma threshold of the PHYSICAL enhancement, so no plume grows there.')
            ans = input('Lower the threshold? enter a new sigma multiplier (e.g. 1.0), or "r" to reject: ')
            if ans.strip().lower() == 'r':
                bool_det = False; mask = None; break
            try:
                k = float(ans); continue
            except ValueError:
                print('Not a valid number.'); continue

        n_pix = int(grown.sum())
        #--- preview the grown plume north-up, cropped to its bounding box ---
        mask_warp, _ = warp_array_for_display(list_gcp, grown.astype(np.float32), ref_gt=gt, ref_shape=rad_warp.shape)
        mask_warp_bool = np.nan_to_num(mask_warp) > 0.5
        if mask_warp_bool.any():
            rw, cw = np.where(mask_warp_bool)
            pad = 15
            y0, ye = max(rw.min() - pad, 0), min(rw.max() + pad, mask_warp_bool.shape[0])
            x0, xe = max(cw.min() - pad, 0), min(cw.max() + pad, mask_warp_bool.shape[1])
            disp = np.where(mask_warp_bool, enh_warp, np.nan)  # physical enhancement inside the mask (what gets quantified)
            _vals_in = enh_warp[mask_warp_bool]; _vals_in = _vals_in[np.isfinite(_vals_in)]
            vmax_p = float(np.nanpercentile(_vals_in, 98)) if _vals_in.size else 1.0  # scale to the mask's own range (emit_ime recipe)
            figp, axp = plt.subplots(figsize=(10, 10))
            mp = axp.imshow(disp[y0:ye, x0:xe], vmin=0, vmax=max(vmax_p, 1e-6), cmap='inferno')
            figp.suptitle(f'Auto plume  (sigma={k:.2f}, {n_pix} px)\nCheck the console: accept / adjust / reject.', fontsize=16)
            cb = plt.colorbar(mp); cb.set_label(r'$\Delta$' + f'X{gas} (ppm·m) — physical', fontsize=15)
            axp.add_artist(ScaleBar(pix_res_display, "m", length_fraction=0.25)); axp.axis('off')
            plt.show(block=False); plt.pause(0.1)
        else:
            figp = None

        warn = '' if n_pix >= DEFAULT_AREA_MIN else f'  (WARNING: only {n_pix} px, below the {DEFAULT_AREA_MIN}-px minimum)'
        print(f'Grown plume: {n_pix} pixels at sigma={k:.2f}.{warn}')
        ans = input('Accept this plume? y = accept / a = adjust sigma / r = reject: ').strip().lower()
        if figp is not None:
            plt.close(figp)

        if ans == 'y':
            mask = grown; bool_det = True; break
        elif ans == 'a':
            newk = input('  New sigma multiplier (e.g. 2.5): ')
            try:
                k = float(newk)
            except ValueError:
                print('  Not a valid number; keeping current sigma.')
            continue
        elif ans == 'r':
            mask = None; bool_det = False; break
        else:
            print('Not a valid answer.'); continue

    plt.close(fig)
    return mask, bool_det, source_coord, k


def _safe_render_map(mask, gas_enh, ref_rad, lat, lon, list_gcp, lat_s, lon_s, gas, psave, name, sigma_used):
    #Auto-generate the plume-on-basemap PNG right after acceptance (no double work, correct path).
    #Lazy import breaks the auto_plume <-> plot_plume_map cycle; try/except so a map failure never
    #loses the already-computed Q.
    try:
        from plot_plume_map import render_plume_map
        render_plume_map(mask, gas_enh, ref_rad, lat, lon, list_gcp, lat_s, lon_s, gas, psave, name, sigma=sigma_used)
    except Exception as e:
        print(f'  [map] could not render plume map ({e}); the Q result is unaffected.')


def emission_quantification_auto(dxgas_show, dxgas_quan, ref_rad, mission, gas, path_folder, name, psave,
                                 sigma_mult=DEFAULT_SIGMA_MULT):
    """Copy of quant_func_v2.emission_quantification, but using select_plume_auto (semi-automatic
    delineation) and wind_speed_bilinear (bilinear GEOS-FP). extract_Q / ppmm_to_kg / georreference
    / Ueff are the ORIGINAL LARS functions, reused unchanged. On acceptance it also auto-renders the
    plume-on-basemap PNG (plot_plume_map.render_plume_map)."""

    ts, lat_c, lon_c, lat, lon = location_and_time(path_folder, name, mission)
    list_gcp = gcp_from_placemark(path_folder + 'placemark_' + name + '.placemark')

    mask, bool_det, source_coord, sigma_used = select_plume_auto(dxgas_show, ref_rad, mission, gas, lat, lon, list_gcp, sigma_mult)

    if gas in ('ch4', 'co2', 'c2h4', 'c2h2'):
        if bool_det:
            lat_s, lon_s = georreference(path_folder, name, psave, name + '_tool4', gas, mask, source_coord)
            if mission in ('EMIT', 'PRISMA', 'AVIRIS-NG'):
                lat_s, lon_s = lat[source_coord[1], source_coord[0]], lon[source_coord[1], source_coord[0]]
            u10 = wind_speed_bilinear(ts, lat_s, lon_s, path_folder, psave, name)
            Q, err_Q, u10, err_u10 = extract_Q(dxgas_quan, mask, u10, mission, gas)
            _safe_render_map(mask, dxgas_show, ref_rad, lat, lon, list_gcp, lat_s, lon_s, gas, psave, name, sigma_used)
        else:
            Q, err_Q, u10, err_u10, lat_s, lon_s = None, None, None, None, None, None
            print('No detection')
        return Q, err_Q, u10, err_u10, lat_s, lon_s, ts, bool_det

    elif gas == 'nh3':
        if bool_det:
            lat_s, lon_s = georreference(path_folder, name, psave, name + '_tool4', gas, mask, source_coord)
            if mission in ('EMIT', 'PRISMA', 'AVIRIS-NG'):
                lat_s, lon_s = lat[source_coord[1], source_coord[0]], lon[source_coord[1], source_coord[0]]
            u10 = wind_speed_bilinear(ts, lat_s, lon_s, path_folder, psave, name)
            Q_1, err_Q_1, Q_2, err_Q_2, u10, err_u10 = extract_Q(dxgas_quan, mask, u10, mission, gas)
            _safe_render_map(mask, dxgas_show, ref_rad, lat, lon, list_gcp, lat_s, lon_s, gas, psave, name, sigma_used)
        else:
            Q_1, err_Q_1, Q_2, err_Q_2, u10, err_u10, lat_s, lon_s = None, None, None, None, None, None, None, None
            print('No detection')
        return Q_1, err_Q_1, Q_2, err_Q_2, u10, err_u10, lat_s, lon_s, ts, bool_det

    else:
        if bool_det:
            lat_s, lon_s = georreference(path_folder, name, psave, name + '_tool4', gas, mask, source_coord)
            if mission in ('EMIT', 'PRISMA', 'AVIRIS-NG'):
                lat_s, lon_s = lat[source_coord[1], source_coord[0]], lon[source_coord[1], source_coord[0]]
            u10 = wind_speed_bilinear(ts, lat_s, lon_s, path_folder, psave, name)
        else:
            u10, lat_s, lon_s = None, None, None
        return u10, lat_s, lon_s, ts, bool_det
