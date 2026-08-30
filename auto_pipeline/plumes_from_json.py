#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Siembra automatica de plumas a partir del asset `ql_ch4_json` de Tanager.

El GeoJSON oficial (Carbon Mapper) trae, por pluma detectada en la escena, un
Point con la FUENTE y en properties: emission (kg/h), ime (kg), wind_speed_avg,
wind_direction_avg y plume_quality. Eso es exactamente lo que el modo interactivo
le pide al operador con un clic, asi que aqui lo leemos y crecemos la mascara
sola: no hay clics y se cuantifican TODAS las plumas de la escena en una corrida.

La mascara se hace igual que en auto_plume.grow_plume_from_source (umbral
k*sigma sobre el MF filtrado 3x3 + componente conexa que contiene la fuente),
de modo que el criterio de borde sigue siendo el mismo y reproducible.

Se usa desde pipeline_tanager.py con --auto-json; no modifica nada del flujo
interactivo.
"""

import json
import os
import glob

import numpy as np
import pyproj
from scipy.signal import medfilt2d
from skimage import measure

DEFAULT_SIGMA = 2.0
SIGMA_LADDER = (2.0, 1.5, 1.0, 0.75, 0.5)   # se prueba en orden hasta que la fuente supere el umbral
SNAP_RADIUS_PX = 3                           # la fuente reportada puede caer 1-2 px fuera del realce
MIN_PIX = 5                                  # por debajo de esto la mascara no es una pluma


def find_plumes_json(rad_file):
    """Busca el *_ql_ch4_json.geojson junto al .h5 de radiancia (mismo scene id)."""
    d = os.path.dirname(os.path.abspath(rad_file))
    scene = os.path.basename(rad_file).split('_ortho_')[0].split('_basic_')[0]
    #Solo se acepta un GeoJSON del MISMO scene id. Antes habia un fallback a
    #'*ql_ch4*.geojson' y eso, en una carpeta con varias escenas, cargaba en
    #silencio las plumas de OTRA escena (o un GeoJSON vacio) sin avisar.
    hits = sorted(glob.glob(os.path.join(d, f'{scene}*ql_ch4*.geojson')))
    return hits[0] if hits else None


def read_plumes_json(json_file, gas='ch4'):
    """Devuelve la lista de plumas del GeoJSON como dicts planos (lon/lat + metadata oficial)."""
    with open(json_file, 'r', encoding='utf-8') as f:
        gj = json.load(f)

    out = []
    for feat in gj.get('features', []):
        geom = feat.get('geometry') or {}
        if geom.get('type') != 'Point':
            continue                      # solo sembramos desde el punto-fuente
        lon, lat = geom['coordinates'][:2]
        pr = feat.get('properties', {})
        pid = str(pr.get('plume_id', ''))
        out.append(dict(
            plume_id=pid,
            label=pid.split('_')[-1] if pid else '?',
            lon=float(lon), lat=float(lat),
            quality=pr.get('plume_quality', 'unknown'),
            q_official=pr.get('emission'),
            q_official_err=pr.get('emission_uncertainty'),
            ime_official=pr.get('ime'),
            fetch_official=pr.get('fetch'),      # la L que usa Carbon Mapper en Q=IME*3600*Ueff/L
            u10_official=pr.get('wind_speed_avg'),
            wind_from_official=pr.get('wind_direction_avg'),
            wind_source_official=pr.get('wind_source'),
            datetime=pr.get('datetime'),
        ))
    return out


def lonlat_to_pixel(lon, lat, gt, epsg, shape):
    """(lon,lat) -> (row, col) en la malla ortorrectificada; None si cae fuera."""
    tr = pyproj.Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True)
    x, y = tr.transform(lon, lat)
    col = int(np.floor((x - gt[0]) / gt[1]))
    row = int(np.floor((y - gt[3]) / gt[5]))
    if not (0 <= row < shape[0] and 0 <= col < shape[1]):
        return None
    return row, col


def _snap_to_local_max(filt, row, col, radius=SNAP_RADIUS_PX):
    """La fuente reportada es el origen de la emision, no el pixel mas realzado:
    se mueve al maximo local dentro de +/-radius para que el crecimiento agarre."""
    r0, r1 = max(row - radius, 0), min(row + radius + 1, filt.shape[0])
    c0, c1 = max(col - radius, 0), min(col + radius + 1, filt.shape[1])
    win = filt[r0:r1, c0:c1]
    if not np.isfinite(win).any():
        return row, col
    dr, dc = np.unravel_index(np.nanargmax(win), win.shape)
    return r0 + int(dr), c0 + int(dc)


def grow_mask(data, row, col, sigma_ladder=SIGMA_LADDER, min_pix=MIN_PIX):
    """Umbral k*sigma + componente conexa que contiene la fuente, bajando k hasta
    que la fuente quede por encima del umbral. Devuelve (mask, k_usado, status)."""
    filt = medfilt2d(np.nan_to_num(np.asarray(data, dtype=np.float32)), kernel_size=3)
    std = float(np.nanstd(data))
    row, col = _snap_to_local_max(filt, row, col)

    for k in sigma_ladder:
        binary = (filt > k * std) & (filt < 1e5)
        labels = measure.label(binary, connectivity=1)
        lab = labels[row, col]
        if lab == 0:
            continue
        mask = (labels == lab)
        if int(mask.sum()) < min_pix:
            continue
        return mask, k, 'ok'

    return None, sigma_ladder[-1], 'source_below_threshold'


def seed_masks(data, plumes, gt, epsg, sigma_ladder=SIGMA_LADDER, min_pix=MIN_PIX,
               qualities=None):
    """Crece una mascara por pluma del GeoJSON.

    qualities: set de plume_quality a conservar (None = todas).
    Dos fuentes vecinas pueden caer en la MISMA componente conexa; en ese caso la
    segunda se marca 'merged_with' y no se cuantifica dos veces la misma masa.
    """
    kept, skipped = [], []
    for pl in plumes:
        pl = dict(pl)
        if qualities and pl['quality'] not in qualities:
            pl['status'] = 'skipped_quality'
            skipped.append(pl); continue

        rc = lonlat_to_pixel(pl['lon'], pl['lat'], gt, epsg, data.shape)
        if rc is None:
            pl['status'] = 'outside_scene'
            skipped.append(pl); continue
        row, col = rc

        mask, k, status = grow_mask(data, row, col, sigma_ladder, min_pix)
        if status != 'ok':
            pl['status'] = status
            skipped.append(pl); continue

        dup = next((q for q in kept if np.array_equal(q['mask'], mask)), None)
        if dup is not None:
            pl['status'] = 'merged_with_' + dup['label']
            skipped.append(pl); continue

        pl.update(mask=mask, sigma=k, row=row, col=col,
                  n_pix=int(mask.sum()), status='ok')
        kept.append(pl)

    return kept, skipped
