#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Punto de entrada unico para cuantificar CH4 (u otro gas) en escenas Tanager.

Elige solo el camino segun lo que traiga la escena:

  CON GT   -> existe <scene>_ql_ch4_json.geojson CON features. Las plumas se
              siembran de ahi (fuente, viento y emision oficiales), la mascara
              sale de la huella del quicklook oficial, y se cuantifica con los
              DOS matched filters para compararlos entre si y contra el oficial.
              No hay ningun clic.  -> compare_mf.py

  SIN GT   -> no hay GeoJSON, o lo hay pero vacio. Se corren los dos MF, se
              proponen candidatos (union de ambos) sobre el mapa wavelet, y tu
              eliges con un clic cuales son plumas.  -> pick_plumes_wavelet.py

Con varias escenas de golpe, cada una toma su propio camino. Las que necesiten
clics se dejan para el final, para que la tanda automatica corra sin pararse.

Convencion de cuantificacion: por defecto la de Carbon Mapper (Ueff = u10,
L = sqrt(area)), que es la unica que hace los numeros comparables con el
producto oficial. Con --ueff-mode lars se vuelve a la parametrizacion del repo.

Uso:
  python auto_pipeline/quantify_tanager.py tanager/*_ortho_radiance_hdf5.h5
  python auto_pipeline/quantify_tanager.py <.h5> --mode pick        # forzar el picker
  python auto_pipeline/quantify_tanager.py <.h5> --mode gt          # forzar el automatico
  python auto_pipeline/quantify_tanager.py <.h5> --ueff-mode lars --l-mode hull
"""

import os
import sys
import glob
import json
import argparse
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(os.path.dirname(_HERE), 'scripts')
for _p in (_SCRIPTS, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from plumes_from_json import find_plumes_json


def has_ground_truth(rad_file):
    """True solo si hay un GeoJSON del MISMO scene id y ademas trae plumas.
    Un GeoJSON vacio (Planet lo entrega asi cuando no detecto nada en ese tile)
    cuenta como SIN GT: hay que ir al picker."""
    js = find_plumes_json(rad_file)
    if not js or not os.path.exists(js):
        return False, None, 'no hay GeoJSON para este scene id'
    try:
        with open(js, 'r', encoding='utf-8') as f:
            feats = json.load(f).get('features', [])
    except Exception as e:
        return False, js, f'GeoJSON ilegible ({e})'
    if not feats:
        return False, js, 'el GeoJSON existe pero viene vacio'
    return True, js, f'{len(feats)} pluma(s) oficiales'


def has_quicklook(rad_file):
    p = os.path.dirname(os.path.abspath(rad_file))
    scene = os.path.basename(rad_file).split('_ortho_')[0].split('_basic_')[0]
    ql = os.path.join(p, f'{scene}_ortho_ql_ch4.tif')
    return os.path.exists(ql), ql


def run(cmd):
    print('\n$ ' + ' '.join(cmd), flush=True)
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser(
        description='Cuantifica plumas Tanager con los dos MF, eligiendo solo entre el camino automatico (con GT) y el picker interactivo (sin GT).')
    ap.add_argument('rad_files', nargs='+', help='Uno o varios .h5 ortho_radiance de Tanager.')
    ap.add_argument('--gas', default='ch4')
    ap.add_argument('--mode', choices=['auto', 'gt', 'pick'], default='auto',
                    help="'auto' (por defecto) decide segun haya GeoJSON con plumas; 'gt' y 'pick' fuerzan un camino.")
    ap.add_argument('-o', '--output-dir', default=None)
    ap.add_argument('--ueff-mode', choices=['cm', 'lars'], default='cm')
    ap.add_argument('--l-mode', choices=['sqrt-area', 'fetch', 'hull'], default='sqrt-area')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--dry-run', action='store_true', help='Solo dice que camino tomaria cada escena.')
    #se pasan tal cual al picker
    ap.add_argument('--sigma', type=float, default=2.0)
    ap.add_argument('--min-pix', type=int, default=10)
    ap.add_argument('--grow-on', choices=['lars', 'adv'], default='lars')
    ap.add_argument('--wind', choices=['geos', 'era5'], default='era5')
    ap.add_argument('--wind-value', type=float, default=None)
    args = ap.parse_args()

    files = []
    for pat in args.rad_files:
        files.extend(sorted(glob.glob(pat)) or [pat])

    gt_files, pick_files = [], []
    print('=' * 78)
    for f in files:
        if not os.path.exists(f):
            print(f'  [!] {os.path.basename(f)}: no existe, lo salto'); continue
        ok_ql, ql = has_quicklook(f)
        gt, js, why = has_ground_truth(f)
        if args.mode == 'gt':
            gt = True
        elif args.mode == 'pick':
            gt = False
        #El camino automatico necesita la huella del quicklook para la mascara.
        if gt and not ok_ql:
            print(f'  [!] {os.path.basename(f)}: hay GT pero falta {os.path.basename(ql)}; va al picker')
            gt = False
        (gt_files if gt else pick_files).append(f)
        print(f'  {os.path.basename(f):<52} -> {"GT (automatico)" if gt else "picker (clics)":<18} [{why}]')
    print('=' * 78)

    if args.dry_run:
        return

    rc = 0
    if gt_files:
        cmd = [sys.executable, os.path.join(_HERE, 'compare_mf.py'), *gt_files,
               '--gas', args.gas, '--ueff-mode', args.ueff_mode, '--l-mode', args.l_mode]
        if args.output_dir:
            cmd += ['-o', args.output_dir]
        if args.force:
            cmd += ['--force']
        rc |= run(cmd)

    #Las interactivas al final: asi la tanda automatica no se queda esperando un clic.
    for f in pick_files:
        cmd = [sys.executable, os.path.join(_HERE, 'pick_plumes_wavelet.py'), f,
               '--gas', args.gas, '--ueff-mode', args.ueff_mode,
               '--l-mode', ('hull' if args.l_mode == 'hull' else 'sqrt-area'),
               '--sigma', str(args.sigma), '--min-pix', str(args.min_pix),
               '--grow-on', args.grow_on]
        if args.wind_value is not None:
            cmd += ['--wind-value', str(args.wind_value)]
        else:
            cmd += ['--wind', args.wind]
        if args.output_dir:
            cmd += ['-o', args.output_dir]
        if args.force:
            cmd += ['--force']
        rc |= run(cmd)

    return rc


if __name__ == '__main__':
    raise SystemExit(main() or 0)
