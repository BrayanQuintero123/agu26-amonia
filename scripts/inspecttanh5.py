#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspecciona la estructura de un archivo HDF5 de Tanager (basic_radiance_hdf5)
sin cargar los datos pesados en memoria (solo shapes, dtypes y atributos).

Uso:
    python inspect_tanager_h5.py "/ruta/al/archivo_basic_radiance.h5"

Pega la salida completa en el chat.
"""

import sys
import h5py


def dump(name, obj, indent=0):
    pad = "  " * indent
    if isinstance(obj, h5py.Dataset):
        print(f"{pad}[DATASET] {name}  shape={obj.shape}  dtype={obj.dtype}")
    else:
        print(f"{pad}[GROUP]   {name}")

    # Atributos (aquí suelen vivir wavelengths, fwhm, sza, vza, fill_value, etc.)
    for k, v in obj.attrs.items():
        v_repr = v if not hasattr(v, "shape") or getattr(v, "size", 0) <= 10 else f"array shape={v.shape} dtype={v.dtype}"
        print(f"{pad}    attr: {k} = {v_repr}")


def walk(f):
    print("=" * 70)
    print("ATRIBUTOS GLOBALES (root)")
    print("=" * 70)
    for k, v in f.attrs.items():
        v_repr = v if not hasattr(v, "shape") or getattr(v, "size", 0) <= 10 else f"array shape={v.shape} dtype={v.dtype}"
        print(f"  attr: {k} = {v_repr}")

    print()
    print("=" * 70)
    print("ARBOL DE GRUPOS Y DATASETS")
    print("=" * 70)

    def visitor(name, obj):
        depth = name.count("/")
        dump(name, obj, indent=depth)

    f.visititems(visitor)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    path = sys.argv[1]
    with h5py.File(path, "r") as f:
        walk(f)