# -*- coding: utf-8 -*-
"""
Tanager_reader.py

Soporta dos productos:
  - basic_radiance_hdf5  : geometria de sensor (SWATHS/HYP), con Lat/Lon embebidos
  - ortho_radiance_hdf5  : orto-rectificado   (GRIDS/HYP),  geotransform en StructMetadata

Auto-detecta cual es segun la estructura interna del .h5.

Firma comun:
    img, wvl, fwhm, sza, vza = read_Tanager(p, n)
    lat, lon                  = read_Tanager_latlon(p, n)   # solo basic
    geo                       = get_ortho_framing(p, n)     # ambos
"""

import os, re
import numpy as np
import h5py
import pyproj

_SWATH_DATA = 'HDFEOS/SWATHS/HYP/Data Fields'
_SWATH_GEO  = 'HDFEOS/SWATHS/HYP/Geolocation Fields'
_GRID_DATA  = 'HDFEOS/GRIDS/HYP/Data Fields'


def _resolve_path(p, n):
    candidate = os.path.join(p, n + '.h5')
    if os.path.exists(candidate):
        return candidate
    candidate2 = os.path.join(p, n)
    if os.path.exists(candidate2):
        return candidate2
    raise FileNotFoundError(
        f"No se encontro el .h5: '{candidate}' ni '{candidate2}'"
    )


def _detect_data_path(f):
    """Devuelve la ruta al grupo Data Fields y si es swath (basic) o grid (ortho)."""
    if _SWATH_DATA in f:
        return _SWATH_DATA, 'swath'
    if _GRID_DATA in f:
        return _GRID_DATA, 'grid'
    raise KeyError("No se encontro ni SWATHS/HYP ni GRIDS/HYP en el HDF5")


def _read_masked(dset):
    arr = dset[:].astype(np.float32)
    fill = dset.attrs.get('_FillValue', None)
    if fill is not None:
        arr[arr == float(fill)] = np.nan
    return arr


def _parse_struct_metadata(f):
    """Lee geotransform y EPSG del StructMetadata (usado por ortho y por basic para EPSG)."""
    raw = f['HDFEOS INFORMATION/StructMetadata.0'][()].decode()
    ul   = re.search(r'UpperLeftPointMtrs=\(([^)]+)\)', raw)
    lr   = re.search(r'LowerRightMtrs=\(([^)]+)\)', raw)
    xdim = re.search(r'XDim=(\d+)', raw)
    ydim = re.search(r'YDim=(\d+)', raw)
    zone = re.search(r'ZoneCode=(\d+)', raw)
    if not all([ul, lr, xdim, ydim, zone]):
        return None
    ulx, uly = [float(v) for v in ul.group(1).split(',')]
    lrx, lry = [float(v) for v in lr.group(1).split(',')]
    cols = int(xdim.group(1))
    rows = int(ydim.group(1))
    px = (lrx - ulx) / cols
    py = (lry - uly) / rows   # negativo
    epsg = 32600 + int(zone.group(1))
    return dict(geotransform=[ulx, px, 0.0, uly, 0.0, py],
                epsg_code=epsg, rows=rows, cols=cols)


# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------

def read_Tanager(p, n):
    """
    Lee radiancia TOA y angulos de cualquier producto Tanager HDF5.

    Returns
    -------
    img        : (M, N, B) float32  — NaN donde hay fill-value
    wvl        : (B,) float32  nm
    fwhm       : (B,) float32  nm
    sza        : float  grados
    vza        : float  grados
    """
    fpath = _resolve_path(p, n)
    with h5py.File(fpath, 'r') as f:
        data_path, _ = _detect_data_path(f)
        rad_ds = f[f'{data_path}/toa_radiance']

        rad = rad_ds[:].astype(np.float32)          # (B, M, N)
        fill_rad = rad_ds.attrs.get('_FillValue', None)
        img = np.transpose(rad, (1, 2, 0))           # (M, N, B)
        if fill_rad is not None:
            img[img == float(fill_rad)] = np.nan

        wvl  = np.asarray(rad_ds.attrs['wavelengths'], dtype=np.float32)
        fwhm = np.asarray(rad_ds.attrs['fwhm'],        dtype=np.float32)

        sza = float(np.nanmean(_read_masked(f[f'{data_path}/sun_zenith'])))
        vza = float(np.nanmean(_read_masked(f[f'{data_path}/sensor_zenith'])))

    return img, wvl, fwhm, sza, vza


def read_Tanager_latlon(p, n):
    """
    Devuelve (lat, lon) arrays (M, N) del producto basic (SWATH).
    Para el producto ortho usa get_ortho_framing() en su lugar.
    """
    fpath = _resolve_path(p, n)
    with h5py.File(fpath, 'r') as f:
        data_path, kind = _detect_data_path(f)
        if kind != 'swath':
            raise ValueError("read_Tanager_latlon solo aplica al producto basic (SWATH). "
                             "Para ortho usa get_ortho_framing().")
        lat = f[f'{_SWATH_GEO}/Latitude'][:]
        lon = f[f'{_SWATH_GEO}/Longitude'][:]
    return lat.astype(np.float32), lon.astype(np.float32)


def read_Tanager_masks(p, n):
    """Devuelve dict de mascaras booleanas (True = pixel invalido)."""
    fpath = _resolve_path(p, n)
    with h5py.File(fpath, 'r') as f:
        data_path, _ = _detect_data_path(f)
        cloud  = f[f'{data_path}/beta_cloud_mask'][:]
        cirrus = f[f'{data_path}/beta_cirrus_mask'][:]
        nodata = f[f'{data_path}/nodata_pixels'][:]
    return dict(cloud=cloud.astype(bool),
                cirrus=cirrus.astype(bool),
                nodata=nodata.astype(bool))


def get_ortho_framing(p, n, px_m=30.0):
    """
    Devuelve dict con geotransform GDAL, epsg_code, rows, cols.
    - Para ortho (GRIDS): lee StructMetadata directamente.
    - Para basic (SWATHS): deriva la malla ortho desde Lat/Lon + EPSG del strip_id
      usando pixel size px_m (defecto 30 m, igual que el producto ortho de Tanager).
    """
    fpath = _resolve_path(p, n)
    with h5py.File(fpath, 'r') as f:
        _, kind = _detect_data_path(f)
        if kind == 'grid':
            return _parse_struct_metadata(f)
        # --- basic: derivar desde Lat/Lon ---
        lat = f[f'{_SWATH_GEO}/Latitude'][:]
        lon = f[f'{_SWATH_GEO}/Longitude'][:]
        # EPSG: UTM zona N/S segun longitud media y latitud media
        lon_mean = float(np.nanmean(lon))
        lat_mean = float(np.nanmean(lat))
        zone = int((lon_mean + 180) / 6) + 1
        epsg = 32600 + zone if lat_mean >= 0 else 32700 + zone

    import pyproj
    tr = pyproj.Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True)
    xs, ys = tr.transform(lon.ravel(), lat.ravel())
    ok = np.isfinite(xs) & np.isfinite(ys)
    xmin, xmax = xs[ok].min(), xs[ok].max()
    ymin, ymax = ys[ok].min(), ys[ok].max()
    ulx = xmin - px_m / 2
    uly = ymax + px_m / 2
    cols = int(np.ceil((xmax - xmin) / px_m)) + 1
    rows = int(np.ceil((ymax - ymin) / px_m)) + 1
    return dict(geotransform=[ulx, px_m, 0.0, uly, 0.0, -px_m],
                epsg_code=epsg, rows=rows, cols=cols)


def Tanager_loc_time_gcps(path_folder, name):
    """
    Equivalente a EMIT_loc_time_gcps para el pipeline auto.
    - Extrae timestamp del nombre: YYYYMMDD_HHMMSS_... -> YYYYMMDDHHmmss
    - Calcula grilla lat/lon desde geotransform (ortho) o arrays SWATH (basic)
    - Escribe el .placemark para gcp_from_placemark
    - Devuelve (time, lat_c, lon_c, lat, lon)
    """
    # --- timestamp ---
    parts = name.split('_')
    time = parts[0] + parts[1]   # YYYYMMDD + HHMMSS

    # --- lat/lon ---
    geo = get_ortho_framing(path_folder, name)
    gt, epsg = geo['geotransform'], geo['epsg_code']
    rows_n, cols_n = geo['rows'], geo['cols']

    fpath = _resolve_path(path_folder, name)
    with h5py.File(fpath, 'r') as f:
        _, kind = _detect_data_path(f)
        if kind == 'swath':
            lat_full = f[f'{_SWATH_GEO}/Latitude'][:].astype(np.float32)
            lon_full = f[f'{_SWATH_GEO}/Longitude'][:].astype(np.float32)
        else:
            # ortho: construir grid desde geotransform
            tr = pyproj.Transformer.from_crs(f'EPSG:{epsg}', 'EPSG:4326', always_xy=True)
            gx = gt[0] + (np.arange(cols_n) + 0.5) * gt[1]
            gy = gt[3] + (np.arange(rows_n) + 0.5) * gt[5]
            GX, GY = np.meshgrid(gx, gy)
            lons, lats = tr.transform(GX.ravel(), GY.ravel())
            lat_full = lats.reshape(rows_n, cols_n).astype(np.float32)
            lon_full = lons.reshape(rows_n, cols_n).astype(np.float32)

    lat_c = float(np.nanmean(lat_full))
    lon_c = float(np.nanmean(lon_full))

    # --- .placemark (GCPs cada ~100 px, mismo formato que EMIT) ---
    placemark_path = os.path.join(path_folder, f'placemark_{name}.placemark')
    arr_i = np.arange(0, lat_full.shape[0], 100)
    arr_j = np.arange(0, lat_full.shape[1], 100)
    with open(placemark_path, 'w') as f:
        f.write('<?xml version="1.0" encoding="ISO-8859-1"?>\n<Placemarks>\n')
        cont = 0
        for i in arr_i:
            for j in arr_j:
                la, lo = lat_full[i, j], lon_full[i, j]
                if not (np.isfinite(la) and np.isfinite(lo)):
                    continue
                cont += 1
                f.write(
                    f'<Placemark name="GCP_{cont}">'
                    f'<LATITUDE>{la:.6f}</LATITUDE>'
                    f'<LONGITUDE>{lo:.6f}</LONGITUDE>'
                    f'<PIXEL_X>{j}</PIXEL_X>'
                    f'<PIXEL_Y>{i}</PIXEL_Y>'
                    f'</Placemark>\n'
                )
        f.write('</Placemarks>\n')

    return time, lat_c, lon_c, lat_full, lon_full
