import numpy as np
from osgeo import gdal, osr
import rasterio
import matplotlib.pyplot as plt

from EMIT_reader import read_EMIT
from georreferencing import location_and_time


# ---------------------------------------------------------
# 1) Ortorectificación del cubo completo
# ---------------------------------------------------------

def orthorectify_cube(img, lat, lon, out_tif, gcp_step=50, resample="near"):
    """
    Ortorectifica un cubo hiperespectral completo (M,N,B) a una grilla lat/lon regular.
    """
    M, N, B = img.shape
    print(f"Cubo: {M}x{N}, {B} bandas")

    # GCPs
    gcps = []
    for i in range(0, M, gcp_step):
        for j in range(0, N, gcp_step):
            gcps.append(gdal.GCP(float(lon[i, j]), float(lat[i, j]), 0, float(j), float(i)))
    print(f"{len(gcps)} GCPs generados")

    # Raster en memoria con todas las bandas
    driver = gdal.GetDriverByName("MEM")
    ds = driver.Create("", N, M, B, gdal.GDT_Float32)

    data = np.nan_to_num(img, nan=-9999).astype(np.float32)
    for b in range(B):
        band = ds.GetRasterBand(b + 1)
        band.WriteArray(data[:, :, b])
        band.SetNoDataValue(-9999)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetGCPs(gcps, srs.ExportToWkt())

    print("Ortorectificando...")
    gdal.Warp(out_tif, ds, dstSRS="EPSG:4326", resampleAlg=resample,
              format="GTiff", dstNodata=-9999,
              creationOptions=["COMPRESS=LZW", "BIGTIFF=YES"])

    ds = None
    print(f"Guardado: {out_tif}")


# ---------------------------------------------------------
# 2) Plot de una sola banda
# ---------------------------------------------------------

def plot_ortho_band(path, band=100, cmap='gray', percentile=(2, 98)):
    with rasterio.open(path) as src:
        print(f"Bandas: {src.count}, tamaño: {src.width}x{src.height}")
        data = src.read(band).astype(float)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    valid = data[~np.isnan(data)]
    vmin, vmax = np.percentile(valid, percentile)

    plt.figure(figsize=(8, 10))
    im = plt.imshow(data, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label='Radiancia')
    plt.xlabel('Longitud'); plt.ylabel('Latitud')
    plt.title(f'Banda {band} (ortorectificada)')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# 3) Plot RGB (o falso color)
# ---------------------------------------------------------

def band_from_wvl(wvl, target_nm):
    """Encuentra el índice de banda más cercano a una longitud de onda dada"""
    return np.abs(wvl - target_nm).argmin()

def plot_ortho_rgb(path, wvl, r_nm=650, g_nm=560, b_nm=470, percentile=(2, 98)):
    r_idx = band_from_wvl(wvl, r_nm)
    g_idx = band_from_wvl(wvl, g_nm)
    b_idx = band_from_wvl(wvl, b_nm)
    print(f"R: banda {r_idx} ({wvl[r_idx]:.0f} nm), G: banda {g_idx} ({wvl[g_idx]:.0f} nm), B: banda {b_idx} ({wvl[b_idx]:.0f} nm)")

    with rasterio.open(path) as src:
        r = src.read(r_idx + 1).astype(float)  # rasterio es 1-indexed
        g = src.read(g_idx + 1).astype(float)
        b = src.read(b_idx + 1).astype(float)
        if src.nodata is not None:
            r[r == src.nodata] = np.nan
            g[g == src.nodata] = np.nan
            b[b == src.nodata] = np.nan
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    def stretch(channel):
        valid = channel[~np.isnan(channel)]
        lo, hi = np.percentile(valid, percentile)
        return np.clip((channel - lo) / (hi - lo), 0, 1)

    rgb = np.dstack([stretch(r), stretch(g), stretch(b)])
    rgb = np.nan_to_num(rgb, nan=0)

    plt.figure(figsize=(8, 10))
    plt.imshow(rgb, extent=extent)
    plt.xlabel('Longitud'); plt.ylabel('Latitud')
    plt.title(f'RGB compuesto ({r_nm}/{g_nm}/{b_nm} nm)')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# 4) Uso
# ---------------------------------------------------------

if __name__ == "__main__":

    p = "C:/Users/jefer/OneDrive/Documentos/GitHub/HS_tool/"   # con / no \
    n = "EMIT_L1B_RAD_001_20250218T073025_2504905_004"
    out_tif = "radiancia_cubo_ortho.tif"

    # Leer radiancia + lat/lon por pixel
    img, wvl, fwhm, sza, vza = read_EMIT(p, n)
    ts, lat_c, lon_c, lat, lon = location_and_time(p, n, mission="EMIT")

    # Ortorectificar (esto puede tardar unos minutos)
    orthorectify_cube(img, lat, lon, out_tif, gcp_step=50)

    # Plot banda única
    plot_ortho_band(out_tif, band=100)

    # Plot RGB "real" (visible)
    plot_ortho_rgb(out_tif, wvl, r_nm=650, g_nm=560, b_nm=470)

    # Plot falso color SWIR (opcional, descomenta si quieres verlo)
    # plot_ortho_rgb(out_tif, wvl, r_nm=2300, g_nm=1650, b_nm=850)