import numpy as np
import rasterio
from rasterio.warp import transform
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.path
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
import pyproj

from wind_v2 import wind_speed_manual

# ---------------------------------------------------------
# 1) Cargar el TIF de Tanager
# ---------------------------------------------------------

def load_tanager_tif(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(float)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan

        pix_w = abs(src.transform.a)
        pix_h = abs(src.transform.e)
        pix_area_m2 = pix_w * pix_h
        pix_area_cm2 = pix_area_m2 * 1e4  # m2 -> cm2
        pix_res_m = np.sqrt(pix_area_m2)

        rows, cols = np.indices(data.shape)
        xs, ys = rasterio.transform.xy(src.transform, rows.ravel(), cols.ravel())
        lon, lat = transform(src.crs, "EPSG:4326", xs, ys)
        lon = np.array(lon).reshape(data.shape)
        lat = np.array(lat).reshape(data.shape)

        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        crs_src = src.crs

    print(f"Pixel: {pix_w:.1f} x {pix_h:.1f} m ({pix_area_cm2:.2e} cm2)")
    return data, lat, lon, extent, pix_area_cm2, pix_res_m, crs_src


def parse_timestamp_from_name(name):
    date_part = name[3:11]       # YYYYMMDD
    time_part = name[12:18]      # HHMMSS
    return date_part + time_part


# ---------------------------------------------------------
# 2) Selección manual de la pluma (fuente + polígono)
# ---------------------------------------------------------

def select_plume_tanager(data, extent, percentile_max=99.5, cmap='inferno'):
    valid = data[~np.isnan(data)]
    vmin, vmax = 0, np.percentile(valid, percentile_max)  # <- percentil en vez de std_mult

    fig, ax = plt.subplots(figsize=(10, 12), facecolor='black')
    ax.set_facecolor('black')
    im = ax.imshow(data, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, label=r'$\Delta$XNH$_3$ (molec/cm$^2$)')
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(colors='white')
    fig.suptitle('1) Zoom (opcional), Enter para continuar\n'
                  '2) Un click en la fuente\n'
                  '3) Delinea la pluma con clicks, Enter para cerrar',
                  color='white')
    plt.show(block=False)
    plt.pause(0.5)
    ax.set_autoscale_on(False)

    # Zoom opcional
    while True:
        if plt.waitforbuttonpress():
            break

    plt.pause(0.5)

    # Fuente
    source = plt.ginput(n=1, timeout=0, show_clicks=True)
    source_x, source_y = source[0]
    ax.scatter(source_x, source_y, s=150, facecolor='white', edgecolor='r',
               linewidth=3, marker='*')
    plt.draw()
    plt.pause(0.5)

    # Polígono de la pluma
    polygon = plt.ginput(n=-1, timeout=0, show_clicks=True,
                          mouse_add=1, mouse_pop=3, mouse_stop=2)
    plt.close()

    ny, nx = data.shape
    x0, x1, y0, y1 = extent
    xv, yv = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y1, y0, ny))
    points = np.column_stack((xv.ravel(), yv.ravel()))
    path = matplotlib.path.Path(polygon)
    mask = path.contains_points(points).reshape(data.shape)

    return mask, (source_x, source_y)


# ---------------------------------------------------------
# 3) Plot final tipo paper: zoom a la pluma, fondo negro, scalebar
# ---------------------------------------------------------

def zoom_bbox_from_mask(mask, extent, data_shape, margin_px=150):
    ny, nx = data_shape
    x0, x1, y0, y1 = extent
    rows, cols = np.where(mask)
    r0, r1 = max(rows.min() - margin_px, 0), min(rows.max() + margin_px, ny)
    c0, c1 = max(cols.min() - margin_px, 0), min(cols.max() + margin_px, nx)

    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y1, y0, ny)  # invertido porque origin='upper'
    return (xs[c0], xs[c1 - 1], ys[r1 - 1], ys[r0])


def plot_plume_zoom(data, extent, mask=None, zoom_bbox=None,
                     percentile_max=99.5, cmap='inferno', pix_res_m=30):
    valid = data[~np.isnan(data)]
    vmin, vmax = 0, np.percentile(valid, percentile_max)

    if zoom_bbox is None and mask is not None:
        zoom_bbox = zoom_bbox_from_mask(mask, extent, data.shape, margin_px=150)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
    ax.set_facecolor('black')
    im = ax.imshow(data, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')

    if zoom_bbox is not None:
        xmin, xmax, ymin, ymax = zoom_bbox
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    ax.set_aspect('equal')
    ax.axis('off')

    scalebar = ScaleBar(1, "m", length_fraction=0.25, color='white',
                         box_alpha=0, location='lower left')
    ax.add_artist(scalebar)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# 4) L = distancia máxima del casco convexo (Ec. 3, Balasus et al.)
# ---------------------------------------------------------

def convex_hull_max_distance(mask, pix_res_m):
    rows, cols = np.where(mask)
    if len(rows) < 3:
        return np.sqrt(len(rows)) * pix_res_m

    points = np.column_stack((cols, rows)).astype(float) * pix_res_m
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    L = np.max(pdist(hull_points))

    return L


# ---------------------------------------------------------
# 5) IME + Q, siguiendo Ec. 3 del paper
# ---------------------------------------------------------

def compute_Q_nh3(data, mask, pix_area_cm2, pix_res_m, ts, source_lonlat,
                   pbase, psave, name):
    N_avogadro = 6.02214076e23
    M_NH3 = 0.0170305  # kg/mol

    col_masked = np.where(mask, data, 0)
    col_masked = np.nan_to_num(col_masked, nan=0)
    total_molec = np.sum(col_masked * pix_area_cm2)
    IME = (total_molec / N_avogadro) * M_NH3  # kg

    N_pix = np.sum(mask)
    L = convex_hull_max_distance(mask, pix_res_m)

    lon_s, lat_s = source_lonlat

    print(f"IME = {IME:.2f} kg, L (convex hull) = {L:.1f} m, N_pix = {N_pix}")

    u10 = wind_speed_manual(ts, lat_s, lon_s, pbase, psave, name)  # m/s

    Q = IME * 3600 * u10 / L  # kg/h

    print(f"u10 = {u10:.2f} m/s ({u10*3600:.0f} m/h)")
    print(f"Q(NH3) = {Q:.2f} kg/h")

    return Q, u10, IME, L


# ---------------------------------------------------------
# 6) Uso
# ---------------------------------------------------------

if __name__ == "__main__":

    tif_path = r"C:\Users\jefer\OneDrive\Documentos\GitHub\HS_tool\tan20250114t062040c00s4001.tif"
    name = "tan20250114t062040c00s4001"
    pbase = "C:/output/"
    psave = "C:/output/"

    data, lat, lon, extent, pix_area_cm2, pix_res_m, crs_src = load_tanager_tif(tif_path)
    ts = parse_timestamp_from_name(name)
    print(f"Timestamp: {ts}")

    mask, source_xy = select_plume_tanager(data, extent, percentile_max=99.5)

    if crs_src.to_epsg() != 4326:
        transformer = pyproj.Transformer.from_crs(crs_src, "EPSG:4326", always_xy=True)
        lon_s, lat_s = transformer.transform(source_xy[0], source_xy[1])
    else:
        lon_s, lat_s = source_xy

    Q, u10, IME, L = compute_Q_nh3(data, mask, pix_area_cm2, pix_res_m, ts,
                                     (lon_s, lat_s), pbase, psave, name)

    # Plot final "bonito", con zoom automático a la pluma delineada
    plot_plume_zoom(data, extent, mask=mask, percentile_max=99.5, pix_res_m=pix_res_m)