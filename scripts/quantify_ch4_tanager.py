"""Cuantificacion IME de las plumas de CH4 de Tanager y comparacion con Carbon Mapper.

Carbon Mapper usa Q = IME * u / L (verificado contra su propio JSON). Usando SU
viento y SU fetch, toda diferencia en Q viene del enhancement, no del viento.
"""
import json
import numpy as np
import rasterio
from skimage import measure
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
import pyproj

P = "C:/Users/jefer/OneDrive/Documentos/GitHub/agu26-amonia2/TanagerScene 20241004_081921_28_4001 Core Imagery/"
OUT = "C:/Users/jefer/OneDrive/Documentos/GitHub/agu26-amonia2/out_tanager/"
SCENE = "20241004_081921_28_4001"

R_GAS, M_CH4 = 8.314462, 0.016043     # J/(mol K), kg/mol
P_SURF, T_SURF = 99000.0, 300.0       # Pa, K -- Deir ez-Zor ~200 m, Oct, 11:19 local

# 1 ppm.m -> kg/m2 de CH4
KG_PER_M2_PER_PPMM = (P_SURF / (R_GAS * T_SURF)) * 1e-6 * M_CH4


def convex_hull_L(mask, pix_res_m):
    r, c = np.where(mask)
    if len(r) < 3:
        return np.sqrt(max(len(r), 1)) * pix_res_m
    pts = np.column_stack((c, r)).astype(float) * pix_res_m
    return np.max(pdist(pts[ConvexHull(pts).vertices]))


# --- datos -----------------------------------------------------------------
mf = np.load(OUT + 'mf_ch4.npy')                       # ppm.m, malla ortho
gt = [609150.0, 30.0, 0.0, 3922020.0, 0.0, -30.0]
rows, cols, epsg = 1040, 943, 32637
PIX = 30.0
A_PIX = PIX * PIX                                       # m2

with rasterio.open(P + f'{SCENE}_ortho_ql_ch4.tif') as s:
    ql, qt = s.read(1), s.transform

gx = gt[0] + (np.arange(cols) + 0.5) * gt[1]
gy = gt[3] + (np.arange(rows) + 0.5) * gt[5]
GX, GY = np.meshgrid(gx, gy)
cix = ((GX - qt.c) / qt.a).astype(int)
ciy = ((GY - qt.f) / qt.e).astype(int)
inb = (cix >= 0) & (cix < ql.shape[1]) & (ciy >= 0) & (ciy < ql.shape[0])
plume = np.zeros(GX.shape, bool)
plume[inb] = ql[ciy[inb], cix[inb]] > 0
lab = measure.label(plume, connectivity=2)

feats = json.load(open(P + f'{SCENE}_ql_ch4_json.geojson'))['features']
tr = pyproj.Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True)

# --- asignar cada fuente reportada a su cluster ----------------------------
recs = []
for f in feats:
    pr = f['properties']
    lon, lat = f['geometry']['coordinates']
    x, y = tr.transform(lon, lat)
    j = int((x - gt[0]) / gt[1]); i = int((y - gt[3]) / gt[5])
    L_id = 0
    if 0 <= i < rows and 0 <= j < cols:
        L_id = lab[i, j]
    if L_id == 0:   # la fuente cae fuera de la mascara: buscamos el cluster mas cercano
        best, bd = 0, 1e18
        for c_ in range(1, lab.max() + 1):
            ri, ci = np.where(lab == c_)
            d = np.min((ci - j)**2 + (ri - i)**2)
            if d < bd:
                bd, best = d, c_
        L_id = best if bd < (10**2) else 0
    recs.append(dict(pid=pr['plume_id'].split('_')[-1], cluster=L_id,
                     emission=pr['emission'], unc=pr['emission_uncertainty'],
                     ime_cm=pr['ime'], fetch=pr['fetch'],
                     u=pr['wind_speed_avg'], u_std=pr['wind_speed_std'],
                     quality=pr['plume_quality']))

# --- IME propio por cluster ------------------------------------------------
sigma = np.nanstd(mf[(~plume) & np.isfinite(mf)])
print(f"factor: 1 ppm.m = {KG_PER_M2_PER_PPMM*1e6:.3f} mg/m2 de CH4 "
      f"(P={P_SURF/100:.0f} hPa, T={T_SURF:.0f} K)")
print(f"sigma del fondo = {sigma:.1f} ppm.m\n")

ime_mine = {}
for c_ in range(1, lab.max() + 1):
    m = (lab == c_) & np.isfinite(mf)
    if m.sum() == 0:
        continue
    enh = np.clip(mf[m], 0, None)          # el IME solo integra realce positivo
    ime_mine[c_] = dict(ime=float(np.sum(enh) * KG_PER_M2_PER_PPMM * A_PIX),
                        npx=int(m.sum()),
                        L=float(convex_hull_L(lab == c_, PIX)))

# --- tabla comparativa -----------------------------------------------------
hdr = ('id', 'cl', 'IME_CM', 'IME_yo', 'ratio', 'u', 'fetch', 'Q_CM', 'Q_yo', 'dif%', 'cal')
print('%-3s %3s %8s %8s %6s %6s %6s %9s %9s %7s %5s' % hdr)
print('-' * 82)
agg = {}
for r in recs:                              # varias fuentes pueden compartir cluster
    agg.setdefault(r['cluster'], []).append(r)

rows_out = []
for c_, group in sorted(agg.items()):
    if c_ == 0 or c_ not in ime_mine:
        for r in group:
            print('%-3s %3s %8.1f %8s %6s %6.2f %6.0f %9.1f %9s %7s %5s'
                  % (r['pid'], '-', r['ime_cm'], '-', '-', r['u'], r['fetch'],
                     r['emission'], '-', '-', r['quality'][:4]))
        continue
    mine = ime_mine[c_]
    ime_cm = sum(r['ime_cm'] for r in group)       # sumamos si comparten cluster
    q_cm = sum(r['emission'] for r in group)
    u = np.mean([r['u'] for r in group])
    fetch = np.mean([r['fetch'] for r in group])
    # Q propio con SU viento y SU fetch -> aisla el enhancement
    q_mine = mine['ime'] * u * 3600 / fetch
    ids = '+'.join(r['pid'] for r in group)
    dif = 100 * (q_mine - q_cm) / q_cm
    print('%-3s %3d %8.1f %8.1f %6.2f %6.2f %6.0f %9.1f %9.1f %+7.0f %5s'
          % (ids, c_, ime_cm, mine['ime'], mine['ime']/ime_cm, u, fetch,
             q_cm, q_mine, dif, group[0]['quality'][:4]))
    rows_out.append((ids, ime_cm, mine['ime'], q_cm, q_mine))

print('-' * 82)
rt = np.array([r[2]/r[1] for r in rows_out])
print(f"\nratio IME (yo/CM): mediana {np.median(rt):.2f}  media {rt.mean():.2f}  "
      f"rango {rt.min():.2f}-{rt.max():.2f}")
tot_cm = sum(r[3] for r in rows_out); tot_me = sum(r[4] for r in rows_out)
print(f"total escena: CM {tot_cm:.0f} kg/h | yo {tot_me:.0f} kg/h "
      f"({100*(tot_me-tot_cm)/tot_cm:+.0f}%)")
