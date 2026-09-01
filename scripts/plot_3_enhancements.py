"""Tres PNGs con el mismo encuadre: CH4 propio, NH3 propio y el quicklook de Carbon Mapper."""
import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import rasterio

P = "C:/Users/jefer/OneDrive/Documentos/GitHub/agu26-amonia2/TanagerScene 20241004_081921_28_4001 Core Imagery/"
OUT = "C:/Users/jefer/OneDrive/Documentos/GitHub/agu26-amonia2/out_tanager/"
SCENE = "20241004_081921_28_4001"

# Malla ortho comun (la que declara el propio HDF5)
gt = [609150.0, 30.0, 0.0, 3922020.0, 0.0, -30.0]
rows, cols, epsg = 1040, 943, 32637
extent = [gt[0], gt[0] + cols*gt[1], gt[3] + rows*gt[5], gt[3]]


def frame(ax):
    ax.set_xlabel(f'Easting (m, EPSG:{epsg})')
    ax.set_ylabel('Northing (m)')
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.ticklabel_format(style='plain', axis='both')


def save(arr, vmin, vmax, cmap, cbl, title, fname, mask_zero=False):
    a = np.array(arr, dtype=float)
    if mask_zero:
        a[a == 0] = np.nan
    fig, ax = plt.subplots(figsize=(9.5, 11.5))
    ax.set_facecolor('#111111')
    im = ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent,
                   interpolation='nearest')
    plt.colorbar(im, ax=ax, label=cbl, shrink=0.65, pad=0.02)
    ax.set_title(title, fontsize=11)
    frame(ax)
    fig.tight_layout()
    fig.savefig(OUT + fname, dpi=150, facecolor='white')
    plt.close(fig)
    print('->', fname)


# --- 1) CH4 propio -------------------------------------------------------
ch4 = np.load(OUT + 'mf_ch4.npy')
s4 = np.nanstd(ch4)
save(ch4, 0, 3*s4, 'inferno', r'$\Delta$XCH$_4$ (ppm$\cdot$m)',
     f'CH$_4$ enhancement - matched filter propio\nTanager basic_radiance | {SCENE}',
     'enh_1_ch4_propio.png')

# --- 2) NH3 propio -------------------------------------------------------
nh3 = np.load(OUT + 'mf_nh3_basic.npy')
s3 = np.nanstd(nh3)
save(nh3, 0, 3*s3, 'viridis', r'$\Delta$XNH$_3$ (ppm$\cdot$m)',
     f'NH$_3$ enhancement - matched filter propio\nTanager basic_radiance | {SCENE}',
     'enh_2_nh3_propio.png')

# --- 3) Quicklook CH4 de Carbon Mapper -----------------------------------
with rasterio.open(P + f'{SCENE}_ortho_ql_ch4.tif') as s:
    ql = s.read(1).astype(float)
    qt = s.transform
ql[ql == 0] = np.nan
ext_ql = [qt.c, qt.c + s.width*qt.a, qt.f + s.height*qt.e, qt.f]

fig, ax = plt.subplots(figsize=(9.5, 11.5))
ax.set_facecolor('#111111')
im = ax.imshow(ql, cmap='inferno', vmin=0, vmax=250, extent=ext_ql,
               interpolation='nearest')
plt.colorbar(im, ax=ax, label='CH$_4$ quicklook (DN 1-250, no cuantitativo)',
             shrink=0.65, pad=0.02)
ax.set_title(f'CH$_4$ - quicklook oficial Carbon Mapper / Planet\n'
             f'ortho_ql_ch4 | {SCENE}', fontsize=11)
frame(ax)
fig.tight_layout()
fig.savefig(OUT + 'enh_3_ch4_carbonmapper.png', dpi=150, facecolor='white')
plt.close(fig)
print('-> enh_3_ch4_carbonmapper.png')
