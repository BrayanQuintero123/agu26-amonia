import matplotlib
matplotlib.use('TkAgg')  # o 'Qt5Agg' si tienes PyQt5 instalado

from main import main
# ... resto de tu código
from main import main

site = 'Uzbekistan1'   # el nombre que quieras darle
gas = 'nh3'
p = "C:/Users/jefer/OneDrive/Documentos/GitHub/HS_tool/"   # con / no \
n = "EMIT_L1B_RAD_001_20241216T085215_2435106_004"
# en run.py
psave = "C:/output/"

main(p, n, psave, gas, site)