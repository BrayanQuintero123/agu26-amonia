#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mapeo gas -> (banda_show, banda_quan) del cubo L2 (hs_tool4_deltaxgas.retrieval_maps).
En su propio modulo para que pipeline_auto y plot_plume_map lo compartan sin
importarse mutuamente (evita import circular).

show = banda sobre la que se detecta/delimita ; quan = banda que se integra (IME).
"""

GAS_BANDS = {
    'ch4':  (1, 1),     # ch4-MF(ppmm)
    'co2':  (3, 3),     # co2-MF(ppmm)  (Ueff = u10 simplification)
    'c2h4': (11, 7),    # show: c2h4-MF-SWIR ; quan: c2h4-MF-2300nm
    'c2h2': (25, 23),   # show: c2h2-MF-SWIR ; quan: c2h2-MF-1500nm
    'nh3':  (19, 13),   # show: nh3-MF-SWIR  ; quan: nh3-MF-2300nm  (also gives tau=1h estimate)
}
