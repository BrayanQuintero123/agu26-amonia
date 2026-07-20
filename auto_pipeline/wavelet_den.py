#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anomalia wavelet 'den' para VISUALIZAR y DELIMITAR la pluma (no para cuantificar).

Codigo portado de wavelet_plume_finder_public/Wavelet_functions.py (funciones
consist_signals, modify_wt_coeffs, _match_coeff_dims, denoise_image,
wavelet_processing). Se copian aqui las minimas necesarias para NO arrastrar la
dependencia de Image_proc_functions (cv2, shapely, geojson, osgeo). Solo necesita
pywt + numpy.

Idea (emit_ime::detect_sources_wavelet): con set_zero=(1,0,0,0) se anulan los
coeficientes de aproximacion (fondo regional) conservando el detalle (anomalias
locales); el resultado 'den' tiene fondo ~= 0 y resalta las fuentes, quitando el
fondo topografico que ensucia el enhancement del matched filter. 'den' NO conserva
masa -> se usa SOLO para geometria/visualizacion, jamas como valor fisico.
"""

import math
import numpy as np
import pywt


def _consist_signals(image, scale):
    output_image = image.copy()
    mu, sigma = np.mean(output_image.flatten()), np.std(output_image.flatten())
    threshold = mu + scale * sigma
    output_image[output_image > threshold] = np.max(output_image.flatten())
    return output_image


def _modify_wt_coeffs(image, wavelet, level, set_zero):
    coeffs = pywt.wavedec2(image, wavelet, level=level, mode='symmetric')
    approximations = coeffs[0]
    details = coeffs[1:]

    (approx_zero, detail0_zero, detail1_zero, detail2_zero) = set_zero
    modified_details = []
    if approx_zero:
        approximations = np.zeros_like(approximations)
    for (cH, cV, cD) in details:
        cH = np.zeros_like(cH) if detail0_zero else cH
        cV = np.zeros_like(cV) if detail1_zero else cV
        cD = np.zeros_like(cD) if detail2_zero else cD
        modified_details.append((cH, cV, cD))

    modified_coeffs = list((approximations, *modified_details))
    return modified_coeffs


def _match_coeff_dims(a_coeff, d_coeff):
    size_diffs = np.subtract(a_coeff.shape, d_coeff.shape)
    if np.any((size_diffs < 0) | (size_diffs > 1)):
        raise ValueError("incompatible coefficient array sizes")
    return a_coeff[tuple(slice(s) for s in d_coeff.shape)]


def _denoise_image(image, wavelet, noiseSigma):
    row, col = image.shape
    n = math.floor(np.log2(row))

    coeffs = pywt.wavedec2(image, wavelet, level=n, mode='symmetric')
    approximation = coeffs[0]
    details = coeffs[1:]

    threshold = noiseSigma * np.sqrt(2 * np.log2(row * col))
    denoised_details = []
    for detail in details:
        denoised_detail = pywt.threshold(detail, threshold, mode='soft')
        denoised_details.append(tuple(denoised_detail))

    denoised_coeffs = (approximation, *denoised_details)
    denoised_image = pywt.waverec2(denoised_coeffs, wavelet, mode='symmetric')
    denoised_image = _match_coeff_dims(denoised_image, image)
    return denoised_image


def _wavelet_processing(image, wavelet, denoise_wavelet, scale, level, noiseSigma, set_zero):
    input_image = _consist_signals(image, scale)
    modified_coeffs = _modify_wt_coeffs(input_image, wavelet, level, set_zero)
    reconstructed_image = pywt.waverec2(modified_coeffs, wavelet, mode='symmetric')
    reconstructed_image[reconstructed_image < 0] = 0
    reconstructed_image = _match_coeff_dims(reconstructed_image, input_image)
    subtract_image = input_image - reconstructed_image
    subtract_image[subtract_image < 0] = 0
    denoised_image = _denoise_image(subtract_image, denoise_wavelet, noiseSigma)
    return denoised_image


def compute_den(enh_ppmm, wavelet='haar', denoise_wavelet='bior4.4',
                scale=2.0, level=3, noise_sigma=0.02, set_zero=(1, 0, 0, 0)):
    """Wavelet-denoised anomaly map from a matched-filter enhancement (ppm·m).
    Same preprocessing/params as emit_ime::detect_sources_wavelet. Background ~= 0,
    sources highlighted. Returns a float32 array of the same shape as enh_ppmm."""
    img = np.nan_to_num(np.asarray(enh_ppmm, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    img[img < 0] = 0.0
    den = _wavelet_processing(img, wavelet, denoise_wavelet, scale, level, noise_sigma, set_zero)
    return den
