#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 10:26:32 2026

@author: jroger
"""

import numpy as np
import spectral.io.envi as envi
import gc #to remove unnecessary variables (too heavy) 
 
def read_AVNG_img(path_img, name_img):
    
    try:
        img_spec = envi.open(path_img + name_img + "_img.hdr"); 
    except:
        img_spec = envi.open(path_img + name_img + ".hdr")
        
    wvl = img_spec.metadata['wavelength']
    fwhm = img_spec.metadata['fwhm']
    fill_val = -9999.0
 
    img = np.array(img_spec.load()) * 10; #We multiply to obtain rad units of $Wm^{-2}sr^{-1}\mu m^{-1}$ (as we normally have with PRISMA and EnMAP)
 
    img_v2 = np.full(img.shape, np.nan); 
    img_v2[img != fill_val*10] = img[img != fill_val*10]; img = img_v2
    del img_v2, img_spec; gc.collect()
 
    fwhm_aux = np.zeros((len(wvl)))
    for i in range(0, len(wvl)):
        wvl[i] = float(wvl[i])
        fwhm_aux[i] = float(fwhm[i])
    wvl = np.array(wvl)
    fwhm = np.array(fwhm_aux)
 
    try:
        img_obs = np.array(envi.open(path_img + name_img + "_obs_ort.hdr").load()) 
        img_obs_v2 = np.full(img_obs.shape, np.nan);
        img_obs_v2[img_obs != fill_val] = img_obs[img_obs != fill_val]; img_obs = img_obs_v2
        vza, sza = img_obs[:,:,2], img_obs[:,:,4]; 
        vza_mu, vza_std, sza_mu, sza_std = np.nanmean(vza), np.nanstd(vza), np.nanmean(sza), np.nanstd(sza)
        print('VZA = ' , round(np.nanmean(vza),2), ' +- ', round(np.nanstd(vza),2))
        print('SZA = ' , round(np.nanmean(sza),2), ' +- ', round(np.nanstd(sza),2))
        del img_obs, img_obs_v2, vza, sza; gc.collect()
    except:
        vza_mu, sza_mu = 0, 30 #Default for this kind of file - maybe we can check in archive
        print('VZA = 0, SZA = 30 - Default for this kind of file - maybe we can check in archive')
        
    verbose = 0
    #return img, wvl, fwhm, vza_mu, sza_mu, glt_x, glt_y, map_info
    
    return img, wvl, fwhm, vza_mu, sza_mu


def AVNG_loc_time_gcps(path_img, name_img): 
    
    img_spec = np.array(envi.open(path_img + name_img + "_loc.hdr").load()); 

    lon, lat = img_spec[:,:,0], img_spec[:,:,1]

    lat_c, lon_c = (np.min(lat) + np.max(lat))/2, (np.min(lon) + np.max(lon))/2 #Mean latitude and longitude
    
    with open(path_img + 'placemark_' + name_img + '.placemark', 'w') as f:

        lines_once = ['<?xml version="1.0" encoding="ISO-8859-1"?>', '<Placemarks>', '</Placemarks>']
        f.write(lines_once[0])
        f.write('\n')
        f.write(lines_once[1])
        f.write('\n')
        arr_i = np.arange(0,lon.shape[0],100)
        arr_j = np.arange(0,lon.shape[1],100)
        cont = 0
        for i_,i in enumerate(arr_i):
            for j_,j in enumerate(arr_j):
                cont += 1
                lines_loop = ['\t<Placemark name="' + 'gcp_' + str(cont) + '">','\n', '\t\t<LABEL>GCP ' + str(cont) + '</LABEL>', '\n', '\t\t<DESCRIPTION />', '\n', '\t\t<LATITUDE>' + str(lat[i,j]) + '</LATITUDE>', '\n', '\t\t<LONGITUDE>' + str(lon[i,j]) + '</LONGITUDE>', '\n', '\t\t<PIXEL_X>' + str(j+1) + '</PIXEL_X>', '\n', '\t\t<PIXEL_Y>' + str(i+1) + '</PIXEL_Y>', '\n', '\t\t<STYLE_CSS>symbol:plus; stroke:#ff8800; stroke-opacity:0.8; stroke-width:1.0</STYLE_CSS>', '\n', '\t</Placemark>', '\n']
                f.writelines(lines_loop)
        f.write(lines_once[2])
        
    time_a, time_b = name_img[3:11], name_img[12:18]
    time = time_a + time_b  
    
        
    return time, lat_c, lon_c, lat, lon


#path_img = '/home1/jroger/Desktop/postdoc/MAMAP/1st_test/avirisng/point1/ang20170825t201509_rdn_v2p9/'
#name_img = 'ang20170825t201509_rdn_v2p9'
#time, lat_c, lon_c, lat, lon = AVIRIS_NG_loc_time_gcps(path_img, name_img)
