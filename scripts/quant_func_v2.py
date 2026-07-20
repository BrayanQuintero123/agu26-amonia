#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 17:03:59 2026

@author: javier
"""

## Based on 'quantification_func_v1' from 'https://github.com/jarojuan96/HS_CH4_Emission_Quantification.git'


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.signal import medfilt2d

from georreferencing import location_and_time, georreference, gcp_from_placemark, warp_array_for_display, warped_pixel_to_lonlat, lonlat_to_native_pixel
from wind_v2 import wind_speed_manual

import variable_definition

def extract_deltaP(): #extract P difference between the layer boundaries where the gas is expected to be - it is based on the specific ATM we used (adapt to own atmosphere)
    
    p_atm_file = variable_definition.p_atm
    fn_in = p_atm_file + 'RFM_midlatitude_summer.dat' #path to where your ATM is
    data = np.loadtxt(fn_in, comments="#")
    
    z, P, H = data[:,0][::-1], data[:,1][::-1], 0.5
    
    idx = np.where(np.abs(z - H) == np.min(np.abs(z - H)))[0][0]
    P_0, P_H = P[idx-1], P[idx]
    delta_P = P_0 - P_H
    H = H * 1000 #km->m
    
    #print(z, P, z[idx], P[idx])
    
    return delta_P, H #mb, m

def ppmm_to_kg(sum_xgas, gsd, gas): #converts the summatory of concentration enhancement within the plume to kg (IME)
    
    M_air = 0.0289644 #kg/mol - molar mass of air
    g = 9.80665 #m/s2 - NIST
    
    #Ueff and other wind related parameters (+ pix_res)
    if gas == 'ch4':
        M_gas = 0.01604246 #kg/mol - molar mass of ch4
    elif gas == 'co2':
        M_gas = 0.0440095 #kg/mol - molar mass of ch4
    elif gas == 'nh3':
        M_gas = 0.0170305 #kg/mol - molar mass of nh3
    elif gas == 'c2h4':
        M_gas = 0.0280532 #kg/mol - molar mass of c2h4
    elif gas == 'c2h2':
        M_gas = 0.02604 #kg/mol - molar mass of c2h2
        
    
    delta_P, H = extract_deltaP() #mb, m
    delta_P = delta_P * 100 #mb -> Pa
    
    #print(M_gas, M_air, H, g, delta_P)
    
    factor = (gsd**2) * M_gas * delta_P * 10**-6 / (g * M_air * H)
    
    IME = factor * sum_xgas #ppmm -> kg 
    
    return IME, factor

# Function to draw a line between 2 points
def draw_line(ax, start, end):
    ax.plot([start[0], end[0]], [start[1], end[1]], color='r')

def select_plume(plot_xch4, ref_rad, mission, gas, lat, lon, list_gcp): #Plume identification (if there's any). 1) zoom-in to the plume area. 2) pinpoint the source. 3) delineate the plume
    #Display is warped north-up (like the georreferenced tifs) for easier interpretation; all clicks are translated back to native sensor pixels below, so the mask/quantification math is untouched.

    if mission == 'EMIT':
        pix_res = 60 #m
    elif mission == 'EnMAP' or mission == 'PRISMA' or mission == 'GF5' or mission == 'ZY1':
        pix_res = 30 #m
    elif mission == 'AVIRIS-NG':
        pix_res = 4 #m (our specific acquisitions have pix_size 3-5 m)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True, constrained_layout=True)


    # set these map plot boundary values
    std_ret = np.nanstd(plot_xch4);
    vminval_ret, vmaxval_ret = 0, 2 * std_ret  # 2-std seems enough to detect plumes
    plot_xch4 = medfilt2d(plot_xch4, kernel_size = 3) #Instead, we plot the filtered-retrieval (better for detectability) -- kept in native pixel space, used for the mask math below

    rad = np.ones(ref_rad.shape)*np.nan;
    rad[ref_rad > 0] =  ref_rad[ref_rad > 0]; #avoiding potential corrupted pixels (although probably already masked)
    mean_rad = np.nanmean(rad); std_rad = np.nanstd(rad)
    vminval_rad, vmaxval_rad = mean_rad - 3*std_rad, mean_rad + 3*std_rad  # enough to see the image

    rad_warp, gt = warp_array_for_display(list_gcp, rad)
    gas_warp, _ = warp_array_for_display(list_gcp, plot_xch4, ref_gt=gt, ref_shape=rad_warp.shape)

    mappable1 = ax1.imshow(rad_warp, vmin=vminval_rad, vmax=vmaxval_rad, cmap='gray')
    mappable2 = ax2.imshow(gas_warp, vmin=vminval_ret, vmax=vmaxval_ret, cmap='plasma')
    
    fig.suptitle(
        '0) Check in your console whether you want to proceed.\n'
        '1) Zoom into plume area (use the magnifying glass icon). Press enter when done.\n'
        '2) Pinpoint the emission source (just one click).\n'
        '3) Delineate plume shape by left-clicking on the edges. Press enter when done.',
        fontsize=20
    )
    
    # One shared colorbar (important: use one mappable)
    cbar1 = fig.colorbar(mappable1, ax=ax1, orientation='horizontal', fraction = 0.04, pad=0.02)
    cbar1.set_label(r'L' + r' (W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)', fontsize=20)
    cbar1.ax.tick_params(labelsize=14)
    
    cbar2 = fig.colorbar(mappable2, ax=ax2, orientation='horizontal', fraction = 0.04, pad=0.02)
    cbar2.set_label(r'$\Delta$' + f'X{gas} (ppm·m)', fontsize=20)
    cbar2.ax.tick_params(labelsize=14)
    
    # Scalebar: attach to ONE axis only (choose one)
    #The display is resampled to the warped (north-up) grid, whose pixel size differs from the native gsd used for Q -- use the warped resolution here so the on-screen scale bar is accurate.
    pix_res_display = abs(gt[5]) * 111320 #deg -> m, ~constant along a meridian
    scalebar1 = ScaleBar(pix_res_display, "m", length_fraction=0.25)
    scalebar2 = ScaleBar(pix_res_display, "m", length_fraction=0.25)
    ax1.add_artist(scalebar1)
    ax2.add_artist(scalebar2)
    
    ax1.axis('off')
    ax2.axis('off')
    
    plt.rc('legend', fontsize=14)
    
    plt.show(block=False)
    plt.pause(0.1)
    
    ax1.set_autoscale_on(False) #Avoids resetting the zoom-in afterwards
    ax2.set_autoscale_on(False)
    
    #----------------
    
    #0) Accept whether we want to proceed with the quantification after a quicklook
    
    decision_made = False   
    while decision_made == False:
        decision = input('Check the map. Do you still want to proceed? y/n: ')
        if decision == 'n':
            quant_bool = False
            decision_made = True
        elif decision == 'y':
            quant_bool = True
            decision_made = True
        else:
            print('Not valid answer. Repeat again.')
    
    if quant_bool == False:
        plt.close()
        mask, bool_det, source_coord = None, False, None
        
    else:
        
        plt.show(block=False)
        plt.pause(0.1)
    
        # 1) Zooming-in or just accept the view
        while True:
            key_pressed = plt.waitforbuttonpress()
            if key_pressed:
                break
    
            
        #----------------
        
        # 2) Pinpoint plume source

        plt.draw()
        source = plt.ginput(n=1, timeout=0, show_clicks=True) #Pixel coordinates in the warped (north-up) display
        source_wx, source_wy = source[0][0], source[0][1]
        source_lon, source_lat = warped_pixel_to_lonlat(source_wx, source_wy, gt)
        source_x, source_y = lonlat_to_native_pixel(source_lon, source_lat, lat, lon) #back to native sensor pixel, used everywhere downstream
        source_coord = [source_x, source_y]

        ax1.scatter(source_wx, source_wy, s=150, facecolor='white', edgecolor='r', linewidth=3, marker='*') #Draws the source location
        ax2.scatter(source_wx, source_wy, s=150, facecolor='white', edgecolor='r', linewidth=3, marker='*')
        plt.draw()

        #print('Source pixel (native): ', source_x, source_y)

        #----------------

        # 3) Plume delineation

        polygon_warp = plt.ginput(n=-1, timeout=0, show_clicks=True, mouse_add=1, mouse_pop=3, mouse_stop=2)

        for i in range(len(polygon_warp)-1):
            draw_line(ax1, polygon_warp[i], polygon_warp[i+1])
            draw_line(ax2, polygon_warp[i], polygon_warp[i+1])

        plt.draw()  # Show the polygon
        plt.show()

        polygon = [] #translate every vertex from the warped display back to native pixel coordinates
        for (wx, wy) in polygon_warp:
            p_lon, p_lat = warped_pixel_to_lonlat(wx, wy, gt)
            p_x, p_y = lonlat_to_native_pixel(p_lon, p_lat, lat, lon)
            polygon.append((p_x, p_y))

        xv, yv = np.meshgrid(range(plot_xch4.shape[1]), range(plot_xch4.shape[0]), indexing='xy')
        points = np.hstack((xv.reshape((-1, 1)), yv.reshape((-1, 1))))

        path = matplotlib.path.Path(polygon) if len(polygon) >= 3 else None
        mask = path.contains_points(points) if path is not None else np.zeros(xv.shape[0]*xv.shape[1], dtype=bool)
        mask.shape = xv.shape
        plt.pause(0.5)
        plt.close()

        if not mask.any(): #degenerate polygon (too few vertices, or none of the vertices enclosed a pixel center) -- treat as no detection instead of crashing
            print('Empty selection (need at least 3 left-clicks enclosing an area). No detection.')
            bool_det = False
        else:
            rows, cols = np.where(mask) #native-space bounding box around the mask (the old warped-display zoom bounds no longer apply to the native array)
            pad = 10
            y0, ye = max(rows.min()-pad, 0), min(rows.max()+pad, mask.shape[0])
            x0, xe = max(cols.min()-pad, 0), min(cols.max()+pad, mask.shape[1])

            fig, ax = plt.subplots(figsize=(12,12));
            mappable = ax.imshow((mask*plot_xch4)[y0:ye,x0:xe], vmin=vminval_ret, vmax=vmaxval_ret, cmap='plasma')
            fig.suptitle('Masked Plume. \n Please close the window when done.', fontsize=20)
            cbar = plt.colorbar(mappable)
            cbar.ax.get_yaxis().labelpad = 15
            cbar.ax.set_ylabel('$\mathregular{\Delta XCH_4}$ [ppm]', rotation=270, fontsize=20)
            cbar.ax.tick_params(labelsize=14)
            scalebar = ScaleBar(pix_res, "m", length_fraction=0.25)
            ax.add_artist(scalebar)
            ax.axis('off')
            plt.rc('legend', fontsize=14)
            while plt.fignum_exists(fig.number):
                plt.pause(0.1)
            bool_det = True
        
    return mask, bool_det, source_coord

def extract_ueff (u10, mission, gas):
    
    if u10 < 4: 
        bool_thres = False
        err_u10 = 0.5 * u10  # 50% error (<4 m/s)
    else: #Carvalho et al., (2019)
        bool_thres = True
        err_u10 = 2.0  # 2 m/s (≥4 m/s)
        
    if gas == 'ch4':
        
        if mission == 'EMIT':
            a, b = 0.31, 0.4 #Guanter et al., (2024)
            pix_res = 60 #m
        elif mission == 'EnMAP' or mission == 'PRISMA' or mission == 'GF5' or mission == 'ZY1':
            a, b = 0.34, 0.44 #Guanter et al., (2021); Roger et al., (2024)
            pix_res = 30 #m
        elif mission == 'AVIRIS-NG': #Simplification - probably not accurate
            pix_res = 4
            a, b = 1, 0
            
        ueff = a*u10+b
        
        return ueff, a, err_u10, pix_res, bool_thres
    
    elif gas == 'co2': #simplification. calibrations should be done
        
        if mission == 'EMIT':
            a, b = 1, 0 
            pix_res = 60 #m
        elif mission == 'EnMAP' or mission == 'PRISMA' or mission == 'GF5' or mission == 'ZY1':
            a, b = 1, 0 
            pix_res = 30 #m
        elif mission == 'AVIRIS-NG':
            pix_res = 4
            a, b = 1, 0
            
        ueff = a*u10+b
        
        return ueff, a, err_u10, pix_res, bool_thres
    
    elif gas == 'c2h4':
        
        if mission == 'EMIT':
            a, b = 0.48, 0.59 #Extracted from our own simulations
            pix_res = 60 #m
        elif mission == 'EnMAP' or mission == 'PRISMA' or mission == 'GF5' or mission == 'ZY1':
            a, b = 0.43, 0.61 #Currently only valid for GF5A (simulations)
            pix_res = 30 #m
        elif mission == 'AVIRIS-NG': #Simplification - probably not accurate
            pix_res = 4
            a, b = 1, 0
            
        ueff = a*u10+b
        
        return ueff, a, err_u10, pix_res, bool_thres
    
    elif gas == 'c2h2':
        
        if mission == 'EMIT':
            a, b = 0.47, 0.68 #Don't know yet
            pix_res = 60 #m
        elif mission == 'EnMAP' or mission == 'PRISMA' or mission == 'GF5' or mission == 'ZY1':
            a, b = None, None #Don't know yet
            pix_res = 30 #m
        elif mission == 'AVIRIS-NG': #Simplification - probably not accurate
            pix_res = 4
            a, b = 1, 0
            
        ueff = a*u10+b
        
        return ueff, a, err_u10, pix_res, bool_thres
    
    
    elif gas == 'nh3':
                
        if mission == 'EMIT':
            pix_res = 60 #m
            
            #tau = inf
            a_1, b_1 = 0.47, 0.7
            #tau = 1 h
            a_2, b_2 = 0.45, 0.9
            
        elif mission == 'EnMAP' or mission == 'GF5':
            pix_res = 30 #m
            
            #tau = inf
            a_1, b_1 = 0.42, 0.68
            #tau = 1 h
            a_2, b_2 = 0.41, 0.85
            
        elif mission == 'PRISMA':
            pix_res = 30 #m
            
            #tau = inf
            a_1, b_1 = 0.43, 0.61
            #tau = 1 h
            a_2, b_2 = 0.42, 0.76
            
        elif mission == 'AVIRIS-NG': #Simplification - probably not accurate
            pix_res = 4
            a_1, b_1 = 1, 0
            a_2, b_2 = 1, 0
            
            
        ueff_1 = a_1*u10+b_1
        ueff_2 = a_2*u10+b_2
        
        return ueff_1, a_1, ueff_2, a_2, err_u10, pix_res, bool_thres            
    

def extract_Q(xgas_im, mask, u10, mission, gas):
    
    #IME, L and Ueff 
    sum_xgas = np.sum(xgas_im*mask) #xgas_im is in ppmm
    
    if gas == 'ch4' or gas == 'co2' or gas == 'c2h4' or gas == 'c2h2':
        ueff, a, err_u10, gsd, bool_thres = extract_ueff (u10, mission, gas)
    elif gas == 'nh3':
        ueff_1, a_1, ueff_2, a_2, err_u10, gsd, bool_thres = extract_ueff (u10, mission, gas)
        
    IME, conv_factor = ppmm_to_kg(sum_xgas, gsd, gas)
    L = np.sqrt(np.sum(mask)*gsd**2) #m
    N_pix = np.sum(mask) #Num pix within the mask
    
    if gas == 'ch4' or gas == 'c2h4' or gas == 'c2h2' or gas == 'co2':
        Q = IME * 3600 * ueff / L #3600 to convert to kg/h
        if bool_thres: 
            err_Q = np.sqrt((3600*IME*a*2/L)**2 + (3600*ueff*conv_factor*np.sqrt(N_pix)*np.std(xgas_im)/L)**2) #err(U10) = 2 m/s
        else:
            err_Q = np.sqrt((3600*IME*a*u10*(1/2)/L)**2 + (3600*ueff*conv_factor*np.sqrt(N_pix)*np.std(xgas_im)/L)**2)#50% err U10
            
        print(f'{gas}: Q = {round(Q,2)} ± {round(err_Q,2)} kg/h')
        
        return Q, err_Q, u10, err_u10
        
    elif gas == 'nh3':
        Q_1 = IME * 3600 * ueff_1 / L #3600 to convert to kg/h
        Q_2 = IME * 3600 * ueff_2 / L #3600 to convert to kg/h
        if bool_thres: 
            err_Q_1 = np.sqrt((3600*IME*a_1*2/L)**2 + (3600*ueff_1*conv_factor*np.sqrt(N_pix)*np.std(xgas_im)/L)**2) #err(U10) = 2 m/s
            err_Q_2 = np.sqrt((3600*IME*a_2*2/L)**2 + (3600*ueff_2*conv_factor*np.sqrt(N_pix)*np.std(xgas_im)/L)**2) #err(U10) = 2 m/s
        else:
            err_Q_1 = np.sqrt((3600*IME*a_1*u10*(1/2)/L)**2 + (3600*ueff_1*conv_factor*np.sqrt(N_pix)*np.std(xgas_im)/L)**2)#50% err U10
            err_Q_2 = np.sqrt((3600*IME*a_2*u10*(1/2)/L)**2 + (3600*ueff_2*conv_factor*np.sqrt(N_pix)*np.std(xgas_im)/L)**2)#50% err U10
        
        print(f'{gas}: Q(tau->inf) = {round(Q_1,2)} ± {round(err_Q_1,2)} kg/h')
        print(f'{gas}: Q(tau->1h) = {round(Q_2,2)} ± {round(err_Q_2,2)} kg/h')
    
        return Q_1, err_Q_1, Q_2, err_Q_2, u10, err_u10
    

    

def emission_quantification (dxgas_show, dxgas_quan, ref_rad, mission, gas, path_folder, name, psave):

    #lat/lon (native pixel space) and GCPs are needed up front now: select_plume warps the display north-up and
    #needs them to translate clicks back into native sensor pixels, before we even know whether there's a detection.
    ts, lat_c, lon_c, lat, lon = location_and_time(path_folder, name, mission)
    list_gcp = gcp_from_placemark(path_folder + 'placemark_' + name + '.placemark')

    mask, bool_det, source_coord = select_plume(dxgas_show, ref_rad, mission, gas, lat, lon, list_gcp)

    if gas == 'ch4' or gas == 'co2' or gas == 'c2h4' or gas == 'c2h2': #for the moment, only tau->infinity for c2h4
        if bool_det:
            lat_s, lon_s = georreference(path_folder, name, psave, name + '_tool4', gas, mask, source_coord)
            if mission == 'EMIT' or mission == 'PRISMA' or mission == 'AVIRIS-NG':
                lat_s, lon_s = lat[source_coord[1], source_coord[0]], lon[source_coord[1], source_coord[0]] #more accurate
            u10 = wind_speed_manual(ts, lat_s, lon_s, path_folder, psave, name) #To search winds, we change coordinates from central scene to source
            Q, err_Q, u10, err_u10 = extract_Q(dxgas_quan, mask, u10, mission, gas)
        else:
            Q, err_Q, u10, err_u10, lat_s, lon_s = None, None, None, None, None, None
            print('No detection')
        return Q, err_Q, u10, err_u10, lat_s, lon_s, ts, bool_det

    elif gas == 'nh3':

        if bool_det:
            lat_s, lon_s = georreference(path_folder, name, psave, name + '_tool4', gas, mask, source_coord)
            if mission == 'EMIT' or mission == 'PRISMA' or mission == 'AVIRIS-NG':
                lat_s, lon_s = lat[source_coord[1], source_coord[0]], lon[source_coord[1], source_coord[0]] #more accurate
            u10 = wind_speed_manual(ts, lat_s, lon_s, path_folder, psave, name)
            Q_1, err_Q_1, Q_2, err_Q_2, u10, err_u10 = extract_Q(dxgas_quan, mask, u10, mission, gas)
        else:
            Q_1, err_Q_1, Q_2, err_Q_2, u10, err_u10, lat_s, lon_s = None, None, None, None, None, None, None, None
            print('No detection')
        return Q_1, err_Q_1, Q_2, err_Q_2, u10, err_u10, lat_s, lon_s, ts, bool_det

    else:
        if bool_det:
            lat_s, lon_s = georreference(path_folder, name, psave, name + '_tool4', gas, mask, source_coord)
            if mission == 'EMIT' or mission == 'PRISMA' or mission == 'AVIRIS-NG':
                lat_s, lon_s = lat[source_coord[1], source_coord[0]], lon[source_coord[1], source_coord[0]] #more accurate
            u10 = wind_speed_manual(ts, lat_s, lon_s, path_folder, psave, name) #To search winds, we change coordinates from central scene to source
        else:
            u10, lat_s, lon_s = None, None, None

        return u10, lat_s, lon_s, ts, bool_det







