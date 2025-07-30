import numpy as np
import pandas as pd

# Read input Excel files
weights_rd_data =  pd.read_excel('../data/RDs_weights_data.xlsx', sheet_name=None)
rts_24_data =  pd.read_excel('../data/rts_24_data.xlsx', sheet_name=None)

# Generate dictionaries where each item is a sheet from the above Excel files
weights_rd = {}
rts_24 = {}
for weights_sheet_name, weight_rd_df in weights_rd_data.items():
    weights_rd.update({str(weights_sheet_name): weight_rd_df})
for rts_24_sheet_name, rts_24_df in rts_24_data.items():
    rts_24.update({str(rts_24_sheet_name): rts_24_df})

weights = weights_rd['weights']
RD1 = weights_rd['RD1']
RD2 = weights_rd['RD2']
RD3 = weights_rd['RD3']
RD4 = weights_rd['RD4']
RD5 = weights_rd['RD5']
RD6 = weights_rd['RD6']
RD7 = weights_rd['RD7']
RD8 = weights_rd['RD8']
RD9 = weights_rd['RD9']
RD10 = weights_rd['RD10']

lines = rts_24['TL']
buses = rts_24['Buses']
ESS = rts_24['ESS']
CG = rts_24['CG']
RES = rts_24['RES']
loads = rts_24['loads']
UB = rts_24['UB']

years_data = range(1,2)
tol = 0.005

SEl_data = []
for line, rel in zip(lines['Transmission line'], lines['From bus']):
    SEl_data.append([line, int(rel)])

gamma_dyth_data = []
for load, zone in zip(loads['Load'], loads['Zone']):
    for y in years_data:
        for j, RD in enumerate([RD1, RD2, RD3, RD4, RD5, RD6, RD7, RD8, RD9, RD10]):
            for RTP, gamma_west, gamma_east in zip(RD['RTP'], RD['gammaD_dth_west'], RD['gammaD_dth_east']):
                if zone == 'West':
                    gammaD = gamma_west
                elif zone == 'East':
                    gammaD = gamma_east
                gamma_dyth_data.append([load, y, j+1, RTP, gammaD])

gamma_ryth_data = []
for res, zone in zip(RES['Generating unit'], RES['Zone']):
    for y in years_data:
        for j, RD in enumerate([RD1, RD2, RD3, RD4, RD5, RD6, RD7, RD8, RD9, RD10]):
            for RTP, gamma_south, gamma_north in zip(RD['RTP'], RD['gammaRW_rth_south'], RD['gammaRW_rth_north']):
                if zone == 'South':
                    gammaR = gamma_south
                elif zone == 'North':
                    gammaR = gamma_north
                gamma_ryth_data.append([res, y, j+1, RTP, gammaR])

sigma_yt_data = []
for y in years_data:
    for RD, sigma in zip(weights['RD'], weights['sigma_t [days]']):
        sigma_yt_data.append([y, RD, sigma])

tau_yth_data = []
for y in years_data:
    for i, RD in enumerate([RD1, RD2, RD3, RD4, RD5, RD6, RD7, RD8, RD9, RD10]):
        for RTP, duration in zip(RD['RTP'], RD['tau_th [h]']):
            tau_yth_data.append([y, i+1, RTP, duration])

ES_syt0_data = []
for s, es0 in zip(ESS['Storage unit'], ESS['ES_s0 [MWh]']):
    for y in years_data:
        for RTP in RD1['RTP']:
            ES_syt0_data.append([s, y, RTP, es0])