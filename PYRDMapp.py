# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:24:21 2024

@author: ashto
"""

# app.py
import streamlit as st

# Title of the app
st.title("PYRDM App")

# A simple text input
name = st.text_input("Enter your name:")

# Display a message if the name is entered
if name:
    st.write(f"Hello, {name}!")

# A simple slider
age = st.slider("Select your age", 0, 100, 25)

# Display the selected age
st.write(f"You are {age} years old.")

# A simple button
if st.button("Click Me"):
    st.write("You clicked the button!")

##############################################################################
##########  PYRDM CODE 3.0 CREATED BY ASHTON EAVES
##############################################################################
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import streamlit as st

#from plotnine import ggplot, geom_point, aes, stat_smooth, facet_wrap
#os.chdir('C:/Users/ashto/OneDrive/Documents/GitHub/PYRDM_Github')

S1 = pd.read_excel('./inputs/RCP0.xlsm')
S2 = pd.read_excel('./inputs/RCP45.xlsm')
S3 = pd.read_excel('./inputs/RCP85.xlsm')
S4 = pd.read_excel('./inputs/RCP45def.xlsm')
S5 = pd.read_excel('./inputs/RCP85def.xlsm')
S6 = pd.read_excel('./inputs/RCP45bonds.xlsm')
S7 = pd.read_excel('./inputs/RCP45rates.xlsm')
S8 = pd.read_excel('./inputs/RCP85bonds.xlsm')
S9 = pd.read_excel('./inputs/RCP85rates.xlsm')

##############################################################################
#Loop through files, classify and index:
##############################################################################
list_of_dfs = [S1, S2, S3, S4, S5, S6, S7, S8, S9]
output_path = './outputs'
for index, data in enumerate(list_of_dfs):

##############################################################################
#Conditional statements for variables:

#Total savings, better than at the start of the period.
    data.loc[data['Annual Expected Loss'] < 25000000, 'Annual Expected Loss'] = 0
    data.loc[data['Annual Expected Loss'] >= 100000000, 'Annual Expected Loss'] = 8
    data.loc[(data['Annual Expected Loss'] >= 75000000)&(data['Annual Expected Loss'] < 100000000), 
       'Annual Expected Loss'] = 6
    data.loc[(data['Annual Expected Loss'] >= 50000000)&(data['Annual Expected Loss'] < 75000000), 
       'Annual Expected Loss'] = 4
    data.loc[(data['Annual Expected Loss'] >= 25000000)&(data['Annual Expected Loss'] < 50000000), 
       'Annual Expected Loss'] = 2
#Property premium justified before decline and the initial premium.
    data.loc[data['Property premium'] >= 5000, 'Property premium'] = 3
    data.loc[(data['Property premium'] >= 0)&(data['Property premium'] < 1000), 'Property premium'] = 4  
    data.loc[(data['Property premium'] >= 1000)&(data['Property premium'] < 2000), 'Property premium'] = 0
    data.loc[(data['Property premium'] >= 2000)&(data['Property premium'] < 3000), 'Property premium'] = 1
    data.loc[(data['Property premium'] >= 3000)&(data['Property premium'] < 5000), 'Property premium'] = 2
#hhldconsumprt, tracking increase with time.
    data.loc[data['hhldconsumprt'] < 1.24, 'hhldconsumprt'] = 4
    data.loc[data['hhldconsumprt'] >= 1.255, 'hhldconsumprt'] = 0 
    data.loc[(data['hhldconsumprt'] >= 1.25)&(data['hhldconsumprt'] < 1.255), 'hhldconsumprt'] = 1
    data.loc[(data['hhldconsumprt'] >= 1.245)&(data['hhldconsumprt'] < 1.25), 'hhldconsumprt'] = 2
    data.loc[(data['hhldconsumprt'] >= 1.24)&(data['hhldconsumprt'] < 1.245), 'hhldconsumprt'] = 3
#unemployment rate, the lower the better.
    data.loc[data['unemploymentrt'] >= 0.048, 'unemploymentrt'] = 4
    data.loc[(data['unemploymentrt'] >= 0.044)&(data['unemploymentrt'] < 0.045), 'unemploymentrt'] = 1
    data.loc[(data['unemploymentrt'] >= 0.045)&(data['unemploymentrt'] < 0.046), 'unemploymentrt'] = 2
    data.loc[(data['unemploymentrt'] >= 0.046)&(data['unemploymentrt'] < 0.048), 'unemploymentrt'] = 3
    data.loc[data['unemploymentrt'] < 0.044, 'unemploymentrt'] = 0
#Govt balance at start of period, more consumption is better.
    data.loc[data['CentralG Consumption'] >= 2200, 'CentralG Consumption'] = 0 
    data.loc[data['CentralG Consumption'] < 1400, 'CentralG Consumption'] = 4 
    data.loc[(data['CentralG Consumption'] >= 1800)&(data['CentralG Consumption'] < 2000), 
       'CentralG Consumption'] = 2     
    data.loc[(data['CentralG Consumption'] >= 1400)&(data['CentralG Consumption'] < 1800), 
       'CentralG Consumption'] = 3    
    data.loc[(data['CentralG Consumption'] >= 2000)&(data['CentralG Consumption'] < 2200), 
       'CentralG Consumption'] = 1
#Govt balance at start of period, more consumption is better.
    data.loc[data['LocalG Consumption'] < 200, 'LocalG Consumption'] = 4 
    data.loc[data['LocalG Consumption'] >= 350, 'LocalG Consumption'] = 0
    data.loc[(data['LocalG Consumption'] >= 250)&(data['LocalG Consumption'] < 300),
       'LocalG Consumption'] = 2     
    data.loc[(data['LocalG Consumption'] >= 300)&(data['LocalG Consumption'] < 350), 
       'LocalG Consumption'] = 3    
    data.loc[(data['LocalG Consumption'] >= 200)&(data['LocalG Consumption'] < 250), 
       'LocalG Consumption'] = 1
#Total value added, tracking increase with time.
    data.loc[data['dtotalvalueadded'] < 8000, 'dtotalvalueadded'] = 8
    data.loc[(data['dtotalvalueadded'] >= 8000)&(data['dtotalvalueadded'] < 9500), 'dtotalvalueadded'] = 6
    data.loc[(data['dtotalvalueadded'] >= 9500)&(data['dtotalvalueadded'] < 11000), 'dtotalvalueadded'] = 4
    data.loc[(data['dtotalvalueadded'] >= 11000)&(data['dtotalvalueadded'] < 12500), 'dtotalvalueadded'] = 2
    data.loc[data['dtotalvalueadded'] >= 12500, 'dtotalvalueadded'] = 0
#actual production, tracking increase with time.
    data.loc[data['totactualprod'] < 24000, 'totactualprod'] = 4
    data.loc[(data['totactualprod'] >= 24000)&(data['totactualprod'] < 28000), 'totactualprod'] = 3
    data.loc[(data['totactualprod'] >= 28000)&(data['totactualprod'] < 32000), 'totactualprod'] = 2            
    data.loc[(data['totactualprod'] >= 32000)&(data['totactualprod'] < 36000), 'totactualprod'] = 1
    data.loc[data['totactualprod'] >= 36000, 'totactualprod'] = 0
#Pinvestcc, price of investment capital, lower the better.
    data.loc[data['Pinvestcc'] >= 1.4, 'Pinvestcc'] = 4
    data.loc[data['Pinvestcc'] <= 1.05, 'Pinvestcc'] = 0
    data.loc[(data['Pinvestcc'] >= 1.05)&(data['Pinvestcc'] < 1.2), 'Pinvestcc'] = 1
    data.loc[(data['Pinvestcc'] >= 1.2)&(data['Pinvestcc'] < 1.3), 'Pinvestcc'] = 2
    data.loc[(data['Pinvestcc'] >= 1.3)&(data['Pinvestcc'] < 1.4), 'Pinvestcc'] = 3
#Landuse ratio, higher the ratio is better.
    data.loc[data['Landuse ratio'] < 0.5, 'Landuse ratio'] = 4
    data.loc[(data['Landuse ratio'] >= 0.5)&(data['Landuse ratio'] < 1),'Landuse ratio'] = 3
    data.loc[(data['Landuse ratio'] >= 1)&(data['Landuse ratio'] < 1.5),'Landuse ratio'] = 2
    data.loc[(data['Landuse ratio'] >= 1.5)&(data['Landuse ratio'] < 2),'Landuse ratio'] = 1  
    data.loc[data['Landuse ratio'] >= 2, 'Landuse ratio'] = 0

#############################################################################
#Export loops to multiple CSVs in output folder:
    filepath = os.path.join(output_path, 'S_'+str(index)+'.csv')
    data.to_csv(filepath)
    
#############################################################################
#Read outputs back in for min regret analysis:
#############################################################################
RCP0_B = pd.read_csv('./outputs/S_0.csv')
RCP45_B = pd.read_csv('./outputs/S_1.csv')
RCP85_B = pd.read_csv('./outputs/S_2.csv')
RCP45_def = pd.read_csv('./outputs/S_3.csv')
RCP85_def = pd.read_csv('./outputs/S_4.csv')
RCP45_bonds = pd.read_csv('./outputs/S_5.csv')
RCP45_rates = pd.read_csv('./outputs/S_6.csv')
RCP85_bonds = pd.read_csv('./outputs/S_7.csv')
RCP85_rates = pd.read_csv('./outputs/S_8.csv')
#Delete index column:
a = RCP0_B.values
a = np.delete(a, (0), axis=1)
b = RCP45_B.values
b = np.delete(b, (0), axis=1)
c = RCP85_B.values
c = np.delete(c, (0), axis=1)
d = RCP45_def.values
d = np.delete(d, (0), axis=1)
e = RCP85_def.values
e = np.delete(e, (0), axis=1)
f = RCP45_bonds.values
f = np.delete(f, (0), axis=1)
g = RCP45_rates.values
g = np.delete(g, (0), axis=1)
h = RCP85_bonds.values
h = np.delete(h, (0), axis=1)
j = RCP85_rates.values
j = np.delete(j, (0), axis=1)
#Column and row names for variables and timesteps:
column_names1 = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9','V10']
row_names1    = ['2020','2020.25','2020.5','2020.75','2021','2021.25','2021.5','2021.75','2022',
                 '2022.25','2022.5','2022.75','2023','2023.25','2023.5','2023.75','2024',
                 '2024.25','2024.5','2024.75','2025','2025.25','2025.5','2025.75','2026',
                 '2026.25','2026.5','2026.75','2027','2027.25','2027.5','2027.75','2028',
                 '2028.25','2028.5','2028.75','2029','2029.25','2029.5','2029.75','2030',
                 '2030.25','2030.5','2030.75','2031','2031.25','2031.5','2031.75','2032',
                 '2032.25','2032.5','2032.75','2033','2033.25','2033.5','2033.75','2034',
                 '2034.25','2034.5','2034.75','2035','2035.25','2035.5','2035.75','2036',
                 '2036.25','2036.5','2036.75','2037','2037.25','2037.5','2037.75','2038',
                 '2038.25','2038.5','2038.75','2039','2039.25','2039.5','2039.75','2040',
                 '2040.25','2040.5','2040.75','2041','2041.25','2041.5','2041.75','2042',
                 '2042.25','2042.5','2042.75','2043','2043.25','2043.5','2043.75','2044',
                 '2044.25','2044.5','2044.75','2045','2045.25','2045.5','2045.75','2046',
                 '2046.25','2046.5','2046.75','2047','2047.25','2047.5','2047.75','2048',
                 '2048.25','2048.5','2048.75','2049','2049.25','2049.5','2049.75','2050',
                 'sum','rank']
column_names2 = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9','V10' ,'sum','rank']
row_names2    = ['2020','2020.25','2020.5','2020.75','2021','2021.25','2021.5','2021.75','2022',
                 '2022.25','2022.5','2022.75','2023','2023.25','2023.5','2023.75','2024',
                 '2024.25','2024.5','2024.75','2025','2025.25','2025.5','2025.75','2026',
                 '2026.25','2026.5','2026.75','2027','2027.25','2027.5','2027.75','2028',
                 '2028.25','2028.5','2028.75','2029','2029.25','2029.5','2029.75','2030',
                 '2030.25','2030.5','2030.75','2031','2031.25','2031.5','2031.75','2032',
                 '2032.25','2032.5','2032.75','2033','2033.25','2033.5','2033.75','2034',
                 '2034.25','2034.5','2034.75','2035','2035.25','2035.5','2035.75','2036',
                 '2036.25','2036.5','2036.75','2037','2037.25','2037.5','2037.75','2038',
                 '2038.25','2038.5','2038.75','2039','2039.25','2039.5','2039.75','2040',
                 '2040.25','2040.5','2040.75','2041','2041.25','2041.5','2041.75','2042',
                 '2042.25','2042.5','2042.75','2043','2043.25','2043.5','2043.75','2044',
                 '2044.25','2044.5','2044.75','2045','2045.25','2045.5','2045.75','2046',
                 '2046.25','2046.5','2046.75','2047','2047.25','2047.5','2047.75','2048',
                 '2048.25','2048.5','2048.75','2049','2049.25','2049.5','2049.75','2050']
#Column and row names for scenarios:
column_names3 = ['RCP0', 'RCP45', 'RCP8.5', 'RCP4.5 Defence', 'RCP8.5 Defence','RCP4.5 Bonds',
                 'RCP4.5 Rates', 'RCP8.5 Bonds', 'RCP8.5 Rates']
column_names4 = ['RCP0', 'RCP4.5', 'RCP8.5', 'RCP4.5 Defence', 'RCP8.5 Defence','RCP4.5 Bonds',
                 'RCP4.5 Rates', 'RCP8.5 Bonds', 'RCP8.5 Rates','sum','rank']
#Scenario names:
#labels = ['RCP0', 'RCP4.5', 'RCP8.5', 'RCP4.5 Defence', 'RCP8.5 Defence', 'RCP4.5 Bonds', 
#          'RCP4.5 Rates', 'RCP8.5 Bonds', 'RCP8.5 Rates']
labels = ['1. No SLR', '2. Baseline', '3. Worst-case baseline', '4. Defend', 
          '5. Worst-case defend', '6. Managed retreat bonds', '7. Managed retreat rates', 
          '8. Worst-case managed retreat bonds', '9. Worst-case managed retreat rates']
##############################################################################
#Condensed Min Regret Matrix:
##############################################################################
def sum_rank(i):
    #difference or least regret between variables and scenarios:
    p = np.array(i)
    q = np.min(p,axis=0)
    r = np.min(p,axis=1)
    cdif = p-q
    rdif = p-r[:,None]
    #find the sum of the rows and columns for the difference arrays:
    sumc = np.sum(cdif,axis=0)
    sumr = np.sum(rdif,axis=1)
    sumra = np.reshape(sumr,(121,1))
    #append the scenario array with the column sums:
    sumcol = np.zeros((122,10))
    sumcol = np.append([cdif],[sumc])
    sumcol.shape = (122,10)
    #rank columns:
    order0 = sumc.argsort()
    rank0 = order0.argsort()
    rankcol = np.zeros((123,10))
    rankcol = np.append([sumcol],[rank0])
    rankcol.shape = (123,10)
    #append the variable array with row sums:
    sumrow = np.zeros((121,11))
    sumrow = np.hstack((rdif,sumra))
    #rank rows:
    order1 = sumr.argsort()
    rank1 = order1.argsort()
    rank1r = np.reshape(rank1,(121,1))
    rankrow = np.zeros((121,12))
    rankrow = np.hstack((sumrow,rank1r))
    #Add row and column headers for least regret for df0:
    table1 = np.zeros((124,11))
    table1 = pd.DataFrame(rankcol, columns=column_names1, index=row_names1)
    #Add row and column headers for least regret for df1:
    table2 = np.zeros((122,13))
    table2 = pd.DataFrame(rankrow, columns=column_names2, index=row_names2)
    return table1, table2
#list operations:
ans = [sum_rank(i) for i in [a,b,c,d,e,f,g,h,j]]
print(ans)
#Variable ouput arrays (use ans[0][0][0] to query index).
#Syntax for internal array = A[start_index_row : stop_index_row, 
    #start_index_columnn : stop_index_column)]
ans_output = pd.DataFrame(ans)
ans_output.to_csv('./outputs/ans_output.csv')

##############################################################################
#Min scenarios and timesteps across all variables:
##############################################################################  
#Sum variables into one column:   
S_RCP0_B_all = ans[0][1].iloc[:,-2]
S_RCP45_B_all = ans[1][1].iloc[:,-2]
S_RCP85_B_all = ans[2][1].iloc[:,-2]
S_RCP45_def = ans[3][1].iloc[:,-2]
S_RCP45_def = ans[4][1].iloc[:,-2]
S_RCP85_bonds_all = ans[5][1].iloc[:,-2]
S_RCP85_rates_all = ans[6][1].iloc[:,-2]
S_RCP85_bonds_all = ans[7][1].iloc[:,-2]
S_RCP85_rates_all = ans[8][1].iloc[:,-2]
#Append scenarios into one matrix:
Scenarios = []
Scenarios.append((ans[0][1].iloc[:,-2]))
Scenarios.append((ans[1][1].iloc[:,-2]))
Scenarios.append((ans[2][1].iloc[:,-2]))
Scenarios.append((ans[3][1].iloc[:,-2]))
Scenarios.append((ans[4][1].iloc[:,-2]))
Scenarios.append((ans[5][1].iloc[:,-2]))
Scenarios.append((ans[6][1].iloc[:,-2]))
Scenarios.append((ans[7][1].iloc[:,-2]))
Scenarios.append((ans[8][1].iloc[:,-2]))
Scenarios = np.array(Scenarios)
Scenarios.flatten()
ScenariosT = np.transpose(Scenarios)
#print('ScenariosT:', ScenariosT)
#print(ScenariosT.shape)
#Create least-regret matrix on scenarios and timesteps:  
def sum_rank2(j):
    s = np.array(j)
    t = np.min(s,axis=0)
    u = np.min(s,axis=1)
    tdif = s-t
    udif = s-u[:,None]
    add_zeros2 = np.zeros((2,1))
    #find the sum of the rows and columns for the difference arrays:
    sums = np.sum(tdif,axis=0)
    print('sums',sums)
    sumt = np.sum(udif,axis=1)
    print('sumt',sumt)
    sumv = np.sum(udif,axis=0)
    print('sumv',sumv)
    sumw = np.sum(tdif,axis=1)
    print('sumw',sumw)
    sums_reshape = np.append([sums],[add_zeros2])
    sumt_reshape = np.reshape(sumt,(121,1))
    sumw_reshape = np.reshape(sumw,(121,1))
    sumw_reshape2 = np.zeros((122,1))
    sumw_reshape2 = np.append([sumw_reshape],[add_zeros2])
    sumv_reshape = np.append([sumv],[add_zeros2])
    #append the scenario array with the column sums:
    sumcolj = np.zeros((121,10))
    sumcolj = np.append([tdif],[sumw])
    sumcolj.shape = (121,10)
     #rank columns:
    orderj = sums.argsort()
    orderj2 = sumv.argsort()
    rankj = orderj.argsort()
    rankj2 = orderj2.argsort()
    rankj2_reshape = np.append([rankj2],[add_zeros2])
    #append the array with row sums
    sumrowj = np.zeros((121,10))
    sumrowj = np.hstack((udif,sumt_reshape))
    sumrowj2 = np.zeros((121,10))
    sumrowj2 = np.hstack((tdif,sumw_reshape))
    #rank rows
    order1j = sumt.argsort()
    rank1j = order1j.argsort()
    rank1j = np.reshape(rank1j,(121,1))
    order2j = sumv.argsort()
    rank2j = order2j.argsort()
    rank2j = np.reshape(rank2j,(9,1))
    #append the array with row sums
    rankrowj = np.zeros((121,11))
    rankrowj = np.hstack((sumrowj,rank1j))
    rankrowj2 = np.zeros((122,11))
    rankrowj2 = np.append([rankrowj],[sumv_reshape])  
    #Add alternative summation of rows and columns:
    sumcolj2 = np.zeros((121,11))
    sumcolj2 = np.append([sumcolj],[rank1j])
    rankcolj2 = np.zeros((122,11))
    rankcolj2 = np.append(sumcolj2,sums_reshape)
    rankcolj3 = np.zeros((123,11))
    rankcolj3 = np.append(rankcolj2,rankj2_reshape)
    rankcolj3_reshape = np.reshape(rankcolj3,(123,11))
    #Add alternative summation of rows and columns:   
    rank2j2 = np.append([rank2j],[add_zeros2])    
    rankrowj3 = np.zeros((123,11))
    rankrowj3 = np.append([rankrowj2],[rank2j2])
    rankrowj3_reshape = np.reshape(rankrowj3,(123,11))   
    #Add row and column headers for least regret for df0:
    table0 = np.zeros((123,11))
    table0 = pd.DataFrame(rankcolj3_reshape, columns=column_names4, index=row_names1)
    #Add row and column headers for least regret for df1:
    table1 = np.zeros((123,11))
    table1 = pd.DataFrame(rankrowj3_reshape, columns=column_names4, index=row_names1)
    return table0, table1
#list operations:
ans1 = [sum_rank2(j) for j in [ScenariosT]]
print('Scenarios all ans1', ans1)

##############################################################################
#############   RESULTS
##############################################################################

#Best timestep across all scenarios:
#############################################################################  
print(ans1[0][0][:-2])
ans1_output = pd.DataFrame(ans1[0][0][:])
ans2 = pd.DataFrame(ans1[0][1][:])
print('ans2', ans2)
ans2.to_csv('./outputs/ans2_output.csv')
#Transpose list for plot on first dataframe:
ans3 = pd.DataFrame.transpose(ans1[0][0].iloc[:-2,0:9])
#Transpose list for plot on second dataframe:
ans4 = pd.DataFrame.transpose(ans1[0][1].iloc[:-2,0:9])
#Min scenario value by index
print('Scenario rank:', ans1[0][0][-1:])
ans1[0][0][-1:].to_csv('./outputs/rank_output.csv')

#Scenario boxplot for all timesteps (ans4):
bp0=plt.figure(figsize=(12, 10), dpi= 80, facecolor='lightgrey', edgecolor='k')
bp0=plt.boxplot(ans4,patch_artist=True, showmeans=True)
#fill with colors
#colors = ['lightcyan', 'paleturquoise', 'aquamarine', 'turquoise', 'mediumturquoise',
#          'lightseagreen', 'teal', 'darkslategrey', 'black']
colors = ['blue', 'black', 'lightgrey', 'mediumorchid', 'mediumorchid',
          'sandybrown', 'yellowgreen', 'sandybrown', 'yellowgreen']
for patch, color in zip(bp0['boxes'], colors):
    patch.set_facecolor(color)
plt.yticks(fontsize=14)
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9], ['1. No SLR', '2. Baseline', 
           '3. Worst-case baseline', '4. Defend', '5. Worst-case defend', 
           '6. Managed retreat bonds', '7. Managed retreat rates', 
          '8. Worst-case bonds', '9. Worst-case rates'], 
    rotation=30, fontsize=14)
#plt.title('BOXPLOT OF SCENARIOS ACROSS ALL TIMESTEPS', fontsize=16, color='navy')
plt.xlabel('Scenario', fontsize=16, color='black')
plt.ylabel('Minimum regret (Deviation from ideal scenario)', fontsize=16, color='black')
plt.grid(color='lightgrey', linestyle='-', linewidth=0.3)
plt.tight_layout()
plt.show()

##############################################################################
#Best scenario across all timesteps
##############################################################################
#Min timestep (index) value by scenarios:
print('Timestep rank:', ans1[0][0].iloc[:-2,-2])

#lineplot by scenario for all timesteps  
plt.figure(figsize=(20, 10), dpi= 80, facecolor='lightgrey', edgecolor='k')
ysmoothed = gaussian_filter1d(ans1[0][1].iloc[:-2,:-2], sigma=2.3)
plt.plot(ysmoothed)
np.savetxt('./outputs/ysmoothed_output.csv', ysmoothed, delimiter=",")


#plt.plot(ans1[0][1].iloc[:-2,:-2])
print(ans1[0][1].iloc[:-2,:-2])
plt.yticks(fontsize=14)
plt.xticks([0, 20, 40, 60, 80, 100, 120], ['2020', '2025', '2030', '2035', '2040', '2045', '2050']
    , rotation=30, fontsize=14)
plt.title('LINE GRAPH OF SCENARIOS BY TIMESTEP', fontsize=16, color='black')
plt.xlabel('YEAR', fontsize=16, color='black')
plt.ylabel('MINIMUM REGRET (Deviation from ideal scenario)', fontsize=16, color='black')
lg = plt.legend(labels, title='SCENARIOS', fontsize=16)
title = lg.get_title()
title.set_fontsize(14)
plt.grid(color='lightgrey', linestyle='-', linewidth=0.3)
plt.show()
##############################################################################
##############################################################################
