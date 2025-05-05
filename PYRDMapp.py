# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:24:21 2024

@author: ashto
"""
##############################################################################
##########  PYRDM CODE 3.0 CREATED BY ASHTON EAVES
##############################################################################

##############################################################################
##########  PYRDM CODE 3.0 CREATED BY ASHTON EAVES
##############################################################################

import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import os  # You'll likely still need this for path operations, though not for saving intermediate CSVs
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def load_data_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        try:
            df = pd.read_csv(io.StringIO(response.text))
            st.write(f"Successfully loaded: {url.split('/')[-1]}") # Confirmation message
            return df
        except Exception as e:
            st.error(f"Error reading CSV from {url}: {e}")
            return None
    else:
        st.error(f"Error loading data from {url}. Status code: {response.status_code}")
        return None
    
   

st.title("🌊 PYRDM App — managed-retreat.com")

# ----------------------------
# Define thresholds and scoring logic (using lambda functions)
# ----------------------------

def transform_data(df):
    transformations = {
        'Annual Expected Loss': [
            (lambda x: x < 25_000_000, 0),
            (lambda x: x >= 100_000_000, 8),
            (lambda x: (75_000_000 <= x) & (x < 100_000_000), 6),
            (lambda x: (50_000_000 <= x) & (x < 75_000_000), 4),
            (lambda x: (25_000_000 <= x) & (x < 50_000_000), 2)
        ],
        'Property premium': [
            (lambda x: x >= 5000, 3),
            (lambda x: (0 <= x) & (x < 1000), 4),
            (lambda x: (1000 <= x) & (x < 2000), 0),
            (lambda x: (2000 <= x) & (x < 3000), 1),
            (lambda x: (3000 <= x) & (x < 5000), 2)
        ],
        'hhldconsumprt': [
            (lambda x: x < 1.24, 4),
            (lambda x: x >= 1.255, 0),
            (lambda x: (1.25 <= x) & (x < 1.255), 1),
            (lambda x: (1.245 <= x) & (x < 1.25), 2),
            (lambda x: (1.24 <= x) & (x < 1.245), 3)
        ],
        'unemploymentrt': [
            (lambda x: x >= 0.048, 4),
            (lambda x: (0.044 <= x) & (x < 0.045), 1),
            (lambda x: (0.045 <= x) & (x < 0.046), 2),
            (lambda x: (0.046 <= x) & (x < 0.048), 3),
            (lambda x: x < 0.044, 0)
        ],
        'CentralG Consumption': [
            (lambda x: x >= 2200, 0),
            (lambda x: x < 1400, 4),
            (lambda x: (1800 <= x) & (x < 2000), 2),
            (lambda x: (1400 <= x) & (x < 1800), 3),
            (lambda x: (2000 <= x) & (x < 2200), 1)
        ],
        'LocalG Consumption': [
            (lambda x: x < 200, 4),
            (lambda x: x >= 350, 0),
            (lambda x: (250 <= x) & (x < 300), 2),
            (lambda x: (300 <= x) & (x < 350), 3),
            (lambda x: (200 <= x) & (x < 250), 1)
        ],
        'dtotalvalueadded': [
            (lambda x: x < 8000, 8),
            (lambda x: (8000 <= x) & (x < 9500), 6),
            (lambda x: (9500 <= x) & (x < 11000), 4),
            (lambda x: (11000 <= x) & (x < 12500), 2),
            (lambda x: x >= 12500, 0)
        ],
        'totactualprod': [
            (lambda x: x < 24000, 4),
            (lambda x: (24000 <= x) & (x < 28000), 3),
            (lambda x: (28000 <= x) & (x < 32000), 2),
            (lambda x: (32000 <= x) & (x < 36000), 1),
            (lambda x: x >= 36000, 0)
        ],
        'Pinvestcc': [
            (lambda x: x >= 1.4, 4),
            (lambda x: x <= 1.05, 0),
            (lambda x: (1.05 < x) & (x < 1.2), 1),
            (lambda x: (1.2 < x) & (x < 1.3), 2),
            (lambda x: (1.3 < x) & (x < 1.4), 3)
        ],
        'Landuse ratio': [
            (lambda x: x < 0.5, 4),
            (lambda x: (0.5 <= x) & (x < 1), 3),
            (lambda x: (1 <= x) & (x < 1.5), 2),
            (lambda x: (1.5 <= x) & (x < 2), 1),
            (lambda x: x >= 2, 0)
        ]
    }

    transformed_df = df.copy()
    for col, rules in transformations.items():
        if col in transformed_df.columns:
            for condition, value in rules:
                transformed_df.loc[condition(transformed_df[col]), col] = value
    return transformed_df

# ----------------------------
# Load Data
# ----------------------------

st.subheader("Load your scenario files")

# Option to choose how to load data
option = st.selectbox(
    "Select how to load the data",
    ["Upload Your Own Data", "Load Data from GitHub Repo"]
)

list_of_transformed_dfs = []
file_names = []

if option == "Upload Your Own Data":
    uploaded_files = st.file_uploader("Choose CSV files", type=["csv"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            try:
                df = pd.read_csv(file)
                transformed_df = transform_data(df)
                list_of_transformed_dfs.append(transformed_df)
                file_names.append(file.name)
                st.write(f"Processed: {file.name}")
                st.dataframe(transformed_df.head()) # Show a preview
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")

elif option == "Load Data from GitHub Repo":
    # List of GitHub URLs for CSV files
    github_urls = [
        "https://raw.githubusercontent.com/aceaves/PYRDM_Github/main/inputs/RCP0.csv",
        "https://raw.githubusercontent.com/aceaves/PYRDM_Github/main/inputs/RCP45.csv",
        "https://raw.githubusercontent.com/aceaves/PYRDM_Github/main/inputs/RCP85.csv",
        "https://raw.githubusercontent.com/aceaves/PYRDM_Github/main/inputs/RCP45def.csv",
        "https://raw.githubusercontent.com/aceaves/PYRDM_Github/main/inputs/RCP85def.csv",
        "https://raw.githubusercontent.com/aceaves/PYRDM_Github/main/inputs/RCP45bonds.csv",
        "https://raw.githubusercontent.com/aceaves/PYRDM_Github/main/inputs/RCP45rates.csv",
        "https://raw.githubusercontent.com/aceaves/PYRDM_Github/main/inputs/RCP85bonds.csv",
        "https://raw.githubusercontent.com/aceaves/PYRDM_Github/main/inputs/RCP85rates.csv"
    ]
    github_file_names = [url.split('/')[-1] for url in github_urls]
    selected_urls = st.multiselect("Select files from GitHub Repo", github_file_names, default=github_file_names)

    for url in github_urls:
        if url.split('/')[-1] in selected_urls:
            df = load_data_from_github(url)
            if df is not None:
                transformed_df = transform_data(df.copy()) # Apply transformation immediately
                list_of_transformed_dfs.append(transformed_df)
                file_names.append(url.split('/')[-1])
                st.write(f"Loaded and processed: {url.split('/')[-1]}")
                st.dataframe(transformed_df.head()) # Show a preview

# Create empty numpy arrays for later:
sum_rank = np.array([])
sum_rank2 = np.array([])


# Min Regret Analysis Section
if list_of_transformed_dfs:
    st.subheader("Min Regret Analysis")

    # Assign the loaded and transformed DataFrames to variables
    S1, S2, S3, S4, S5, S6, S7, S8, S9 = None, None, None, None, None, None, None, None, None
    try:
        S1, S2, S3, S4, S5, S6, S7, S8, S9 = list_of_transformed_dfs
    except ValueError as e:
        st.error(f"Error unpacking list_of_transformed_dfs: {e}. Length is {len(list_of_transformed_dfs)}")

    if all(df is not None for df in [S1, S2, S3, S4, S5, S6, S7, S8, S9]):
        #Loop through files, classify and index:
        list_of_dfs = [S1, S2, S3, S4, S5, S6, S7, S8, S9]

        # Delete index column
        a = S1.values
        b = S2.values
        c = S3.values
        d = S4.values
        e = S5.values
        f = S6.values
        g = S7.values
        h = S8.values
        j = S9.values
        st.write(f"Length of list_of_dfs: {len(list_of_dfs)}")

        try:
            ans = [sum_rank(i) for i in [a,b,c,d,e,f,g,h,j]]
            st.write(f"Shape of ans: {len(ans) if ans else None}, type of ans[0]: {type(ans[0]) if ans and ans[0] else None}")
            print(ans)
        except Exception as e:
            st.error(f"Error in sum_rank function: {e}")
            ans = None

        if ans:
            try:
                Scenarios = []
                # ... (appending to Scenarios)
                Scenarios = np.array(Scenarios)
                ScenariosT = np.transpose(Scenarios)
                st.write(f"Shape of ScenariosT: {ScenariosT.shape}")
                ans1 = [sum_rank2(j) for j in [ScenariosT]]
                st.write(f"Shape of ans1: {len(ans1) if ans1 else None}, type of ans1[0]: {type(ans1[0]) if ans1 and ans1[0] else None}")
                print('Scenarios all ans1', ans1)
            except Exception as e:
                st.error(f"Error in the Min Regret Analysis (after sum_rank): {e}")
                ans1 = None

            if ans1:
                # ... (plotting code with try-except blocks)
                pass

    else:
        st.warning("Not all DataFrames were loaded and transformed successfully. Min Regret Analysis will not proceed.")


        try:
            ans = [sum_rank(i) for i in [a,b,c,d,e,f,g,h,j]]
            st.write(f"Shape of ans: {len(ans) if ans else None}, type of ans[0]: {type(ans[0]) if ans and ans[0] else None}")
            print(ans)
        except Exception as e:
            st.error(f"Error in sum_rank function: {e}")
            ans = None

        if ans:
            # ... (rest of your Min Regret Analysis code, including sum_rank2 and plotting)
            try:
                Scenarios = []
                # ... (appending to Scenarios)
                Scenarios = np.array(Scenarios)
                ScenariosT = np.transpose(Scenarios)
                st.write(f"Shape of ScenariosT: {ScenariosT.shape}")
                ans1 = [sum_rank2(j) for j in [ScenariosT]]
                st.write(f"Shape of ans1: {len(ans1) if ans1 else None}, type of ans1[0]: {type(ans1[0]) if ans1 and ans1[0] else None}")
                print('Scenarios all ans1', ans1)
            except Exception as e:
                st.error(f"Error in the Min Regret Analysis (after sum_rank): {e}")
                ans1 = None

            if ans1:
                # ... (plotting code with try-except blocks)
                try:
                    bp0 = plt.figure(...)
                    # ...
                    plt.show()
                except Exception as e:
                    st.error(f"Error during boxplot generation: {e}")

                try:
                    plt.figure(...)
                    # ...
                    plt.show()
                except Exception as e:
                    st.error(f"Error during lineplot generation: {e}")


if list_of_transformed_dfs:
    st.subheader("Min Regret Analysis")

    # Assign the loaded and transformed DataFrames to variables (similar to your original code)
    S1, S2, S3, S4, S5, S6, S7, S8, S9 = list_of_transformed_dfs

    ##############################################################################
    #Loop through files, classify and index:
    ##############################################################################
    list_of_dfs = [S1, S2, S3, S4, S5, S6, S7, S8, S9]

    ##############################################################################
    # The conditional statements are now within the transform_data function
    ##############################################################################

    #############################################################################
    #Read outputs back in for min regret analysis: (Now reading from memory)
    #############################################################################
    # No need to read from disk anymore, we use the DataFrames in memory

    # Column and row names (assuming these are consistent in your CSVs)
    column_names1 = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9','V10'] # Adjust based on your actual columns
    row_names1  = ['2020','2020.25','2020.5','2020.75','2021','2021.25','2021.5','2021.75','2022',
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
    column_names2 = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9','V10' ,'sum','rank'] # Adjust
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
    num_rows, num_cols = p.shape

    q = np.min(p, axis=0)
    r = np.min(p, axis=1)
    cdif = p - q
    rdif = p - r[:, None]

    #find the sum of the rows and columns for the difference arrays:
    sumc = np.sum(cdif, axis=0)
    sumr = np.sum(rdif, axis=1)
    sumra = np.reshape(sumr, (num_rows, 1))

    #append the scenario array with the column sums:
    sumcol = np.zeros((num_rows + 1, num_cols))
    sumcol[:-1, :] = cdif  # Assign cdif to the first num_rows rows
    sumcol[-1, :] = sumc   # Assign sumc to the last row

    #rank columns:
    order0 = sumc.argsort()
    rank0 = order0.argsort()
    rankcol = np.zeros((num_rows + 2, num_cols))
    rankcol[:-2, :] = sumcol[:-1, :] # Assign sumcol (excluding last row)
    rankcol[-2, :] = sumcol[-1, :]  # Assign the sum of columns
    rankcol[-1, :] = rank0          # Assign the rank of columns

    #append the variable array with row sums:
    sumrow = np.zeros((num_rows, num_cols + 1))
    sumrow[:, :-1] = rdif # Assign rdif to all rows, excluding the last column
    sumrow[:, -1] = sumr  # Assign sumr to the last column

    #rank rows:
    order1 = sumr.argsort()
    rank1 = order1.argsort()
    rank1r = np.reshape(rank1, (num_rows, 1))
    rankrow = np.zeros((num_rows, num_cols + 2))
    rankrow[:, :-2] = sumrow[:, :-1] # Assign sumrow (excluding last two columns)
    rankrow[:, -2] = sumrow[:, -1]  # Assign the sum of rows
    rankrow[:, -1] = rank1r.flatten() # Assign the rank of rows

    #Add row and column headers for least regret for df0:
    table1 = pd.DataFrame(rankcol, columns=column_names1, index=row_names1 + ['sum', 'rank'])

    #Add row and column headers for least regret for df1:
    table2 = pd.DataFrame(rankrow, columns=column_names2, index=row_names2)

    return table1, table2
    

    #list operations:
    ans = [sum_rank(i) for i in [a,b,c,d,e,f,g,h,j]]
    print(ans)
    #Variable ouput arrays (use ans[0][0][0] to query index).
    #Syntax for internal array = A[start_index_row : stop_index_row, 
        #start_index_columnn : stop_index_column)]
    ans_output = pd.DataFrame(ans)

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
    num_rows, num_cols = s.shape

    t = np.min(s, axis=0)
    u = np.min(s, axis=1)
    tdif = s - t
    udif = s - u[:, None]
    add_zeros2 = np.zeros((2, 1))  # This still seems hardcoded, consider its purpose

    #find the sum of the rows and columns for the difference arrays:
    sums = np.sum(tdif, axis=0)
    print('sums shape:', sums.shape)
    sumt = np.sum(udif, axis=1)
    print('sumt shape:', sumt.shape)
    sumv = np.sum(udif, axis=0)
    print('sumv shape:', sumv.shape)
    sumw = np.sum(tdif, axis=1)
    print('sumw shape:', sumw.shape)

    sums_reshape = np.append([sums], [add_zeros2], axis=0) # Ensure correct appending
    sumt_reshape = np.reshape(sumt, (num_rows, 1))
    sumw_reshape = np.reshape(sumw, (num_rows, 1))
    sumw_reshape2 = np.zeros((num_rows + 1, 1))
    sumw_reshape2[:-1, :] = sumw_reshape
    sumw_reshape2[-1, :] = add_zeros2[0, 0] # Assign a single zero
    sumv_reshape = np.append([sumv], [add_zeros2], axis=0) # Ensure correct appending

    #append the scenario array with the column sums:
    sumcolj = np.zeros((num_rows, num_cols))
    sumcolj[:, :] = tdif # Assign directly
    sumcolj = np.vstack((sumcolj, sumw)) # Stack along rows

    #rank columns:
    orderj = sums.argsort()
    orderj2 = sumv.argsort()
    rankj = orderj.argsort()
    rankj2 = orderj2.argsort()
    rankj2_reshape = np.append([rankj2], [add_zeros2], axis=0) # Ensure correct appending

    #append the array with row sums
    sumrowj = np.zeros((num_rows, num_cols))
    sumrowj[:, :] = udif # Assign directly
    sumrowj = np.hstack((sumrowj, sumt_reshape)) # Stack along columns
    sumrowj2 = np.zeros((num_rows, num_cols))
    sumrowj2[:, :] = tdif # Assign directly
    sumrowj2 = np.hstack((sumrowj2, sumw_reshape)) # Stack along columns

    #rank rows
    order1j = sumt.argsort()
    rank1j = order1j.argsort()
    rank1j = np.reshape(rank1j, (num_rows, 1))
    order2j = sumv.argsort()
    rank2j = order2j.argsort()
    rank2j = np.reshape(rank2j, (num_cols -1, 1)) # Adjust based on sumv's length

    #append the array with row sums
    rankrowj = np.zeros((num_rows, num_cols + 1))
    rankrowj[:, :-1] = sumrowj[:, :-1]
    rankrowj[:, -1] = sumrowj[:, -1].flatten()
    rankrowj2 = np.zeros((num_rows + 1, num_cols + 1))
    rankrowj2[:-1, :] = rankrowj
    rankrowj2[-1, :-1] = sumv_reshape[:-1].flatten() # Assign sumv (excluding last)
    rankrowj2[-1, -1] = sumv_reshape[-1].flatten() # Assign the last element of sumv

    #Add alternative summation of rows and columns:
    sumcolj2 = np.zeros((num_rows, num_cols + 1))
    sumcolj2[:, :-1] = sumcolj[:-1, :]
    sumcolj2[:, -1:] = rank1j
    rankcolj2 = np.zeros((num_rows + 1, num_cols + 1))
    rankcolj2[:-1, :] = sumcolj2
    rankcolj2[-1, :] = sums_reshape.flatten()
    rankcolj3 = np.zeros((num_rows + 2, num_cols + 1))
    rankcolj3[:-1, :] = rankcolj2
    rankcolj3[-1, :-1] = rankj2_reshape[:-1].flatten() # Assign rankj2 (excluding last)
    rankcolj3[-1, -1] = rankj2_reshape[-1].flatten() # Assign last of rankj2
    rankcolj3_reshape = np.reshape(rankcolj3, (num_rows + 2, num_cols + 1))

    #Add alternative summation of rows and columns:
    rank2j2 = np.append([rank2j], [add_zeros2[:rank2j.shape[0]]], axis=0) # Adjust zeros
    rankrowj3 = np.zeros((num_rows + 1, num_cols + 1))
    rankrowj3[:-1, :] = rankrowj2
    rankrowj3[-1, :-1] = rank2j2[:-1].flatten() # Assign rank2j (excluding last)
    rankrowj3[-1, -1] = rank2j2[-1].flatten() # Assign last of rank2j
    rankrowj3_reshape = np.reshape(rankrowj3, (num_rows + 1, num_cols + 1))

    #Add row and column headers for least regret for df0:
    table0 = pd.DataFrame(rankcolj3_reshape, columns=column_names4, index=row_names1 + ['sum_col', 'rank_col'])

    #Add row and column headers for least regret for df1:
    table1 = pd.DataFrame(rankrowj3_reshape, columns=column_names4, index=row_names1 + ['sum_row'])

    return table0, table1

    ans = None  # Initialize ans to None
    try:
        ans = [sum_rank(i) for i in [a, b, c, d, e, f, g, h, j]]
        print(ans)
    except Exception as e:
        st.error(f"Error in sum_rank function: {e}")

    ScenariosT = None  # Initialize to None

    if ans:
        try:
            Scenarios = []
            for i in range(len(a)):
                Scenarios.append([a[i][0], b[i][0], c[i][0], d[i][0], e[i][0], f[i][0], g[i][0], h[i][0], j[i][0]])

            Scenarios = np.array(Scenarios)           # Shape: (N, 9)
            ScenariosT = np.transpose(Scenarios)      # Shape: (9, N)

            print(f"Shape of ScenariosT: {ScenariosT.shape}")
            print(f"First few rows of ScenariosT:\n{ScenariosT[:, :5]}")  # Print first 5 columns

            ans1 = [sum_rank2(ScenariosT)]
            print('Scenarios all ans1:', ans1)

        except Exception as e:
            st.error(f"Error in the Min Regret Analysis (after sum_rank): {e}")
            ans1 = None

            
    #list operations:
    ans1 = [sum_rank2(j) for j in [ScenariosT]]
    print('Scenarios all ans1', ans1)



    ##############################################################################
    #############   RESULTS
    ##############################################################################

    st.write("Scenarios shape:", ScenariosT.shape)
    st.write("ans1 output:", ans1)

    #Best timestep across all scenarios:
    #############################################################################  
    print(ans1[0][0][:-2])
    ans1_output = pd.DataFrame(ans1[0][0][:])
    ans2 = pd.DataFrame(ans1[0][1][:])
    print('ans2', ans2)
    #Transpose list for plot on first dataframe:
    ans3 = pd.DataFrame.transpose(ans1[0][0].iloc[:-2,0:9])
    #Transpose list for plot on second dataframe:
    ans4 = pd.DataFrame.transpose(ans1[0][1].iloc[:-2,0:9])
    #Min scenario value by index
    print('Scenario rank:', ans1[0][0][-1:])

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
    st.pyplot(bp0)

    ##############################################################################
    #Best scenario across all timesteps
    ##############################################################################
    #Min timestep (index) value by scenarios:
    print('Timestep rank:', ans1[0][0].iloc[:-2,-2])

    #lineplot by scenario for all timesteps  
    plt.figure(figsize=(20, 10), dpi= 80, facecolor='lightgrey', edgecolor='k')
    ysmoothed = gaussian_filter1d(ans1[0][1].iloc[:-2,:-2], sigma=2.3)
    plt.plot(ysmoothed)

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
    st.pyplot(ysmoothed)
    ##############################################################################
    ##############################################################################
