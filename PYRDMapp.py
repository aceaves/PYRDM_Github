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

import streamlit as st
import pandas as pd

# ----------------------------
# Load Data
# ----------------------------

st.subheader("Load your scenario files")

# Option to choose how to load data
option = st.selectbox(
    "Select how to load the data",
    ["Upload Your Own Data", "Load Data from GitHub Repo"]
)

# Initialize containers for data
uploaded_dfs = []
github_dfs = []

# Option 1: User uploads CSV files
if option == "Upload Your Own Data":
    uploaded_files = st.file_uploader("Choose CSV files", type=["csv"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            try:
                df = pd.read_csv(file)
                uploaded_dfs.append(df)
                st.success(f"Loaded {file.name}")
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")

# Option 2: Load from GitHub URLs
elif option == "Load Data from GitHub Repo":
    github_urls = [
        "https://raw.githubusercontent.com/aceaves/PYRDM_Github/main/outputs/ans_output.csv",
        "https://raw.githubusercontent.com/aceaves/PYRDM_Github/main/outputs/ans2_output.csv",
        "https://raw.githubusercontent.com/aceaves/PYRDM_Github/main/outputs/rank_output.csv",
        "https://raw.githubusercontent.com/aceaves/PYRDM_Github/main/outputs/ysmoothed_output.csv"
    ]
    github_file_names = [url.split('/')[-1] for url in github_urls]
    selected_files = st.multiselect("Select files from GitHub Repo", github_file_names, default=github_file_names)

    for file_name in selected_files:
        url = next((u for u in github_urls if u.endswith(file_name)), None)
        if url:
            try:
                df = pd.read_csv(url)
                github_dfs.append((file_name, df))
                st.success(f"Loaded {file_name}")
            except Exception as e:
                st.error(f"Failed to load {file_name}: {e}")

# Assign loaded GitHub DataFrames to specific variables for graphing
if github_dfs:
    for i, (name, df) in enumerate(github_dfs):
        if 'ans_output' in name:
            ans1 = df
        elif 'ans2_output' in name:
            ans2 = df
        elif 'rank_output' in name:
            ans3 = df
        elif 'ysmoothed_output' in name:
            ans4 = df

# Optional: Display some of the data
if option == "Load Data from GitHub Repo" and github_dfs:
    st.write("Sample from `ans1` (ans_output.csv):")
    st.dataframe(ans1.head())


######     RESULTS      ###########

##############################################################################
#Scenario boxplot for all timesteps (ans4): 
##############################################################################

# Check if ans4 exists
if 'ans4' in locals():
    fig1, ax1 = plt.subplots(figsize=(12, 10), dpi=80, facecolor='lightgrey', edgecolor='k')
    box = ax1.boxplot(ans4, patch_artist=True, showmeans=True)

    # Fill with colors
    colors = ['blue', 'black', 'lightgrey', 'mediumorchid', 'mediumorchid',
              'sandybrown', 'yellowgreen', 'sandybrown', 'yellowgreen']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    ax1.set_yticks(ax1.get_yticks())
    ax1.set_yticklabels([f"{y:.2f}" for y in ax1.get_yticks()], fontsize=14)

    ax1.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9])
    ax1.set_xticklabels([
        '1. No SLR', '2. Baseline', '3. Worst-case baseline',
        '4. Defend', '5. Worst-case defend',
        '6. Managed retreat bonds', '7. Managed retreat rates',
        '8. Worst-case bonds', '9. Worst-case rates'
    ], rotation=30, fontsize=14)

    ax1.set_xlabel('Scenario', fontsize=16, color='black')
    ax1.set_ylabel('Minimum regret (Deviation from ideal scenario)', fontsize=16, color='black')
    ax1.grid(color='lightgrey', linestyle='-', linewidth=0.3)

    st.pyplot(fig1)
else:
    st.warning("ans4 not loaded. Please load scenario data.")


##############################################################################
#Best scenario across all timesteps
##############################################################################
# Check if ans2 exists
if 'ans2' in locals():
    # Drop last two rows and columns if needed
    data = ans2.iloc[:-2, :-2]

    # Keep only numeric columns
    data_numeric = data.select_dtypes(include=[np.number])

    if data_numeric.empty:
        st.error("No numeric data available for smoothing.")
    else:
        # Convert to NumPy and smooth
        ysmoothed = gaussian_filter1d(data_numeric.to_numpy(), sigma=2.3, axis=0)

        # Plot
        fig2, ax2 = plt.subplots(figsize=(20, 10), dpi=80, facecolor='lightgrey', edgecolor='k')
        ax2.plot(ysmoothed)

        ax2.set_xticks([0, 20, 40, 60, 80, 100, 120])
        ax2.set_xticklabels(['2020', '2025', '2030', '2035', '2040', '2045', '2050'], rotation=30, fontsize=14)
        ax2.set_yticks(ax2.get_yticks())
        ax2.set_yticklabels([f"{y:.2f}" for y in ax2.get_yticks()], fontsize=14)

        ax2.set_title('LINE GRAPH OF SCENARIOS BY TIMESTEP', fontsize=16, color='black')
        ax2.set_xlabel('YEAR', fontsize=16, color='black')
        ax2.set_ylabel('MINIMUM REGRET (Deviation from ideal scenario)', fontsize=16, color='black')
        ax2.grid(color='lightgrey', linestyle='-', linewidth=0.3)

        scenario_labels = ['Scenario ' + str(i+1) for i in range(ysmoothed.shape[1])]
        lg = ax2.legend(scenario_labels, title='SCENARIOS', fontsize=16)
        lg.get_title().set_fontsize(14)

        st.pyplot(fig2)
else:
    st.warning("ans2 not loaded. Please load scenario data.")

##############################################################################
##############################################################################

