# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:24:21 2024

@author: ashto
"""
##############################################################################
##########  PYRDM CODE 3.0 CREATED BY ASHTON EAVES
##############################################################################

import streamlit as st
import pandas as pd
import numpy as np
import pyarrow as pa
import requests
import io

# Function to load data from GitHub (or any predefined URL)
def load_data_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(io.StringIO(response.text))
    else:
        st.error(f"Error loading data from {url}")
        return None
    

st.title("🌊 PYRDM App — managed-retreat.com")

# ----------------------------
# Define thresholds and scoring logic
# ----------------------------

def transform_data(df):
    # Dictionary of transformations
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
            (lambda x: (1.24 <= x) & (x < 1.245), 3),
            (lambda x: (1.245 <= x) & (x < 1.25), 2),
            (lambda x: (1.25 <= x) & (x < 1.255), 1),
            (lambda x: x >= 1.255, 0)
        ],
        'unemploymentrt': [
            (lambda x: x < 0.044, 0),
            (lambda x: (0.044 <= x) & (x < 0.045), 1),
            (lambda x: (0.045 <= x) & (x < 0.046), 2),
            (lambda x: (0.046 <= x) & (x < 0.048), 3),
            (lambda x: x >= 0.048, 4)
        ],
        'CentralG Consumption': [
            (lambda x: x >= 2200, 0),
            (lambda x: (2000 <= x) & (x < 2200), 1),
            (lambda x: (1800 <= x) & (x < 2000), 2),
            (lambda x: (1400 <= x) & (x < 1800), 3),
            (lambda x: x < 1400, 4)
        ],
        'LocalG Consumption': [
            (lambda x: x >= 350, 0),
            (lambda x: (300 <= x) & (x < 350), 3),
            (lambda x: (250 <= x) & (x < 300), 2),
            (lambda x: (200 <= x) & (x < 250), 1),
            (lambda x: x < 200, 4)
        ],
        'dtotalvalueadded': [
            (lambda x: x >= 12500, 0),
            (lambda x: (11000 <= x) & (x < 12500), 2),
            (lambda x: (9500 <= x) & (x < 11000), 4),
            (lambda x: (8000 <= x) & (x < 9500), 6),
            (lambda x: x < 8000, 8)
        ],
        'totactualprod': [
            (lambda x: x >= 36000, 0),
            (lambda x: (32000 <= x) & (x < 36000), 1),
            (lambda x: (28000 <= x) & (x < 32000), 2),
            (lambda x: (24000 <= x) & (x < 28000), 3),
            (lambda x: x < 24000, 4)
        ],
        'Pinvestcc': [
            (lambda x: x <= 1.05, 0),
            (lambda x: (1.05 < x) & (x <= 1.2), 1),
            (lambda x: (1.2 < x) & (x <= 1.3), 2),
            (lambda x: (1.3 < x) & (x <= 1.4), 3),
            (lambda x: x > 1.4, 4)
        ],
        'Landuse ratio': [
            (lambda x: x >= 2, 0),
            (lambda x: (1.5 <= x) & (x < 2), 1),
            (lambda x: (1 <= x) & (x < 1.5), 2),
            (lambda x: (0.5 <= x) & (x < 1), 3),
            (lambda x: x < 0.5, 4)
        ]
    }
   
    for col, rules in transformations.items():
        if col in df.columns:
            new_col = col + '_transformed'
            df[new_col] = pd.Series([np.nan] * len(df), dtype="float64")  # Use np.nan for compatibility with float64
            for condition, value in rules:
                mask = condition(df[col])
                df.loc[mask, new_col] = float(value)  # Explicitly cast to float
    return df

# ----------------------------
# Upload Files
# ----------------------------

st.subheader("Upload your scenario files")

# Option to choose how to load data
option = st.selectbox(
    "Select how to load the data",
    ["Upload Your Own Data", "Load Data from GitHub Repo"]
)

# Load data based on selection
if option == "Upload Your Own Data":
    # File uploader for CSV or Excel files
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        # Handle the file based on its type (CSV or Excel)
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)

        st.write("Data loaded successfully!")
        st.dataframe(df)

        # Apply transformations
        df_transformed = transform_data(df)
        st.write("Transformed Data")
        st.dataframe(df_transformed)

elif option == "Load Data from GitHub Repo":
# List of GitHub URLs
    github_urls = [
        "https://raw.githubusercontent.com/aceaves/PYRDM_Github/main/inputs/RCP0.xlsm",
        "https://raw.githubusercontent.com/aceaves/PYRDM_Github/main/inputs/RCP45.xlsm",
        "https://raw.githubusercontent.com/aceaves/PYRDM_Github/main/inputs/RCP45bonds.xlsm",
        "https://raw.githubusercontent.com/aceaves/PYRDM_Github/main/inputs/RCP45def.xlsm",
        "https://raw.githubusercontent.com/aceaves/PYRDM_Github/main/inputs/RCP45rates.xlsm",
        "https://raw.githubusercontent.com/aceaves/PYRDM_Github/main/inputs/RCP85.xlsm",
        "https://raw.githubusercontent.com/aceaves/PYRDM_Github/main/inputs/RCP85bonds.xlsm",
        "https://raw.githubusercontent.com/aceaves/PYRDM_Github/main/inputs/RCP85def.xlsm",
        "https://raw.githubusercontent.com/aceaves/PYRDM_Github/main/inputs/RCP85rates.xlsm"
    ]

# Allow the user to choose which file to load from the list
    selected_url = st.selectbox("Select the file from GitHub Repo", github_urls)

    df = load_data_from_github(selected_url)
    if df is not None:
        st.write(f"Data loaded from GitHub: {selected_url.split('/')[-1]}")
        st.dataframe(df)
        
        # Apply transformations
        df_transformed = transform_data(df)
        st.write("Transformed Data")
        st.dataframe(df_transformed)

# Option to upload multiple files and process them
uploaded_files = st.file_uploader("Upload multiple Excel files", type="xlsx", accept_multiple_files=True)

if uploaded_files:
    dfs = []
    for file in uploaded_files:
        df = pd.read_excel(file)
        st.write(f"**{file.name}** loaded with {df.shape[0]} rows.")
        
        # Explicitly convert all columns to float64, ignoring errors
        df = df.apply(pd.to_numeric, errors='coerce', axis=0)

        # Check for columns that may have an unsupported dtype (e.g., datetime)
        for col in df.columns:
            if df[col].dtype == 'object':  # This includes datetime or string columns
                df[col] = df[col].astype('str')  # Convert to string if not numeric

        # Ensure that all columns are float64 where possible
        df = df.astype('float64', errors='ignore')

        # Transform data according to the rules
        df_transformed = transform_data(df)
        dfs.append(df_transformed)

    st.success(f"✅ Processed {len(dfs)} files.")

    # Show a preview
    with st.expander("Preview First File (Transformed)"):
        st.dataframe(dfs[0])

else:
    st.info("Please upload your Excel files to begin.")
    
    
##############################################################################
##############################################################################
