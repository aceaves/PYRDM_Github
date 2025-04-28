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
import os
import pyarrow as pa

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
    return transformations

    for col, rules in transformations.items():
        if col in df.columns:
            new_col = col + '_transformed'
            df[new_col] = pd.Series([pd.NA] * len(df), dtype="float64")  # use float64
            for condition, value in rules:
                mask = condition(df[col])
                df.loc[mask, new_col] = float(value)  # explicitly cast to float
    return df

# ----------------------------
# Upload Files
# ----------------------------

st.subheader("Upload your scenario files (Excel)")

uploaded_files = st.file_uploader(
    "Upload multiple Excel files",
    type=["xlsm", "xlsx"],
    accept_multiple_files=True
)

if uploaded_files:
    dfs = []
    for file in uploaded_files:
        df = pd.read_excel(file)
        st.write(f"**{file.name}** loaded with {df.shape[0]} rows.")
        
        # Convert all columns (except the header row) to float
        df = df.apply(pd.to_numeric, errors='coerce')  # This ensures all numeric data is coerced into float

        df_transformed = transform_data(df)
        dfs.append(df_transformed)

    st.success(f"✅ Processed {len(dfs)} files.")

    # Option to download all outputs
    if st.button("Download All as ZIP (coming soon)"):
        st.warning("This feature is not yet implemented.")

    # Show a preview
    with st.expander("Preview First File (Transformed)"):
        st.dataframe(dfs[0])

else:
    st.info("Please upload your Excel files to begin.")

##############################################################################
##############################################################################
