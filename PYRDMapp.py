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
            (lambda x: 75_000_000 <= x < 100_000_000, 6),
            (lambda x: 50_000_000 <= x < 75_000_000, 4),
            (lambda x: 25_000_000 <= x < 50_000_000, 2)
        ],
        'Property premium': [
            (lambda x: x >= 5000, 3),
            (lambda x: 0 <= x < 1000, 4),
            (lambda x: 1000 <= x < 2000, 0),
            (lambda x: 2000 <= x < 3000, 1),
            (lambda x: 3000 <= x < 5000, 2)
        ],
        'hhldconsumprt': [
            (lambda x: x < 1.24, 4),
            (lambda x: 1.24 <= x < 1.245, 3),
            (lambda x: 1.245 <= x < 1.25, 2),
            (lambda x: 1.25 <= x < 1.255, 1),
            (lambda x: x >= 1.255, 0)
        ],
        'unemploymentrt': [
            (lambda x: x < 0.044, 0),
            (lambda x: 0.044 <= x < 0.045, 1),
            (lambda x: 0.045 <= x < 0.046, 2),
            (lambda x: 0.046 <= x < 0.048, 3),
            (lambda x: x >= 0.048, 4)
        ],
        'CentralG Consumption': [
            (lambda x: x >= 2200, 0),
            (lambda x: 2000 <= x < 2200, 1),
            (lambda x: 1800 <= x < 2000, 2),
            (lambda x: 1400 <= x < 1800, 3),
            (lambda x: x < 1400, 4)
        ],
        'LocalG Consumption': [
            (lambda x: x >= 350, 0),
            (lambda x: 300 <= x < 350, 3),
            (lambda x: 250 <= x < 300, 2),
            (lambda x: 200 <= x < 250, 1),
            (lambda x: x < 200, 4)
        ],
        'dtotalvalueadded': [
            (lambda x: x >= 12500, 0),
            (lambda x: 11000 <= x < 12500, 2),
            (lambda x: 9500 <= x < 11000, 4),
            (lambda x: 8000 <= x < 9500, 6),
            (lambda x: x < 8000, 8)
        ],
        'totactualprod': [
            (lambda x: x >= 36000, 0),
            (lambda x: 32000 <= x < 36000, 1),
            (lambda x: 28000 <= x < 32000, 2),
            (lambda x: 24000 <= x < 28000, 3),
            (lambda x: x < 24000, 4)
        ],
        'Pinvestcc': [
            (lambda x: x <= 1.05, 0),
            (lambda x: 1.05 < x <= 1.2, 1),
            (lambda x: 1.2 < x <= 1.3, 2),
            (lambda x: 1.3 < x <= 1.4, 3),
            (lambda x: x > 1.4, 4)
        ],
        'Landuse ratio': [
            (lambda x: x >= 2, 0),
            (lambda x: 1.5 <= x < 2, 1),
            (lambda x: 1 <= x < 1.5, 2),
            (lambda x: 0.5 <= x < 1, 3),
            (lambda x: x < 0.5, 4)
        ]
    }

    for col, rules in transformations.items():
        if col in df.columns:
            for condition, score in rules:
                df.loc[condition(df[col]), col] = score

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
        df_transformed = transform_data(df)
        dfs.append(df_transformed)

    st.success(f"✅ Processed {len(dfs)} files.")

    # Option to download all outputs
    if st.button("Download All as ZIP (coming soon)"):
        st.warning("This feature is not yet implemented.")

    # Show a preview
    with st.expander("Preview First File (Transformed)"):
        st.dataframe(dfs[0])

    # Could add visualizations or comparisons between scenarios here!

else:
    st.info("Please upload your Excel files to begin.")




##############################################################################
##############################################################################
