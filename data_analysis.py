import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("="*60)
print("📊 FAKE NEWS DATASET ANALYSIS")
print("="*60)

# Load Excel File
print("\n📁 Loading Excel dataset...")
df = pd.read_excel('dataset/indianlabels_normalized.xlsx')
print("✅ Dataset loaded successfully!")

# Basic Information
print(f"\n🔍 Dataset Information:")
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"Column names: {df.columns.tolist()}")

# First 5 rows
print("\n📄 First 5 rows:")
print(df.head())

# Data types
print("\n📋 Data types:")
print(df.dtypes)

# Check for missing values
print("\n❓ Missing values:")
print(df.isnull().sum())

# Check class distribution
if 'label' in df.columns:
    print("\n📊 Class Distribution (Label):")
    print(df['label'].value_counts())
    real_count = df[df['label']==0].shape[0]
    fake_count = df[df['label']==1].shape[0]
    print(f"Real News (0): {real_count} ({real_count/len(df)*100:.2f}%)")
    print(f"Fake News (1): {fake_count} ({fake_count/len(df)*100:.2f}%)")
else:
    print("\n⚠️ 'label' column not found!")
    print(f"Available columns: {df.columns.tolist()}")

print("\n" + "="*60)
print("✅ DATA ANALYSIS COMPLETE!")
print("="*60)