import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

print("="*60)
print("🧹 DATA PREPROCESSING")
print("="*60)

# Step 1: Load Dataset
print("\n📁 Loading dataset...")
df = pd.read_excel('dataset/indianlabels_normalized.xlsx')
print(f"✅ Loaded {len(df)} rows")

# Step 2: Filter English News Only
print("\n🗣️ Filtering English news only...")
df_english = df[df['language'] == 'en'].copy()
print(f"✅ English news: {len(df_english)} rows")

# Step 3: Check Class Distribution
print("\n📊 Class Distribution (English only):")
print(df_english['Label'].value_counts())
real_count = df_english[df_english['Label']==0].shape[0]
fake_count = df_english[df_english['Label']==1].shape[0]
print(f"Real News (0): {real_count} ({real_count/len(df_english)*100:.2f}%)")
print(f"Fake News (1): {fake_count} ({fake_count/len(df_english)*100:.2f}%)")

# Step 4: Text Cleaning Function
def clean_text(text):
    """
    Clean text by:
    1. Converting to lowercase
    2. Removing punctuation
    3. Removing numbers
    4. Removing extra spaces
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Step 5: Apply Cleaning
print("\n🧹 Cleaning text...")
df_english['clean_statement'] = df_english['Statement'].apply(clean_text)
print("✅ Text cleaning complete!")

# Step 6: Sample Output
print("\n📄 Sample cleaned text:")
sample_df = df_english[['Statement', 'clean_statement']].head(3)
for idx, row in sample_df.iterrows():
    print(f"\nOriginal: {row['Statement'][:100]}...")
    print(f"Cleaned:  {row['clean_statement'][:100]}...")

# Step 7: Save Cleaned Data
print("\n💾 Saving cleaned dataset...")
df_english.to_excel('dataset/cleaned_dataset.xlsx', index=False)
df_english.to_csv('dataset/cleaned_dataset.csv', index=False)
print("✅ Saved as 'dataset/cleaned_dataset.xlsx' and 'dataset/cleaned_dataset.csv'")

print("\n" + "="*60)
print("✅ PREPROCESSING COMPLETE!")
print("="*60)