import pandas as pd

print("="*60)
print("🔍 DEBUGGING LIAR DATASET LABELS")
print("="*60)

# Load data
df = pd.read_csv('dataset/train.tsv', sep='\t', header=None)
df.columns = ['id', 'statement', 'label', 'subject', 'speaker', 
              'job', 'state', 'party', 'barely_true', 'false', 
              'half_true', 'mostly_true', 'pants_fire', 'context']

print(f"✅ Loaded {len(df)} rows")

# Show unique labels
print("\n📊 Unique labels in dataset:")
print(df['label'].unique())

# Show first 10 rows with labels
print("\n📄 First 10 rows:")
print(df[['statement', 'label']].head(10))

# Check label distribution
print("\n📊 Label distribution:")
print(df['label'].value_counts())

# See what labels exist
print("\n🔍 All unique label values:")
print(df['label'].unique().tolist())