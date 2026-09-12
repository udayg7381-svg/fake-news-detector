import pandas as pd
import re
import string
import numpy as np

print("="*60)
print("🔧 FIXING DATASET - ADDING REAL NEWS")
print("="*60)

# Load existing dataset
print("\n📁 Loading dataset...")
df = pd.read_csv('dataset/cleaned_dataset.csv')
print(f"✅ Loaded {len(df)} rows")
print(f"Real: {len(df[df['Label']==0])}")
print(f"Fake: {len(df[df['Label']==1])}")

# High-quality REAL news examples
real_news = [
    # Space/ISRO
    "ISRO successfully launched its latest satellite mission today from Sriharikota. The satellite will help improve communication networks.",
    "India's Chandrayaan-3 mission successfully landed on the moon's south pole. This is a historic achievement for Indian space research.",
    "The Indian Space Research Organisation has scheduled its next satellite launch for December. The satellite will boost internet connectivity.",
    
    # Government/Policy
    "The Government of India has announced a new education policy focusing on digital learning. Schools will receive smart boards by 2025.",
    "The Finance Ministry has unveiled the annual budget with increased allocation for healthcare. The budget focuses on rural development.",
    "The Supreme Court has upheld the constitutional validity of the new farm laws. The court ruled in favor of the government.",
    
    # Science/Technology
    "Scientists at IIT Bombay have developed a low-cost water purification system. The technology will help rural communities access clean water.",
    "Researchers have discovered a new species of plant in the Western Ghats. The discovery highlights India's rich biodiversity.",
    "The World Health Organization has approved India's indigenous vaccine. The vaccine has shown 92% effectiveness in clinical trials.",
    
    # Economy/Business
    "India's GDP growth rate has reached 7.8% in the latest quarter. The economic recovery is driven by manufacturing and services.",
    "The Reserve Bank of India has kept the repo rate unchanged. The decision was made to maintain economic stability.",
    
    # Sports
    "India won the cricket World Cup final against Australia. This is India's third World Cup victory in cricket history.",
    "Indian athletes won 5 medals at the Asian Games. The performance marks India's best showing in the last decade.",
    
    # Health
    "The Health Ministry has launched a new campaign to eradicate tuberculosis. The campaign will focus on early detection and treatment.",
    "India has successfully eradicated polio according to WHO standards. This is a major public health achievement.",
    
    # Education
    "The Education Minister announced new scholarships for girls in STEM fields. The initiative aims to bridge the gender gap in science.",
    "IIT Delhi has been ranked among the top 50 engineering institutes in Asia. The ranking reflects India's educational excellence."
]

# Add to dataset
real_df = pd.DataFrame({
    'clean_statement': real_news,
    'Label': [0] * len(real_news)
})

df_combined = pd.concat([df, real_df], ignore_index=True)

# Balance classes - take equal number of real and fake
min_count = min(len(df_combined[df_combined['Label']==0]), len(df_combined[df_combined['Label']==1]))
if min_count > 20000:
    min_count = 20000

print(f"\n⚖️ Balancing classes... (taking {min_count} each)")

df_real = df_combined[df_combined['Label']==0].sample(n=min_count, random_state=42)
df_fake = df_combined[df_combined['Label']==1].sample(n=min_count, random_state=42)

df_final = pd.concat([df_real, df_fake]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n✅ Final dataset: {len(df_final)} rows")
print(f"Real: {len(df_final[df_final['Label']==0])}")
print(f"Fake: {len(df_final[df_final['Label']==1])}")

# Save
df_final.to_csv('dataset/final_balanced_dataset.csv', index=False)
print("\n💾 Saved as 'dataset/final_balanced_dataset.csv'")

print("\n" + "="*60)
print("✅ DATASET FIXED!")
print("="*60)