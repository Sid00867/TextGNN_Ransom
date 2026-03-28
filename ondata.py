import pandas as pd
import numpy as np

# 1. Load the original Kaggle CSV
# Ensure the path is correct for your 'TextGNN_Ransom' folder
df = pd.read_csv('./dynamic_api_call_sequence_per_malware_100_0_306.csv')

# 2. Separate the two classes
malware_df = df[df['malware'] == 1]
benign_df = df[df['malware'] == 0]

print(f"Found {len(malware_df)} Malware and {len(benign_df)} Benign samples.")

# 3. CORRECTED UNDERSAMPLING:
# We sample from the MAJORITY (Malware) to match the MINORITY (Benign).
malware_undersampled = malware_df.sample(n=len(benign_df), random_state=42)

# 4. Combine them into one balanced dataset (~2,158 samples total)
balanced_df = pd.concat([malware_undersampled, benign_df])

# 5. Prepare the metadata (Format: ID \t Split \t Label)
# Shuffling ensures the model doesn't see all malware first
indices = balanced_df.index.values
np.random.seed(42)
np.random.shuffle(indices)

split_point = int(0.9 * len(indices))
train_indices = indices[:split_point]

metadata = []
for i in indices:
    split = "train" if i in train_indices else "test"
    # The label (0 or 1) is in the 'malware' column
    label = str(df.iloc[i]['malware'])
    metadata.append(f"{i}\t{split}\t{label}")

# Save to your data folder
with open('data/text_dataset/malware.txt', 'w') as f:
    f.write('\n'.join(metadata))

print(f"Success! Balanced file created with {len(benign_df)} samples per class.")