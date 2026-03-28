import pandas as pd

# Load your downloaded CSV
df = pd.read_csv('./dynamic_api_call_sequence_per_malware_100_0_306.csv')

# Drop the hash column and join all API IDs with spaces
sequences = df.drop('hash', axis=1).astype(str).agg(' '.join, axis=1)

# Save to your corpus directory
with open('data/text_dataset/corpus/malware.txt', 'w') as f:
    for seq in sequences:
        f.write(seq + '\n')