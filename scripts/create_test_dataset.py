import torch
import os

data_path = "dataset/processed/markov_sequences.pt"
output_path = "dataset/processed/test_100.pt"

if not os.path.exists(data_path):
    print(f"Error: {data_path} not found.")
    exit(1)

try:
    data = torch.load(data_path, weights_only=False)
    print(f"Loaded data type: {type(data)}")
    
    sequences = []
    if isinstance(data, dict) and 'sequences' in data:
        sequences = data['sequences']
    elif isinstance(data, list):
        sequences = data
    else:
        print("Unknown data structure")
        exit(1)
        
    print(f"Total sequences: {len(sequences)}")
    
    subset = sequences[:100]
    print(f"Subset size: {len(subset)}")
    
    torch.save(subset, output_path)
    print(f"Saved subset to {output_path}")

except Exception as e:
    print(f"Error: {e}")
