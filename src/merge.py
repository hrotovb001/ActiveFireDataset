import numpy as np
import os
import sys

# Get directory from command line or use current directory
data_dir = sys.argv[1] if len(sys.argv) > 1 else '.'

# Dictionary to store pairs
pairs = {}

# Scan all .npy files
for file_name in os.listdir(data_dir):
    if not file_name.endswith('.npy'):
        continue
    
    file_path = os.path.join(data_dir, file_name)
    
    if '_input.npy' in file_name:
        base = file_name.replace('_input.npy', '')
        arr = np.load(file_path)
        if arr.shape != (45, 45):
            print(f"Warning: {file_name} has shape {arr.shape}, expected (45,45). Skipping.")
            continue
        if base not in pairs:
            pairs[base] = {}
        pairs[base]['input'] = arr
        print(f"Found input for {base}")
        
    elif '_output.npy' in file_name:
        base = file_name.replace('_output.npy', '')
        arr = np.load(file_path)
        if arr.shape != (12, 45, 45):
            print(f"Warning: {file_name} has shape {arr.shape}, expected (12,45,45). Skipping.")
            continue
        if base not in pairs:
            pairs[base] = {}
        pairs[base]['output'] = arr
        print(f"Found output for {base}")

# Check for complete pairs and collect in sorted order
complete_bases = []
input_arrays = []
output_arrays = []

for base in sorted(pairs.keys()):
    if 'input' in pairs[base] and 'output' in pairs[base]:
        complete_bases.append(base)
        input_arrays.append(pairs[base]['input'])
        output_arrays.append(pairs[base]['output'])
    else:
        missing = 'input' if 'input' not in pairs[base] else 'output'
        print(f"Skipping incomplete pair for '{base}': missing {missing}")

if complete_bases:
    # Stack along new first dimension
    merged_input = np.stack(input_arrays, axis=0)   # shape (N, 45, 45)
    merged_output = np.stack(output_arrays, axis=0) # shape (N, 12, 45, 45)
    
    np.save('input.npy', merged_input)
    np.save('output.npy', merged_output)
    
    print(f"\nSaved merged input.npy with shape {merged_input.shape}")
    print(f"Saved merged output.npy with shape {merged_output.shape}")
    print(f"Pairs processed (in alphabetical order): {complete_bases}")
else:
    print("\nNo complete input/output pairs found.")
