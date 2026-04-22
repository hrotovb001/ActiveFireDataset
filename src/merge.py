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
        if arr.shape != (12, 45, 45):
            print(f"Warning: {file_name} has shape {arr.shape}, expected (45,45). Skipping.")
            continue
        if base not in pairs:
            pairs[base] = {}
        pairs[base]['input'] = arr
        print(f"Found input for {base}")
        
    elif '_output.npy' in file_name:
        base = file_name.replace('_output.npy', '')
        arr = np.load(file_path)
        if arr.shape != (45, 45):
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
    merged_input = np.nan_to_num(np.stack(input_arrays, axis=0))
    merged_output = np.nan_to_num(np.stack(output_arrays, axis=0))

    np.save('/mnt/e/Dataset/x.npy', merged_input)
    np.save('/mnt/e/Dataset/y.npy', merged_output)

    train_len = int(0.8 * len(merged_input))
    merged_input = merged_input[:train_len]
    merged_output = merged_output[:train_len]

    merged_input[:, 0] = np.log(merged_input[:, 0] + 1)
    merged_input[:, 1] = merged_input[:, 1] ** (1/3)
    merged_input[:, 5] = merged_input[:, 5] ** (1/3)
    merged_input[:, 8] = np.sqrt(merged_input[:, 8])
    merged_input[:, 9] = np.log(merged_input[:, 9] + 1)
    merged_input[:, 10] = np.sqrt(merged_input[:, 10])

    c_min = merged_input[:, 1:].min(axis=(0, 2, 3), keepdims=True)
    c_max = merged_input[:, 1:].max(axis=(0, 2, 3), keepdims=True)
    factors = np.stack([c_min.squeeze(), c_max.squeeze()], axis=1)

    merged_output = np.log(merged_output + 1)

    frp = np.concatenate([merged_input[0], merged_output], axis=0)
    frp_min = frp.min()
    frp_max = frp.max()
    frp_factors = np.stack([frp_min, frp_max], axis=0)[np.newaxis, ...]

    factors = np.concatenate([frp_factors, factors], axis=0)
    print("factors:", factors)

    np.save('/mnt/e/Dataset/factors.npy', factors)
    
    print(f"\nSaved merged x.npy with shape {merged_input.shape}")
    print(f"Saved merged y.npy with shape {merged_output.shape}")
    print(f"Pairs processed (in alphabetical order): {complete_bases}")
else:
    print("\nNo complete input/output pairs found.")
