from datasets.loader import load_dataset, list_available_datasets

print("Available datasets:", list_available_datasets())

try:
    X, y = load_dataset('DrivFace')
    print(f"✓ Dataset loaded successfully!")
    print(f"  - Feature matrix shape: {X.shape}")
    print(f"  - Labels shape: {y.shape}")
    print(f"  - Unique labels: {sorted(set(y))}")
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    import traceback
    traceback.print_exc()
