import json

JSON_PATH = "outputs/ffpp_test_183_253/preprocessing.json"

with open(JSON_PATH, "r") as f:
    data = json.load(f)

print("\nTOP-LEVEL KEYS:")
print(list(data.keys()))

print("\nFULL STRUCTURE OF TOP LEVEL:")
for key, value in data.items():
    print(f"\nKEY: {key}")
    print(f"TYPE: {type(value).__name__}")

    if isinstance(value, dict):
        print(f"SUB-KEYS: {list(value.keys())}")

    elif isinstance(value, list):
        print(f"LIST LENGTH: {len(value)}")