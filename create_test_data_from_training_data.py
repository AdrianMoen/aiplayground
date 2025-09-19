import json
import random
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <json_file>")
        sys.exit(1)

    json_file = sys.argv[1]

    # Load the JSON data
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        sys.exit(1)

    if not isinstance(data, list):
        print("JSON file must contain a list of objects.")
        sys.exit(1)

    # Extract inputs
    inputs = [[item.get("input"), item.get("output")] for item in data if "input" in item]

    if not inputs:
        print("No 'input' fields found in the JSON file.")
        sys.exit(1)

    # Randomly sample 30 inputs (or all if less than 30)
    sample_size = min(30, len(inputs))
    sampled_inputs = random.sample(inputs, sample_size)

    # Print to terminal
    for i, inp in enumerate(sampled_inputs, start=1):
        print(f"{inp}")

if __name__ == "__main__":
    main()
