import torch

data_path = "dataset/processed/test_100.pt"
try:
    data = torch.load(data_path, weights_only=False)
    print(f"Data type: {type(data)}")
    if isinstance(data, list):
        print(f"Length: {len(data)}")
        if len(data) > 0:
            print(f"First item type: {type(data[0])}")
            print(f"First item keys: {data[0].keys() if isinstance(data[0], dict) else 'Not a dict'}")
            if isinstance(data[0], dict):
                print(f"First item keys: {data[0].keys()}")
                if 'sequences' in data[0]:
                    print(f"First item 'sequences' type: {type(data[0]['sequences'])}")
                    if isinstance(data[0]['sequences'], dict):
                        print(f"First item 'sequences' keys: {data[0]['sequences'].keys()}")
    else:
        print("Data is not a list")

except Exception as e:
    print(f"Error: {e}")
