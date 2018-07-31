import argparse
import json
import os
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("--key")
parser.add_argument("--val")
args = parser.parse_args()
if args.key:
    key_name = args.key

storage_path = os.path.join(tempfile.gettempdir(), 'storage.data')
if not os.path.isfile(storage_path):
    with open(storage_path, 'w') as f:
        pass
with open(storage_path, 'r') as f:
    json_dict = f.read()
    if json_dict:
        storage_dict = json.JSONDecoder().decode(json_dict)
    else:
        storage_dict = {}
if args.val:  # storing data
    key_value = args.val
    # update dictionary
    if key_name in storage_dict:
        storage_dict[key_name] += [key_value]
    else:
        storage_dict[key_name] = [key_value]
    # write dictionary to file
    with open(storage_path, 'w') as f:
        f.write(json.JSONEncoder().encode(storage_dict))
else:  # read data
    if key_name in storage_dict:
        print(', '.join(storage_dict[key_name]))
    else:
        print('')
