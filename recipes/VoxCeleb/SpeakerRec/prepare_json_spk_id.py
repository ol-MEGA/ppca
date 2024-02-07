import argparse
import os
import json

if __name__ == "__main__":
    '''
    This code adds spk_id to LibriSpeech jsons
    JP 07.02.2024
    '''

    parser = argparse.ArgumentParser(description='Add spk_idd to jsons')
    parser.add_argument('--json_folder', '-j', type=str, help=' Path to the ASV anon json files')
    args = parser.parse_args()

    for r, d, f in os.walk(args.json_folder):
        for file in f: 
            if file.endswith('_processed.json'):
                json_path = os.path.join(r, file)
                with open(json_path, "r") as inp:
                    json_dict = json.load(inp)

                for attribute, value in json_dict.items():
                    value['spk_id'] = attribute.split("-")[0]
                
                with open(json_path, mode="w") as out:
                    json.dump(json_dict, out, indent=2)