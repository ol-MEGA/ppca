import json
import random

random.seed(1234)

json_file = '/path/to/jsons/LibriSpeech/train_360.json'

with open(json_file, "r") as f:
    json_in = json.load(f)

split_ratio = {"dev": 10, "train": 90}
num_utt = len(json_in)
num_utt_dev = int(num_utt / split_ratio["dev"])

# dev split
json_dev = {}
utt_id_dev = random.sample(list(json_in.keys()), k=num_utt_dev)
for utt_id in utt_id_dev:
    json_dev[utt_id] = json_in[utt_id]
    json_in.pop(utt_id)

with open(json_file.replace(".", "_{}.".format(split_ratio["dev"])), "w") as out:
    json.dump(json_dev, out, indent=2)

# train split
with open(json_file.replace(".", "_{}.".format(split_ratio["train"])), "w") as out:
    json.dump(json_in, out, indent=2)