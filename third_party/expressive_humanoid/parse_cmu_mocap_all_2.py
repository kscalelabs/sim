import pandas as pd
from ruamel.yaml import YAML

path = "./data/configs/"

def find_entries(word, column):
    subset = column[column.str.contains(word, na=False, case=False)]
    num_entries = len(subset)
    num_unique_entries = subset.nunique()
    indices = subset.index.tolist()
    unique_entries = subset.unique()
    print(f"Found {num_entries} entries containing '{word}', {num_unique_entries} unique entries.")
    return num_entries, num_unique_entries, indices, unique_entries

def save_yaml(motion_ids, motion_descriptions, name):
    yaml = YAML()
    yaml.preserve_quotes = True
    motions_data = {"root": "../retarget_npy/"}
    for i, motion_id in enumerate(motion_ids):
        descrp = motion_descriptions[i].replace("/", ", ").replace("\\", ", ")
        motions_data[motion_id] = {
            "trim_beg": -1,
            "trim_end": -1,
            "weight": 1.0,
            "description": f"{descrp}",
            "difficulty": 4
        }
    final_structure = {"motions": motions_data}
    with open(path + f'motions_autogen_debug_{name}.yaml', 'w') as file:
        yaml.dump(final_structure, file)
    print('YAML file created with the specified structure.')

file_path = path + "cmu-mocap-index-spreadsheet.xls"
data = pd.read_excel(file_path)
description_column = data.iloc[:, 1]
motion_id_column = data.iloc[:, 0]

forbidden_words = ["ladder", "suitcase", "uneven", "terrain", "stair", "stairway", "stairwell", "clean", "box", "climb", "backflip", "handstand", "sit", "hang"]
target_words = ["walk", "navigate", "basketball", "dance", "punch", "fight", "push", "pull", "throw", "catch", "crawl", "wave", "high five", "hug", "drink", "wash", "signal", "balance", "strech", "leg", "bend", "squat", "traffic", "high-five", "low-five"]

target_results = [find_entries(word, description_column) for word in target_words]

print("\n Searching for forbidden words:")
fbd_indices = []
for word in forbidden_words:
    fbd_indices.extend(find_entries(word, description_column)[2])
fbd_indices = list(set(fbd_indices))
print(f"Found {len(fbd_indices)} unique forbidden entries.")

print("\n Filtering forbidden words:")
indices_all = []
for i, result in enumerate(target_results):
    indices = result[2]
    filtered_indices = [index for index in indices if index not in fbd_indices]
    filtered_entries = description_column[filtered_indices]
    filtered_unique_entries = list(set(filtered_entries))
    print(f"Found {len(filtered_indices)} entries for '{target_words[i]}' after filtering, {len(filtered_unique_entries)} unique entries.")

    motion_ids = motion_id_column[filtered_indices]
    indices_all.extend(filtered_indices)
    
    if target_words[i] in ["walk", "dance", "basketball", "punch"]:
        save_yaml(motion_ids, filtered_entries.tolist(), target_words[i])

indices_all_unique = list(set(indices_all))
motion_ids_all_unique = motion_id_column[indices_all_unique]
save_yaml(motion_ids_all_unique, description_column[indices_all_unique].tolist(), "all_no_run_jump")