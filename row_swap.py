import pandas as pd

data = pd.read_csv("test.csv")
og_data = data.copy()
keys = ["comps_attended",
        "trueskill",
        "opr",
        "dpr",
        "awp_per_match",
        "ap_per_match",
        "wp_per_match",
    ]
swap_keys = ["red_1", "red_2", "blue_1", "blue_2"]


def swap_keys():
    data[f"red_score"], data[f"blue_score"] = data[f"blue_score"].copy(), data[f"red_score"].copy()
    for key in keys: 
        data[f"{key}_red_1"], data[f"{key}_blue_1"] = data[f"{key}_blue_1"].copy(), data[f"{key}_red_1"].copy()
        data[f"{key}_red_2"], data[f"{key}_blue_2"] = data[f"{key}_blue_2"].copy(), data[f"{key}_red_2"].copy()

swap_keys()

final = pd.concat([og_data, data], axis=0)
final.to_csv("swapped.csv", index=False)