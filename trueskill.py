import requests
import pandas as pd
import math
import time
API_BASE_TEAM_STATS = "https://vrc-data-analysis.com/v1"
API_KEY = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiIzIiwianRpIjoiMzRjMTNlZGQyMGZhZjNmYmJkM2JkYmZlYWI3MWJiYmYwYTJjZjIwOTgxOGNmZThhMzAzMDY3MDI3ZDhmODUxMDVmMmY0M2Y5NjNmOWI1ZmMiLCJpYXQiOjE3NDQ0MDEzNzUuMjM5MDkyMSwibmJmIjoxNzQ0NDAxMzc1LjIzOTA5MzEsImV4cCI6MjY5MTA4NjE3NS4yMzM2ODEyLCJzdWIiOiIxMjE2MzciLCJzY29wZXMiOltdfQ.jLQ6h1yEhaZYnpSEWxDSDm25vKXMNgkdInXXGbb1EXf45fDAaOUWkp2ckx8nTjZR6qretav9xT6otkmv91HGpoMl5aGUyVlCq00kts0Y_utl8yFKkiVj0DRDFXWz6_g2atX5ZdDfs7GV4QTyFfuWkP36RzSNZxhLkqMT40m2dJ4q0uxmO5TBwAepa4LA1C0lxxiknxUYB0ayFAKapi9khRP6W6f_jBH4jpvkaUbrSj4ofNh0xYXOwNb9klaCdiZh_7Dl809IUbVak-VbGOUEaGI5IRZiNe2UIyekTGJMOrVINFTq-0HCmk7Exa0gV0nM8mC3tiUjyVf89kpxNXbbyDw0eoeq2ApbWktflOpJZ8dR25zmluaztEwpJ3skikUjGpmH_iJGr8V8w45psL3h0NOF07te8-5NSxlAtvD5Aiulrii24UUnszP8x7xKwzAUhqOcQwvJSgiTUsH5rCWR2f67OhtxWl59B_EEmyMfo7E399sMUBM5B8bIiYpHpp4GElQ-YJbC79j4DjwMIu-TXGo10xPw3ic88enFmkGdGBwMO7qiVFnnPhWuO_flqEKRgeoh3gw7fy65kmsR4cHoukJj2ps-r8yzcki2M8qdPC_zmKTCDRrrAGDNKeC7aqoZ3mWE1XDKYQgEt97s3URk66r2FQ1tpr9tnT1P6aWSRQQ" 
API_BASE_ROBOTEVENTS = "https://www.robotevents.com/api/v2"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
    "accept-language": "en",
    "user-agent": "PythonVexDataValidationBotPython/2.0 (Python Script; fuckyourobotevents@robotevents.com)"
}
PER_PAGE = 250 
EVENT_TYPES = ["tournament"]

STATS_HEADERS = {
    "accept-language": "en",
    "user-agent": "Robotevents Fr Fr I am Grant Cox no cap"
}

PARAMS = {
    "season": [190]
}

team_info_df = pd.read_csv('data/state_level_teams_simple.csv')
print(team_info_df.columns.tolist())
final_df = pd.DataFrame()

for i, row in team_info_df.iterrows():
    id = row["id"]
    team_num = row["number"]
    
    data_analysis_response = requests.get(f"{API_BASE_TEAM_STATS}/team/{team_num}", headers=STATS_HEADERS,timeout=20) 
    robot_events_response = requests.get(f"{API_BASE_ROBOTEVENTS}/teams/{id}/events", headers=HEADERS, params=PARAMS, timeout=15)
    data = data_analysis_response.json()
    events_data = robot_events_response.json()
    
    num_sigs = 0
    world_qual = "False"
    for event in events_data["data"]:
        if event["level"] == "Signature":
            num_sigs += 1
        elif event["level"] == "World":
            world_qual = "True" 
    
    team_stats = {
        "number": team_num,
        "region": row["region"] if type(row["region"]) == type("string") else row["country"],
        "comps_attended": len(events_data["data"]),
        "sigs_attended": num_sigs,
        "worlds_qual": world_qual,
        "trueskill": data["trueskill"] if "trueskill" in data else float("nan"),
        "trueskill_ranking": data["trueskill_ranking"] if "trueskill_ranking" in data else float("nan"),
        "opr": data["opr"] if "opr" in data else float("nan"),
        "dpr": data["dpr"] if "dpr" in data else float("nan"),
        "ccwm": data["ccwm"] if "ccwm" in data else float("nan"),
        "ap_per_match": data["ap_per_match"] if "ap_per_match" in data else float("nan"),
        "awp_per_match": data["awp_per_match"] if "awp_per_match" in data else float("nan"),
        "wp_per_match": data["wp_per_match"] if "wp_per_match" in data else float("nan"),
        "drive_score": data["score_driver_max"] if "score_driver_max" in data else float("nan"),
        "auto_score": data["score_auto_max"] if "score_auto_max" in data else float("nan")
    }

    print(team_stats)
        
    import random
    time.sleep(random.randint(3, 6))
    final_df = pd.concat([final_df, pd.DataFrame([team_stats])], ignore_index=True)

final_df.to_csv("sq_team_info.csv",index = False)


