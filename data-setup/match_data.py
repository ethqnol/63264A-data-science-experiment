from types import CodeType
from robotevents_cheeser import generate_header
from enum import Enum
import requests 
import json
import os
import pandas as pd

with open('./data/data.json', 'r') as file:
    data = json.load(file)["data"]
    
print(len(data))

with open('./data/team.json', 'r') as file:
    team_data = json.load(file)
    
team_stats = pd.read_csv('./data/team_data.csv')

OUT_DIR = './data/match_info'
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
    

class ReturnCode(Enum):
   success = 0
   failure = 1
   invalid_data = 2




class RobotEvents:
    def __init__(self) -> None:
        self.api_keys = [
            "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiIzIiwianRpIjoiMzRjMTNlZGQyMGZhZjNmYmJkM2JkYmZlYWI3MWJiYmYwYTJjZjIwOTgxOGNmZThhMzAzMDY3MDI3ZDhmODUxMDVmMmY0M2Y5NjNmOWI1ZmMiLCJpYXQiOjE3NDQ0MDEzNzUuMjM5MDkyMSwibmJmIjoxNzQ0NDAxMzc1LjIzOTA5MzEsImV4cCI6MjY5MTA4NjE3NS4yMzM2ODEyLCJzdWIiOiIxMjE2MzciLCJzY29wZXMiOltdfQ.jLQ6h1yEhaZYnpSEWxDSDm25vKXMNgkdInXXGbb1EXf45fDAaOUWkp2ckx8nTjZR6qretav9xT6otkmv91HGpoMl5aGUyVlCq00kts0Y_utl8yFKkiVj0DRDFXWz6_g2atX5ZdDfs7GV4QTyFfuWkP36RzSNZxhLkqMT40m2dJ4q0uxmO5TBwAepa4LA1C0lxxiknxUYB0ayFAKapi9khRP6W6f_jBH4jpvkaUbrSj4ofNh0xYXOwNb9klaCdiZh_7Dl809IUbVak-VbGOUEaGI5IRZiNe2UIyekTGJMOrVINFTq-0HCmk7Exa0gV0nM8mC3tiUjyVf89kpxNXbbyDw0eoeq2ApbWktflOpJZ8dR25zmluaztEwpJ3skikUjGpmH_iJGr8V8w45psL3h0NOF07te8-5NSxlAtvD5Aiulrii24UUnszP8x7xKwzAUhqOcQwvJSgiTUsH5rCWR2f67OhtxWl59B_EEmyMfo7E399sMUBM5B8bIiYpHpp4GElQ-YJbC79j4DjwMIu-TXGo10xPw3ic88enFmkGdGBwMO7qiVFnnPhWuO_flqEKRgeoh3gw7fy65kmsR4cHoukJj2ps-r8yzcki2M8qdPC_zmKTCDRrrAGDNKeC7aqoZ3mWE1XDKYQgEt97s3URk66r2FQ1tpr9tnT1P6aWSRQQ",
            
            "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiIzIiwianRpIjoiZDAzMDkwNjg5MzcwNjNkY2UxYmY0ZGIxNGY5ZTRkZGQ1NTY1ODRiZDM2ZDhiN2JmNWYyNmE0NWUxZTlmYjRlNjEzZjExZDZkNTE4NjUyODYiLCJpYXQiOjE3NDkwNTM1ODcuNDczMTE0LCJuYmYiOjE3NDkwNTM1ODcuNDczMTE1LCJleHAiOjI2OTU3MzgzODcuNDY3NDM2OCwic3ViIjoiMTIwOTMzIiwic2NvcGVzIjpbXX0.StStXwbfo_e3VJE1G2AgNATtcXm2362DNyXfTBdaO_OBDl5C1OWc0z47c6pm5FaLNmqDbKTp8rKia5w7qrfdraiJiSQREeotzVdQovCXJH6IQBM3HHt84F4FuW38JV7T0wECv2ZRkuOgZO96SX2en-EMZeCyOoupco_J4ems7qLHBnJdBRRhv5h69yTFlxVzv1s99c7NW1dDoiVq3o9V6hz-JDe8_14X8aP5uuLWVWYLzs9B93ttzzvsRz9cieoSYEzecs7Y4kJmF5ZqI13JGn6fzFzlvujqJ7d5LCfEdM-XUlZsLJPhVI32pOMCmlhOrykz3pFRdHSi5EFJQNorfGGN1k43YL2ARWuxJQV4sZoeAmrzvojbkfpNgeKP94NUVHdIbWypIfl9KmPwgncbrCRBUnj0sxPUAcz2U1GWvFaiEsMALmVGLEOo1piGXbQtp5W4a4lAMPiym7lsANuMAuCr3pRd2MLeZVLnSWouUBs4PHkemklXAjPfhsRpVkaA3zYv5QjsE11wZjISTISvZgF1D8_UpeLkvCOcG9rVdqEyrmofnoiZUih_PoD2ea-7Xp4IscN6xh976Dv-D1eri5Bu7v5VO7ufGW5nmhQ4i9Mi1a-zj-ZyVL8d8t6pwS4hMyxv7Nc2rq2w3St1rZhpPoHsmLDgE1kZjERMi1uLWug"
        ]
        self.base_url = "https://vrc-data-analysis.com/v1"
        self.robot_events_url = "https://www.robotevents.com/api/v2"
        self.api_key_num = 0
        self.num_of_keys = len(self.api_keys)
        self.per_page = 250
    
    def select_key(self):
        self.api_key_num = (self.api_key_num + 1) % self.num_of_keys
        return self.api_keys[self.api_key_num]
    
    def generate_header(self):
        return generate_header(self.select_key())
        
    def request(self, url, endpoint, params=None):
        url = f"{url}/{endpoint}"
        headers = self.generate_header()
        response = requests.get(url, headers=headers, params=params)
        try:
            response.json()
        except requests.exceptions.HTTPError as error:
            print(error)
        except requests.exceptions.ReadTimeout as error:
            print(error)
        except requests.exceptions.ConnectionError as error:
            print(error)
        except requests.exceptions.RequestException as error:
            print(error)
            
        if response.ok:
            data = response.json()
            if data["data"]:
                return (ReturnCode.success, data)
            else: 
                return (ReturnCode.invalid_data, None)
        else: 
            return (ReturnCode.failure, None)
        
    def request_all_pages(self, url, endpoint, params):
        all_data = []
        page = 1
        total_pages = 1
        params["per_page"] = self.per_page
        while page <= total_pages:
            params["page"] = page
            req = self.request(url, endpoint, params)
            page += 1
            (code, data) = req
            if code == ReturnCode.success and data != None:
                all_data.extend(data["data"])
                total_pages = data["meta"]["last_page"]
            else:
                return (ReturnCode.failure, all_data)
                
        return (ReturnCode.success, all_data)


class CompInfo:
    def __init__(self, id: int, divisions: dict):
        self.div_ids = []
        self.id = id
        
        for div in divisions:
            self.div_ids.append(div["id"])
            
    def __str__(self) -> str:
        return f"[{self.id} - {self.div_ids}]"

class MatchInfo:
    def __init__(self, red_1_id, red_2_id, blue_1_id, blue_2_id, red_score, blue_score):
        self.red_1_id = red_1_id
        self.red_2_id = red_2_id
        self.blue_1_id = blue_1_id
        self.blue_2_id = blue_2_id
        self.blue_score = blue_score
        self.red_score = red_score
    
        self.team_lookup = team_data
        self.team_stats_df = team_stats
        
    def __str__(self):
        red_1 = self.team_lookup.get(self.red_1_id, self.red_1_id)
        red_2 = self.team_lookup.get(self.red_2_id, self.red_2_id)
        blue_1 = self.team_lookup.get(self.blue_1_id, self.blue_1_id)
        blue_2 = self.team_lookup.get(self.blue_2_id, self.blue_2_id)
    
        return (
            f"Red Alliance: {red_1} & {red_2} (Score: {self.red_score})\n"
            f"Blue Alliance: {blue_1} & {blue_2} (Score: {self.blue_score})"
        )
    
    
    def _get_team_stats(self, team_id: str):
        team_number = str(team_id)
        try:
            stats = self.team_stats_df.loc[self.team_stats_df["number"] == team_number]
            return {
                "comps_attended": stats.get("comps_attended", None),
                "trueskill": stats.get("trueskill", None),
                "opr": stats.get("opr", None),
                "dpr": stats.get("dpr", None),
                "wp_per_match": stats.get("wp_per_match", None)
            }
        except KeyError:
            return None

    
    def generate_full(self):
        red_1_stats = self._get_team_stats(self.red_1_id)
        red_2_stats = self._get_team_stats(self.red_2_id)
        blue_1_stats = self._get_team_stats(self.blue_1_id)
        blue_2_stats = self._get_team_stats(self.blue_2_id)
        

        if red_1_stats is None or red_2_stats is None or blue_1_stats is None or blue_2_stats is None:
            return None

        final_dict = {}
        for key, value in red_1_stats.items():
            final_dict[f"red_1_{key}"] = value
        for key, value in red_2_stats.items():
            final_dict[f"red_2_{key}"] = value
        for key, value in blue_1_stats.items():
            final_dict[f"blue_1_{key}"] = value
        for key, value in blue_2_stats.items():
            final_dict[f"blue_2_{key}"] = value
        
        final_dict["red_score"] = self.red_score
        final_dict["blue_score"] = self.blue_score
        
        return final_dict

    


robot_events = RobotEvents()

all_events: list[CompInfo] = []

# Retrieve IDs & divisions for every state level event
for item in data:
    all_events.append(CompInfo(item["id"], item["divisions"]))
    
    
match_info = []  
for event in all_events:
    for div in event.div_ids:
        (code, match_data) = robot_events.request_all_pages(robot_events.robot_events_url, f"events/{event.id}/divisions/{div}/matches", params = {"round[]": [2, 3, 6]})
        if code != ReturnCode.success:
            print("INVALID DATA; BREAKING EARLY")
            break;
        else:
            for data in match_data:
                alliances = data["alliances"]
                red_alliance = next((a for a in alliances if a["color"] == "red"), None)
                blue_alliance = next((a for a in alliances if a["color"] == "blue"), None)
                
                if red_alliance and blue_alliance:
                    info = MatchInfo(
                        red_1_id=red_alliance["teams"][0]["team"]["name"],
                        red_2_id=red_alliance["teams"][1]["team"]["name"],
                        blue_1_id=blue_alliance["teams"][0]["team"]["name"],
                        blue_2_id=blue_alliance["teams"][1]["team"]["name"],
                        red_score=red_alliance["score"],
                        blue_score=blue_alliance["score"]
                    )
                    final_info = info.generate_full()
                    print(final_info)
                    print("============================")
                    if info != None:
                        match_info.append(info)
                
                
