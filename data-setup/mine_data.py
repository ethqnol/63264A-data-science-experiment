import requests
import csv
import time
import os

API_KEY = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiIzIiwianRpIjoiMzRjMTNlZGQyMGZhZjNmYmJkM2JkYmZlYWI3MWJiYmYwYTJjZjIwOTgxOGNmZThhMzAzMDY3MDI3ZDhmODUxMDVmMmY0M2Y5NjNmOWI1ZmMiLCJpYXQiOjE3NDQ0MDEzNzUuMjM5MDkyMSwibmJmIjoxNzQ0NDAxMzc1LjIzOTA5MzEsImV4cCI6MjY5MTA4NjE3NS4yMzM2ODEyLCJzdWIiOiIxMjE2MzciLCJzY29wZXMiOltdfQ.jLQ6h1yEhaZYnpSEWxDSDm25vKXMNgkdInXXGbb1EXf45fDAaOUWkp2ckx8nTjZR6qretav9xT6otkmv91HGpoMl5aGUyVlCq00kts0Y_utl8yFKkiVj0DRDFXWz6_g2atX5ZdDfs7GV4QTyFfuWkP36RzSNZxhLkqMT40m2dJ4q0uxmO5TBwAepa4LA1C0lxxiknxUYB0ayFAKapi9khRP6W6f_jBH4jpvkaUbrSj4ofNh0xYXOwNb9klaCdiZh_7Dl809IUbVak-VbGOUEaGI5IRZiNe2UIyekTGJMOrVINFTq-0HCmk7Exa0gV0nM8mC3tiUjyVf89kpxNXbbyDw0eoeq2ApbWktflOpJZ8dR25zmluaztEwpJ3skikUjGpmH_iJGr8V8w45psL3h0NOF07te8-5NSxlAtvD5Aiulrii24UUnszP8x7xKwzAUhqOcQwvJSgiTUsH5rCWR2f67OhtxWl59B_EEmyMfo7E399sMUBM5B8bIiYpHpp4GElQ-YJbC79j4DjwMIu-TXGo10xPw3ic88enFmkGdGBwMO7qiVFnnPhWuO_flqEKRgeoh3gw7fy65kmsR4cHoukJj2ps-r8yzcki2M8qdPC_zmKTCDRrrAGDNKeC7aqoZ3mWE1XDKYQgEt97s3URk66r2FQ1tpr9tnT1P6aWSRQQ" 

SEASON_ID = 190
PROGRAM_IDS = [0] 

CSV_FILENAME = "state_level_teams_simple.csv"

# API Configuration
API_BASE_URL = "https://www.robotevents.com/api/v2"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
    "accept-language": "en",
    "user-agent": "PythonVexDataValidationBotPython/2.0 (Python Script; fuckyourobotevents@robotevents.com)"
}
PER_PAGE = 250 
EVENT_TYPES = ["tournament"]



all_state_events = []
unique_teams = {} # team_id : team_data 

print("\n---Fetching State-Level Events---")
page = 1
total_pages = 1 

while page <= total_pages:
    event_params = {
        "season": SEASON_ID,
        "level": "State", 
        "per_page": PER_PAGE,
        "page": page,
        "eventTypes": EVENT_TYPES
    }
    
    print(f"  Fetching events page {page}...")
    try:
        response = requests.get(f"{API_BASE_URL}/events", headers=HEADERS, params=event_params, timeout=30)
        response.raise_for_status() 

        data = response.json()
        
        if 'data' not in data or 'meta' not in data:
             print(f"  Warning: Unexpected response format on page {page}. Stopping event fetch.")
             print(f"  Response: {data}")
             break 

        page_data = data.get('data', [])
        meta = data.get('meta', {})

        if not page_data and page == 1:
            print("  No events found for this query.")
            break 

        all_state_events.extend(page_data)

        current_page = meta.get('current_page', page)
        last_page = meta.get('last_page', current_page) 
        total_pages = last_page 

        print(f"    Fetched page {current_page}/{last_page}. Found {len(page_data)} events. Total collected: {len(all_state_events)}")

        if current_page >= last_page:
            break 

        page += 1
        time.sleep(10) 

    except requests.exceptions.RequestException as e:
        print(f"Error fetching events page {page}: {e}")
        print(f"  Response Status Code: {response.status_code if 'response' in locals() else 'N/A'}")
        print(f"  Response Text: {response.text if 'response' in locals() else 'N/A'}")
        break 
    except Exception as e:
        print(f"An unexpected error occurred processing events page {page}: {e}")
        break 

if not all_state_events:
    print("\nNo state-level events found matching the criteria. Exiting.")
    exit()

print(f"\nFound {len(all_state_events)} state-level events.")


import json

filename = "data.json"

with open(filename, 'w') as file:
    json.dump(data, file, indent=4)


print("\n--- Fetching Teams from each Event ---")
for event in all_state_events:
    event_id = event.get('id')
    event_name = event.get('name', 'Unknown Event Name')
    if not event_id:
        print("  Warning: Found an event with no ID. Skipping.")
        continue

    print(f"\nProcessing Event ID: {event_id} ({event_name})")
    
    team_page = 1
    team_total_pages = 1 
    teams_fetched_for_event = 0

    while team_page <= team_total_pages:
        team_params = {
            "per_page": PER_PAGE,
            "page": team_page,
            "grade": "High School"
        }
        
        print(f"  Fetching teams page {team_page} for event {event_id}...")
        time.sleep(5)
        try:
            team_response = requests.get(f"{API_BASE_URL}/events/{event_id}/teams", headers=HEADERS, params=team_params, timeout=30)
            team_response.raise_for_status()

            team_data = team_response.json()

            if 'data' not in team_data or 'meta' not in team_data:
                print(f"  Warning: Unexpected response format for teams page {team_page}, event {event_id}. Skipping rest of teams for this event.")
                print(f"  Response: {team_data}")
                break

            team_page_data = team_data.get('data', [])
            team_meta = team_data.get('meta', {})

            if not team_page_data and team_page == 1:
                 print(f"    No teams found for event {event_id}.")
                 break 
            teams_fetched_for_event += len(team_page_data)


            for team in team_page_data:
                team_id = team.get('id')
                if not team_id:
                    print("    Warning: Found a team record with no ID. Skipping.")
                    continue


                if team_id not in unique_teams:
                    team_info = {
                        'id': team.get('id'),
                        'number': team.get('number'),
                        'team_name': team.get('team_name'),
                        'robot_name': team.get('robot_name'),
                        'organization': team.get('organization'),
                        'city': team.get('location', {}).get('city'),
                        'region': team.get('location', {}).get('region'),
                        'country': team.get('location', {}).get('country'),
                        'program_id': team.get('program', {}).get('id'),
                        'program_name': team.get('program', {}).get('name'),
                        'grade': team.get('grade'),
                    }
                    unique_teams[team_id] = team_info
            
            team_current_page = team_meta.get('current_page', team_page)
            team_last_page = team_meta.get('last_page', team_current_page)
            team_total_pages = team_last_page # Update total pages for this specific event's teams

            print(f"    Fetched teams page {team_current_page}/{team_last_page}. Found {len(team_page_data)} teams. Total for event so far: {teams_fetched_for_event}")
            filename = "team.json"

            with open(filename, 'w') as file:
                    json.dump(unique_teams, file, indent=4)
            if team_current_page >= team_last_page:
                break 

            team_page += 1
            
            time.sleep(20) 


        except requests.exceptions.RequestException as e:
            status_code = str(team_response.status_code) if 'team_response' in locals() else 'N/A'
            print(f"Error fetching teams page {team_page} for event {event_id}: {e}")
            print(f"  Response Status Code: {team_response.status_code if 'team_response' in locals() else 'N/A'}")
            print(f"  Response Text: {team_response.text if 'team_response' in locals() else 'N/A'}")


            break # Stop fetching teams for this event on error
        except Exception as e:
            print(f"An unexpected error occurred processing teams page {team_page} for event {event_id}: {e}")
            break # Stop fetching teams for this event on error


print(f"Found {len(unique_teams)} unique teams across {len(all_state_events)} regional level events.")


print(f"\n--- Exporting Data to {CSV_FILENAME} ---")
if not unique_teams:
    print("No unique team data collected. CSV file will not be created.")
else:
    fieldnames = [
        'id', 'number', 'team_name', 'robot_name', 'organization', 
        'city', 'region', 'country', 'program_id', 'program_name', 'grade' 
    ]

    try:
        with open(CSV_FILENAME, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(unique_teams.values()) # Write the collected unique team data
        
        print(f"Successfully exported {len(unique_teams)} teams to {CSV_FILENAME}")

    except IOError as e:
        print(f"Error writing to CSV file {CSV_FILENAME}: {e}")
    except Exception as e:
         print(f"An unexpected error occurred during CSV export: {e}")

