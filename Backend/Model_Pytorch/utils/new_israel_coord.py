import json
import requests

data_station_path =  "C:\\Users\\dorsh\\Documents\\GitHub\\WeatherNet\\data code files"
base_url = "https://ims.gov.il/"


def openJsonFile(file_path):
    with open(file_path) as file:
        stations = json.load(file)
    return stations

def getStationsCoordinatesFromIMS(stations):
      # Iterate through the JSON data
    for station_id, station_details in stations.items():
        if "station_page" in station_details:
            # Construct the full URL
            page = station_details["station_page"]
            full_url = base_url + page.replace("node", "station_data")
            try:
                # Fetch the content from the URL
                response = requests.get(full_url)
                response.raise_for_status()  # Check for HTTP request errors
                
                            # Ensure the content is JSON
                if "application/json" in response.headers.get("Content-Type", ""):
                    data = response.json()  # Parse the JSON response
                else:
                    # Fallback: Attempt to manually load JSON from text
                    data = json.loads(response.text)
                data = response.json()  # Parse the JSON response
                
                # Validate if "title" matches "name"
                api_title = data.get("data", {}).get("title", "")
                json_name = station_details.get("name", "")
                
                if api_title == json_name:
                    print(f"Match for Station ID {station_id}: Yes")
                    
                    # Get coordinates and append to the JSON
                    coordinates = data.get("data", {}).get("coordinates_in_a_new_israe", {})
                    if coordinates:
                        station_details["coordinates_in_a_new_israe"] = coordinates
                else:
                    print(f"Match for Station ID {station_id}: No")
            
            except requests.exceptions.RequestException as e:
                print(f"Error fetching URL for Station ID {station_id}: {e}")
            except json.JSONDecodeError:
                print(f"Invalid JSON response for Station ID {station_id}")
    return stations

def saveStationJson(stations, saveFolder = data_station_path):
    # Save the updated JSON data back to the file
    with open(f"{data_station_path}\\stations_details_updated.json", "w") as file:
        json.dump(stations, file, indent=4)

    print("Updated JSON saved to stations_details_updated.json")

if __name__ == "__main__":
    # Load the JSON file
    stations = openJsonFile(f"{data_station_path}\\stations_details.json")
    updatedStations = getStationsCoordinatesFromIMS(stations)
    
    saveStationJson(stations)


