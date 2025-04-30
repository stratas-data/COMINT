import torch
import pandas as pd
import whisper
from typing import Optional, List, Dict, Union
from pydantic import Field
from pydantic import BaseModel
import requests
import json
import geopy.distance
import getpass
import os
import openai

frequency = 155.88
mode = "P25"
SDRcounty = "Wise County"
SDRprovince = "Texas"
SDRcountry = "United States"
NA_counties = "C:/Users/antho/Downloads/v2_NA_counties_OSM.json"
whisper_model = whisper.load_model("large-v2", device="cuda")

def filter_frequencies_and_modes(frequency, mode):
    """
    This function filters rows from two CSV files where:
    - The given `frequency` appears in either 'Frequency Output' or 'Frequency Input'.
    - The given `mode` appears in the 'Mode' column.
    
    It extracts:
    1. A list of lists containing 'Agency/Category', 'Description', and 'Tag'.
    2. A list of counties formatted as 'County Name County'.
    
    Args:
    - frequency (float): The frequency to filter for.
    - mode (str): The mode to filter for.
    
    Returns:
    - context_list (list of lists): Contextual information for the agent.
    - counties_list (list): List of counties formatted as "County Name County".
    """
    # Define file paths
    texas_file = r"C:\Users\antho\Downloads\texas_df.csv"
    nationwide_file = r"C:\Users\antho\Downloads\nationwide\nationwide_df.csv"
    
    # Load CSV files
    df_texas = pd.read_csv(texas_file)
    df_nationwide = pd.read_csv(nationwide_file)
    
    # Combine both DataFrames
    freq_df = pd.concat([df_texas, df_nationwide], ignore_index=True)
    
    # Filter for matching frequency and mode
    filtered_df = freq_df[
        ((freq_df["Frequency Output"] == frequency) | (freq_df["Frequency Input"] == frequency)) & 
        (freq_df["Mode"] == mode)
    ]
    
    # Extract list of lists for context
    frequency_context = filtered_df[["Agency/Category", "Description", "Tag"]].values.tolist()

    # Extract counties and format them
    frequency_counties = [
    f"{county} County" for county in filtered_df["County"].dropna().unique() 
    if county != "Statewide"
    ]
    
    return filtered_df, frequency_context, frequency_counties

filtered_df, frequency_context, frequency_counties = filter_frequencies_and_modes(frequency, mode)

counties_to_load = set(frequency_counties)
counties_to_load.add(SDRcounty)
counties_to_load = list(counties_to_load)

def whisper_transcribe(audio_file):
    import whisper
    import torch

    print("Loading Whisper model...")
    model = whisper.load_model("large-v2", device="cuda")

    print("Loading and processing audio...")
    audio = whisper.load_audio(audio_file, sr=16000)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    mel = mel.to(torch.float16) if model.device.type == "cuda" else mel.to(torch.float32)

    options = whisper.DecodingOptions(
    fp16=True,
    patience=2.0
    )

    result = whisper.decode(model, mel, options)

    # Extract n-best hypotheses with their respective log probabilities
    n_best_hypotheses = []
    for text, logprobs in zip(result.texts, result.all_logprobs):
        n_best_hypotheses.append({"transcription": text.strip(), "logprob": logprobs})

    return n_best_hypotheses

transcriptions = [entry["transcription"] for entry in whisper_transcribe("C:/Users/antho/Downloads/wise-21001-1740486863.m4a")]

class OSMQuery(BaseModel):
    toponym: str
    SDRcounty: str
    SDRprovince: str
    SDRcountry: str

def verify_toponym_with_osm_tool(query: OSMQuery):
    """
    Verify toponyms using OpenStreetMap.
    The function first searches within the county (admin_level=6), 
    and if no results are found, expands the search to the state (admin_level=4).
    """
    print(f"🔍 Received OpenStreetMap query: {query}")  # Debugging Output

    try:
        toponym = query.toponym.strip()
        county = query.SDRcounty.strip()
        province = query.SDRprovince.strip()
        country = query.SDRcountry.strip()

        if not toponym:
            raise ValueError("Toponym is required for the query.")

        print(f"✅ Extracted -> Toponym: '{toponym}', County: '{county}', State: '{province}'")  # Debugging Output

        overpass_url = "https://overpass.kumi.systems/api/interpreter"  # Use a faster Overpass instance

        ### 🔹 Step 1: Try County-Level Search (admin_level=6)
        if county:
            area_query = f'area["name"="{county}"]["admin_level"="6"]->.searchArea;'
            area_ref = "area.searchArea"

            query_string = f"""
            [out:json][timeout:30];
            {area_query}
            (
              nwr["name"="{toponym}"]({area_ref});
            );
            out center;
            """

            print(f"🔍 OSM Query Sent (County-Level):\n{query_string}")  # Debugging Output
            response = requests.get(overpass_url, params={"data": query_string})

            if response.status_code == 200:
                data = response.json()
                if data.get("elements"):
                    print(f"✅ OSM Response (County-Level): {data}")
                    return {"exists": True, "details": data["elements"], "source": county}

        print("❌ No results at county level. Expanding search to state level...")

        ### 🔹 Step 2: Try State-Level Search (admin_level=4)
        if province:
            area_query = f'area["name"="{province}"]["admin_level"="4"]->.searchArea;'
            area_ref = "area.searchArea"

            query_string = f"""
            [out:json][timeout:30];
            {area_query}
            (
              nwr["name"="{toponym}"]({area_ref});
            );
            out center;
            """

            print(f"🔍 OSM Query Sent (Province-Level):\n{query_string}")  # Debugging Output
            response = requests.get(overpass_url, params={"data": query_string})

            if response.status_code == 200:
                data = response.json()
                if data.get("elements"):
                    print(f"✅ OSM Response (Province-Level): {data}")
                    return {"exists": True, "details": data["elements"], "source": province}
                
        print("❌ No results at provincial level. Expanding search to national level...")
                
        if country:
            area_query = f'area["name"="{country}"]["admin_level"="2"]->.searchArea;'
            area_ref = "area.searchArea"

            query_string = f"""
            [out:json][timeout:30];
            {area_query}
            (
              nwr["name"="{toponym}"]({area_ref});
            );
            out center;
            """

            print(f"🔍 OSM Query Sent (National-Level):\n{query_string}")  # Debugging Output
            response = requests.get(overpass_url, params={"data": query_string})

            if response.status_code == 200:
                data = response.json()
                if data.get("elements"):
                    print(f"✅ OSM Response (National-Level): {data}")
                    return {"exists": True, "details": data["elements"], "source": country}

        print("❌ No results found at any level.")
        return {"exists": False, "error": "No matching location found in county, province, or country."}

    except Exception as e:
        print(f"❌ ERROR in OpenStreetMap Tool Execution: {e}")
        return {"error": str(e)}
    
class TDQuery(BaseModel):
    toponym: str
    json_path: str
    SDRcounty: str
    SDRprovince: str
    counties_to_load: List[str]
    OSM_result: Dict[str, Union[str, dict, bool, list]]

def load_county_data(json_path: str, SDRprovince: str, counties_to_load: set) -> Dict[str, tuple]:
    """
    Loads county data for the SDR county and frequency counties within the given province.
    """
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    county_coordinates = {}

    # Find the matching province/state
    for state in data:
        if state["tags"].get("name") == SDRprovince:
            for county in state.get("counties", []):
                county_name = county["tags"].get("name")
                center = county.get("center")

                if county_name in counties_to_load and center:
                    county_coordinates[county_name] = (center["lat"], center["lon"])

    return county_coordinates

def find_nearest_toponym(
    counties_to_load: List[str], OSM_result: dict, NA_county_df: Dict[str, tuple], SDRcounty: str, toponym: str
) -> str:
    """
    Determines the most relevant geographic result and returns a formatted string:
    1. If multiple toponyms exist, returns the nearest instance to SDRcounty.
    2. If only one toponym exists, returns the nearest county to that toponym.
    """
    results = {}

    # Extract toponym locations from OSM result
    toponyms = {
        entry["id"]: (entry["center"]["lat"], entry["center"]["lon"])
        if "center" in entry else
        (entry["lat"], entry["lon"])
        if "lat" in entry and "lon" in entry else None
        for entry in OSM_result.get("details", [])
    }

    # Remove entries that have None (i.e., missing coordinates)
    toponyms = {k: v for k, v in toponyms.items() if v is not None}

    # If no toponyms found, return an error message
    if not toponyms:
        return "No valid toponyms found."

    # Case 1: Multiple toponyms exist → Find the nearest one to SDRcounty
    if len(toponyms) > 1:
        sdr_coords = NA_county_df.get(SDRcounty)

        if not sdr_coords:
            return f"Error: SDRcounty {SDRcounty} not found in county dataset."

        nearest_toponym = None
        nearest_coords = None
        min_distance = float("inf")

        for toponym_id, toponym_coords in toponyms.items():
            distance = geopy.distance.geodesic(sdr_coords, toponym_coords).km
            if distance < min_distance:
                min_distance = distance
                nearest_toponym = toponym_id
                nearest_coords = toponym_coords

        return f"{nearest_coords} is the nearest instance of '{toponym}' to SDR's county ({SDRcounty}), distance: {min_distance:.2f} km."

    # Case 2: Only one toponym exists → Find the nearest county to that toponym
    else:
        single_toponym_id, single_toponym_coords = list(toponyms.items())[0]

        nearest_county = None
        min_distance = float("inf")

        for county, county_coords in NA_county_df.items():
            distance = geopy.distance.geodesic(county_coords, single_toponym_coords).km
            if distance < min_distance:
                min_distance = distance
                nearest_county = county

        return f"{nearest_county} is the nearest relevant county to '{toponym}' {single_toponym_coords}, distance: {min_distance:.2f} km."

# ✅ Modify `topo_disambiguation` to accept a single input object
def topo_disambiguation(query: TDQuery) -> str:
    """
    Main function for toponym disambiguation.
    Returns a formatted string with the most relevant geographic information.
    """
    county_data = load_county_data(query.json_path, query.SDRprovince, set(query.counties_to_load))

    if query.OSM_result is None:
        query.OSM_result = {}

    return find_nearest_toponym(query.counties_to_load, query.OSM_result, county_data, query.SDRcounty, query.toponym)

class CombinedQuery(BaseModel):
    toponym: str
    SDRcounty: str
    SDRprovince: str
    SDRcountry: str
    json_path: str
    counties_to_load: List[str]

def verify_and_disambiguate(**kwargs) -> Dict[str, dict]:
    """
    First verifies the toponym using OpenStreetMap. If no match is found, stops execution.
    If a match is found, proceeds to disambiguate the toponym using county data.
    """
    print('TOOL CALLED')
    query = CombinedQuery(**kwargs)

    ### 🔹 Step 1: Verify Toponym Using OSM
    osm_query = {
        "toponym": query.toponym,
        "SDRcounty": query.SDRcounty,
        "SDRprovince": query.SDRprovince,
        "SDRcountry": query.SDRcountry
    }

    print(f"🔍 Verifying toponym: {osm_query}")  # Debugging Output
    osm_result = verify_toponym_with_osm_tool(OSMQuery(**osm_query))

    # If no match found, STOP execution
    if not osm_result.get("exists", False):
        print("❌ No verified toponym found. Stopping execution.")
        return {"error": "No matching location found in county, province, or country."}

    ### 🔹 Step 2: Disambiguate Toponym (Only if OSM Query Succeeds)
    disambiguation_query = {
        "toponym": query.toponym,
        "json_path": query.json_path,
        "SDRcounty": query.SDRcounty,
        "SDRprovince": query.SDRprovince,
        "counties_to_load": query.counties_to_load,
        "OSM_result": {
        "exists": str(osm_result.get("exists", False)),
        "details": osm_result.get("details", {})
        }
    }

    print(f"🔍 Disambiguating toponym using: {disambiguation_query}")  # Debugging Output
    disambiguation_result = topo_disambiguation(TDQuery(**disambiguation_query))

    return disambiguation_result

schema = {
    "name": "4W",
    "type": "object",
    "properties": {
        "who": {"type": "string", "description": "Identifies the responding unit, call signs, or individuals mentioned."},
        "what": {"type": "string", "description": "Describes the nature of the emergency, event, or action taken."},
        "when": {"type": "string", "description": "Captures any time reference if mentioned."},
        "where": {
            "type": "array",
            "description": "List of detected toponyms (geographical locations).",
            "items": {"type": "string"}
        }
    },
    "required": ["who", "what", "when", "where"],
    "additionalProperties": False
}

messages = [
    {"role": "system", "content": "You are an intelligence analyst who skeptically extracts information from transcriptions of radio transmissions. You specializes in extracting 'Who?,' 'What?,' 'When?,' and 'Where?' (4W) information from radio transmissions."},
    {"role": "user", "content": f"""
    You will process the following transcription hypotheses in light of the transcription's context, and return a four-line 4W summary:

    Transcription Hypotheses:
    ```json
    {transcriptions}
    ```

    Transcription Context: {frequency_context}

    Instructions:
    1. Extract 'Who?': Identify the responding unit, call signs, or individuals mentioned.
    2. Extract 'What?': Determine the nature of the emergency, event, or action taken.
    3. Extract 'When?': Capture any time reference if mentioned.
    4. Extract 'Where?': List each toponym you detect.
    5. If no information is gleaned for one of the four categories, default to 'nil'.
    """}
]

client = openai.OpenAI()

# Step 1: Call OpenAI API (LLM extracts 4W)
response = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=messages,
      response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "4W",
            "strict": True,
            "schema": schema
        }
    }
  )

  # Extract the structured response
structured_response = response.choices[0].message.content

try:
    llm_output = json.loads(structured_response)
except json.JSONDecodeError:
    print("❌ Error: OpenAI response is not valid JSON")
    llm_output = {"who": "nil", "what": "nil", "when": "nil", "where": []}

if llm_output["where"] != "nil":
    verified_where = []  # List to store processed toponyms

    for toponym in llm_output["where"]:
        print(f"🔍 Verifying toponym: {toponym}...")

        function_result = verify_and_disambiguate(
            toponym=toponym,
            SDRcounty=SDRcounty,
            SDRprovince=SDRprovince,
            SDRcountry=SDRcountry,
            json_path=NA_counties,
            counties_to_load=counties_to_load
        )

        # ✅ Process function result
        if function_result.get("error"):  
            verified_where.append(f"Unverified: {toponym}")  # Mark as unverified
            
        else:
            verified_where.append(function_result)

    # ✅ Replace 'where' with the verified list
    llm_output["where"] = verified_where

# ✅ Step 3: Final Output
print("🔹 Final 4W Output:", json.dumps(llm_output, indent=4))