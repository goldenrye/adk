import datetime
import os
import requests

from zoneinfo import ZoneInfo
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from volcenginesdkarkruntime import Ark

from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder

def get_weather(city: str) -> dict:
    """Retrieves current weather for any city using Open-Meteo's free APIs."""
    
    # 1. Geocoding: Convert city name to coordinates
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    try:
        # requests.get will combine the URL and params correctly: 
        # geocoding-api.open-meteo.com&...
        geo_response = requests.get(geo_url, params=geo_params)
        geo_data = geo_response.json()

        if not geo_data.get("results"):
            return {"status": "error", "error_message": f"City '{city}' not found."}

        # Extract coordinates and formal name
        location = geo_data["results"][0]
        lat, lon = location["latitude"], location["longitude"]
        full_name = f"{location['name']}, {location.get('country', '')}"

        # 2. Weather Fetch: Use coordinates to get current weather
        weather_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,weather_code",
            "timezone": "auto"
        }
        
        weather_response = requests.get(weather_url, params=params)
        weather_data = weather_response.json()
        
        current = weather_data.get("current")
        if not current:
            return {"status": "error", "error_message": "Could not retrieve weather data."}

        temp = current["temperature_2m"]
        return {
            "status": "success",
            "report": f"The weather in {full_name} is currently {temp}°C."
        }

    except requests.exceptions.RequestException as e:
        return {"status": "error", "error_message": f"Network error: {str(e)}"}


def get_current_time(city: str) -> dict:
    """
    Returns the current time for any valid city globally using dynamic lookups.
    """
    try:
        # 1. Geocode the city name to get Lat/Long
        geolocator = Nominatim(user_agent="time_explorer")
        location = geolocator.geocode(city, language='en')
        
        if not location:
            return {"status": "error", "message": f"Location '{city}' not found."}

        # 2. Find the timezone identifier (e.g., 'America/New_York')
        tf = TimezoneFinder()
        tz_identifier = tf.timezone_at(lng=location.longitude, lat=location.latitude)

        if not tz_identifier:
            return {"status": "error", "message": "Could not determine timezone."}

        # 3. Get the current time in that timezone
        tz = ZoneInfo(tz_identifier)
        now = datetime.datetime.now(tz)
        
        return {
            "status": "success",
            "city": location.address,
            "timezone": tz_identifier,
            "time": now.strftime("%Y-%m-%d %H:%M:%S %Z%z"),
            "report": f"The current time in {city} is {now.strftime('%I:%M %p')}"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# 1. Get model's API key 
api_key = os.getenv('ARK_API_KEY')

# 2. Define the Doubao Model using the LiteLlm wrapper.
# We use the 'openai/' prefix to tell LiteLLM to use the OpenAI-compatible logic,
# then we point it to the Volcano Engine Ark base URL.
doubao_model = LiteLlm(
    model="volcengine/doubao-seed-1-6-251015",  # Use your specific endpoint/model ID
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=api_key
)

# 3. Create the Agent.
# The framework specifically looks for 'root_agent' as the entry point.
root_agent = LlmAgent(
    model=doubao_model,
    name="doubao_agent",
    description=(
        "Agent to answer questions about the time and weather in a city."
    ),
    instruction=(
        "You are a helpful agent who can answer user questions about the time and weather in a city."
    ),
    tools=[get_weather, get_current_time]
)
