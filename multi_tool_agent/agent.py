import datetime
import os
from zoneinfo import ZoneInfo
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from volcenginesdkarkruntime import Ark

from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder

def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city for which to retrieve the weather report.

    Returns:
        dict: status and result or error msg.
    """
    if city.lower() == "new york":
        return {
            "status": "success",
            "report": (
                "The weather in New York is sunny with a temperature of 25 degrees"
                " Celsius (77 degrees Fahrenheit)."
            ),
        }
    else:
        return {
            "status": "error",
            "error_message": f"Weather information for '{city}' is not available.",
        }

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
