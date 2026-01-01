import os
import time
import requests
import asyncio
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from volcenginesdkarkruntime import Ark
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types # For creating message Content/Parts

APP_NAME = "weather_tutorial_app"
USER_ID  = "user_001"
SESS_ID  = "session_001" 

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

async def call_agent_async(query: str, runner, user_id, session_id):
  """Sends a query to the agent and prints the final response."""
  print(f"\n>>> User Query: {query}")

  # Prepare the user's message in ADK format
  content = types.Content(role='user', parts=[types.Part(text=query)])

  final_response_text = "Agent did not produce a final response." # Default

  # Key Concept: run_async executes the agent logic and yields Events.
  # We iterate through events to find the final answer.
  start = time.time()
  async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):

      # Key Concept: is_final_response() marks the concluding message for the turn.
      if event.is_final_response():
          if event.content and event.content.parts:
             # Assuming text response in the first part
             final_response_text = event.content.parts[0].text
          elif event.actions and event.actions.escalate: # Handle potential errors/escalations
             final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
          # Add more checks here if needed (e.g., specific error codes)
          break # Stop processing events once the final response is found

  print(f"Task for {query} took {time.time() - start:.2f}s")
  print(f"<<< Agent Response: {final_response_text}")

async def init_session(app_name:str,user_id:str,session_id:str):
    await session_service.create_session(
        app_name = APP_NAME,
        user_id = USER_ID, 
        session_id = SESS_ID 
    )
    print(f"Session created: App='{app_name}', User='{user_id}', Session='{session_id}'")

"""
# uncommt the followong code to run parallel run_conversation version 
# run the queries in parallel seems not reduce the total time - maybe something limited by
# the model service or session lock mechanism
async def run_conversation():
    # Schedule all three tasks at once
    tasks = [
        call_agent_async("What is the weather like in London?", runner, USER_ID, SESS_ID),
        call_agent_async("How about San Jose?", runner, USER_ID, SESS_ID),
        call_agent_async("Tell me the weather in New York", runner, USER_ID, SESS_ID)
    ]
    
    # Wait for all of them to finish
    await asyncio.gather(*tasks)
    
    print("All conversations completed concurrently.")
"""

async def run_conversation():
    await call_agent_async("What is the weather like in London?",
                                       runner=runner,
                                       user_id=USER_ID,
                                       session_id=SESS_ID)
    await call_agent_async("How about San Jose?",
                                       runner=runner,
                                       user_id=USER_ID,
                                       session_id=SESS_ID) # Expecting the tool's error message
    await call_agent_async("Tell me the weather in New York",
                                       runner=runner,
                                       user_id=USER_ID,
                                       session_id=SESS_ID)

async def main():

    global session_service, runner

    AGENT_MODEL = LiteLlm(
        model="volcengine/doubao-seed-1-6-251015",  # Use your specific endpoint/model ID
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=os.getenv('ARK_API_KEY')
    )

    weather_agent = LlmAgent(
        name = "weather_agent_v1",
        model = AGENT_MODEL,
        description = "Provides weather information for specific cities.",
        instruction = "You are a helpful weather assistant. "
                      "When the user asks for the weather in a specific city, "
                      "use the 'get_weather' tool to find the information. "
                      "If the tool returns an error, inform the user politely. "
                      "If the tool is successful, present the weather report clearly.",
        tools=[get_weather], # Pass the function directly
    )
    print(f"Agent '{weather_agent.name}' created using model '{AGENT_MODEL}'.")

    session_service = InMemorySessionService()
 
    # Create the specific session where the conversation will happen
    await init_session(APP_NAME, USER_ID, SESS_ID)
    print(f"Session created.")

    # Key Concept: Runner orchestrates the agent execution loop.
    runner = Runner(
        agent=weather_agent, # The agent we want to run
        app_name=APP_NAME,   # Associates runs with our app
        session_service=session_service # Uses our session manager
    )
    print(f"Runner created for agent '{runner.agent.name}'.")

    try:
        await run_conversation()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
