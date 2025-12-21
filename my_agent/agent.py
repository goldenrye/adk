import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from volcenginesdkarkruntime import Ark

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
    instruction="You are a helpful assistant powered by the Doubao model.",
)
