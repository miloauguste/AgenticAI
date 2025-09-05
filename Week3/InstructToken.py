import openai
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="C:/Users/milo.MILOJR-LENOVA/projects/llm_engineering/.env")

openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
openai.api_key = "your_openai_api_key"



response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",  # or "gpt-4"
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a poem about sunrise."}
    ]
)

print(response['choices'][0]['message']['content'])
