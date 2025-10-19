from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os

# Khởi tạo client
client = OpenAI(
    base_url=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
    api_key=os.environ["OPENAI_API_KEY"]
)

# Gửi request chat completion
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ],
    timeout=120
)

print(response.choices[0].message.content)
