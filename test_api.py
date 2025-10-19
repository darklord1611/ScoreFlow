from openai import OpenAI

# Khởi tạo client
client = OpenAI(
    base_url="https://api.yescale.io/v1",  # KHÔNG có khoảng trắng ở đầu
    api_key="sk-1BRrBUMuTy3oIc74wSQ2Cw9Kv4zzeSylUFyemcGUejKxupwt"
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
