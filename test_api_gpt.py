from openai import OpenAI
import os 
client = OpenAI(
    api_key=""
)

try:
    r = client.responses.create(
        model="gpt-4.1-mini",
        input="Say only: API is working"
    )

    print(r.output_text)

except Exception as e:
    print(e)