from openai import OpenAI
from dotenv import load_dotenv
import os

def set_openai_api():
    load_dotenv(override=True)

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print( "No API key was found")
    else:
        print("API key found and looks good so far!")
        client = OpenAI()
        return client
    
client = set_openai_api()

def get_sentiment_label(text):
    
    """Passes a single instruction to the LLM and returns the strict category."""

    system_prompt = """ 
    You are an expert customer support sentiment classifier.
    Read the customer message and classify their emotional state into exactly ONE of these four categories:
    - Frustrated
    - Urgent
    - Polite
    - Neutral

    Output NOTHING but the exact category word. No punctuation, no explanation.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Customer message: '{text}'"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Error: {e}")
        return "Error"
    