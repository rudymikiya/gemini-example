import os

import google.generativeai as genai

# Set proxy if it is needed...........
proxy = 'http://localhost:7897'
os.environ['http_proxy'] = proxy
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy

if __name__ == "__main__":
    # read the api key from MyApiKey.txt
    with open('MyApiKey.txt', 'r') as file:
        api_key = file.read().strip()
        print(f"API key: {api_key}")
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
    response = model.generate_content("Hello, this is my first testing project to use gemini. Do you have any suggestion for it?")
    print(response.text)