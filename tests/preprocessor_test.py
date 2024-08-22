import os

from src.email_preprocessor import EmailPreprocessor

# Set proxy if it is needed...........
proxy = 'http://localhost:7897'
os.environ['http_proxy'] = proxy
os.environ['https_proxy'] = proxy

# read the api key from MyApiKey.txt
with open('../MyApiKey.txt', 'r') as file:
    api_key = file.read().strip()
    print(f"API key: {api_key}")
os.environ["GOOGLE_API_KEY"] = api_key

if __name__ == "__main__":
    # generate summaries json
    email_preprocessor = EmailPreprocessor()
    email_folder = os.path.join(os.getcwd(), "../emails")
    processed_output_file = os.path.join(os.getcwd(), "../out/email_summaries.json")
    email_preprocessor.generate_summaries(email_folder, processed_output_file)
