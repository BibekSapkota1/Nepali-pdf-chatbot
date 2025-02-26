# import google.generativeai as genai
# from dotenv import load_dotenv
# import os

# load_dotenv()

# api_key = os.getenv("GEMINI_API_KEY")
# if not api_key:
#     raise ValueError("GEMINI_API_KEY not found in environment variables")

# try:
#     genai.configure(api_key=api_key)
#     models = list(genai.list_models())
#     for model in models:
#         print(model)
# except Exception as e:
#     print(f"Error: {e}")


# This is a test file to check if the gemini api key is working or not and print the models that are available