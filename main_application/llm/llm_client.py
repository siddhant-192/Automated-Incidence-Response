# import requests

# def generate_text(input_text, url="http://localhost:5010/generate"):
#     """
#     Sends a request to the LLM server to generate text based on the input text.

#     Parameters:
#         input_text (str): The input text to be processed by the model.
#         url (str): The server URL endpoint for text generation (default is localhost:5010).

#     Returns:
#         str: The generated text from the model if the request is successful, or an error message.
#     """
#     data = {
#         "input_text": input_text,
#         "max_tokens": 15000,
#     }
    
#     try:
#         # Send the POST request to the server with streaming enabled
#         response = requests.post(url, json=data, stream=True)

#         # Check if the request was successful
#         if response.status_code == 200:
#             # Collect streamed content
#             result_text = ""
#             for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
#                 result_text += chunk  # Append each chunk of data

#             return result_text  # Return the complete response
#         else:
#             print("Failed to get a response:", response.status_code, response.text)
#             return None
#     except requests.RequestException as e:
#         print("Request failed:", e)
#         return None

import requests
import time

def generate_text(input_text, url="http://localhost:5010/generate"):
    data = {
        "input_text": input_text,
        "max_tokens": 15000,
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=data, stream=True, timeout=30)

            if response.status_code == 200:
                result_text = ""
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        result_text += chunk
                if result_text:  # Ensure content was received
                    return result_text
                else:
                    print("Warning: Received empty response.")
            else:
                print(f"Failed to get a response: {response.status_code} {response.text}")

        except requests.RequestException as e:
            print(f"Request failed: {e}")

        # Retry logic if the request fails
        if attempt < max_retries - 1:
            print("Retrying...")
            time.sleep(2)
        else:
            print("Max retries reached. Exiting.")

    return None  # Return None if all attempts fail
