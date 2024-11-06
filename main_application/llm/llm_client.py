import requests

def generate_text(input_text, url="http://localhost:5010/generate"):
    """
    Sends a request to the LLM server to generate text based on the input text.

    Parameters:
        input_text (str): The input text to be processed by the model.
        url (str): The server URL endpoint for text generation (default is localhost:5010).

    Returns:
        str: The generated text from the model if the request is successful, or an error message.
    """
    data = {
        "input_text": input_text,
        "max_tokens": 30000,
    }
    
    try:
        # Send the POST request to the server with streaming enabled
        response = requests.post(url, json=data, stream=True)

        # Check if the request was successful
        if response.status_code == 200:
            # Collect streamed content
            result_text = ""
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                result_text += chunk  # Append each chunk of data

            return result_text  # Return the complete response
        else:
            print("Failed to get a response:", response.status_code, response.text)
            return None
    except requests.RequestException as e:
        print("Request failed:", e)
        return None