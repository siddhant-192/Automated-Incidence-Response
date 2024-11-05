from flask import Flask, request, jsonify, Response
from openai import OpenAI
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Set up the NVIDIA API client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")  # Ensure this environment variable is set
)

# Check if the NVIDIA API key is set; if not, exit with an error message
if not client.api_key:
    print("Error: NVIDIA API key not found. Please set the NVIDIA_API_KEY environment variable.")
    sys.exit(1)

# /generate endpoint
@app.route("/generate", methods=["POST"])
def generate_text():
    try:
        # Get JSON input data
        data = request.get_json()
        input_text = data.get("input_text", "")
        
        # Set max_tokens to a desired value, e.g., 1000 tokens
        max_tokens = data.get("max_tokens", 1000)  # Allow max_tokens to be set via input, defaulting to 1000

        # Define parameters for text generation
        completion = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            messages=[{"role": "user", "content": input_text}],
            temperature=0.5,
            top_p=1,
            max_tokens=max_tokens,
            stream=True
        )

        # Stream the response chunks to the client
        def stream_response():
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        # Return the streamed response as a text event stream
        return Response(stream_response(), content_type="text/event-stream")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the server
if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5010)
    except Exception as e:
        print("Failed to start the Flask server:", str(e))
        sys.exit(1)