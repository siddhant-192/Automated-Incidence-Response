from flask import Flask, request, jsonify, Response
import openai
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure this environment variable is set

# Check if the OpenAI API key is set; if not, exit with an error message
if not openai.api_key:
    print("Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
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
        completion = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": input_text}],
            temperature=0.5,
            top_p=1,
            max_tokens=max_tokens,
            stream=True
        )

        # Stream the response chunks to the client
        def stream_response():
            for chunk in completion:
                if "choices" in chunk and chunk["choices"][0]["delta"].get("content") is not None:
                    yield chunk["choices"][0]["delta"]["content"]

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
