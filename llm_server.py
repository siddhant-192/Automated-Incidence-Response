from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import sys

# Initialize Flask app
app = Flask(__name__)

# Define model configuration
model_name = 'google/gemma-2b' # Light Model
#model_name = 'meta-llama/Llama-3.1-8B'  # Heavy Model
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")  # Ensure this environment variable is set

# Check if Hugging Face token is set; if not, exit with an error message
if not hf_token:
    print("Error: Hugging Face token not found. Please set the HUGGING_FACE_HUB_TOKEN environment variable.")
    sys.exit(1)

# Define the model directory paths
model_dir = os.path.join("model", model_name.replace('/', '_'))  # Store in 'model/model_name'
tokenizer_dir = os.path.join(model_dir, "tokenizer")

# Load model and tokenizer with error handling
try:
    if not os.path.exists(model_dir):  # Model not found locally
        print("Model not found locally. Downloading and saving...")

        # Download and save the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_dir)

        # Download and save the full model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.bfloat16
        )
        os.makedirs(model_dir, exist_ok=True)
        model.save_pretrained(model_dir)
        print("Model and tokenizer downloaded and saved locally.")
    else:
        print("Loading model and tokenizer from local storage with disk offloading...")
        # Load tokenizer and model from local directories
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

        # Load model with disk offloading
        from accelerate import load_checkpoint_and_dispatch
        model = load_checkpoint_and_dispatch(
            AutoModelForCausalLM.from_pretrained(model_dir),
            model_dir,
            device_map="auto",
            offload_folder=model_dir,  # Folder for disk offloading
            offload_state_dict=True,
            dtype=torch.bfloat16
        )
        print("Model and tokenizer loaded from local storage with disk offloading.")

except Exception as e:
    print("An error occurred during model loading or downloading:", str(e))
    sys.exit(1)

# /generate endpoint
@app.route("/generate", methods=["POST"])
def generate_text():
    try:
        # Get JSON input data
        data = request.get_json()
        input_text = data.get("input_text", "")
        
        # Set max_length to a desired value, e.g., 100 tokens
        max_length = data.get("max_length", 100)  # Allow max_length to be set via input, defaulting to 100

        # Tokenize and generate text with specified max_length
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_length=max_length)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Return the generated text as JSON response
        return jsonify({"generated_text": generated_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the server
if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5010)
    except Exception as e:
        print("Failed to start the Flask server:", str(e))
        sys.exit(1)

