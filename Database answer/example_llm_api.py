"""
Example LLM API implementation for Google Colab.
This should be run on Google Colab and exposed via ngrok.

Usage:
1. Install dependencies: pip install flask transformers torch pyngrok
2. Run this script in Colab
3. Copy the ngrok URL and set it as LLM_API_URL in backend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize model (choose one)
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"  # or "google/gemma-7b"

print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Model loaded successfully!")


@app.route('/generate', methods=['POST'])
def generate():
    """
    Generate response from LLM.
    
    Expected request:
    {
        "prompt": "Your prompt here..."
    }
    
    Returns:
    {
        "response": "Generated text..."
    }
    """
    try:
        data = request.json
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the newly generated part (remove prompt)
        response = full_response[len(prompt):].strip()
        
        return jsonify({"response": response})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model": MODEL_NAME})


if __name__ == '__main__':
    # Expose via ngrok
    try:
        from pyngrok import ngrok
        public_url = ngrok.connect(5000)
        print(f"\n🚀 Public URL: {public_url}")
        print(f"Set this as LLM_API_URL in your backend!\n")
    except ImportError:
        print("pyngrok not installed. Install with: pip install pyngrok")
        print("Or manually expose port 5000 with ngrok")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)


