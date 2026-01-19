#!/bin/bash

# Example curl request to LLM API
# Replace YOUR_NGROK_URL with your actual ngrok URL

NGROK_URL="http://YOUR_NGROK_URL.ngrok.io"

curl -X POST "${NGROK_URL}/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "You are a helpful assistant that answers questions based ONLY on the provided context.\n\nCONTEXT:\nThis is a sample context about machine learning. Machine learning is a subset of artificial intelligence.\n\nQUESTION: What is machine learning?\n\nINSTRUCTIONS:\n- Answer the question using ONLY the information provided in the context above.\n- If the answer cannot be found in the context, respond with exactly: \"Answer not found in the documents.\"\n- Do not use any external knowledge or information not present in the context.\n- Be concise and accurate.\n\nANSWER:"
  }'


