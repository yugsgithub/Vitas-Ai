# !pip install llama-cpp-python

from llama_cpp import Llama

# Load model with M2 GPU acceleration enabled
llm = Llama.from_pretrained(
    repo_id="Shubh769/Vitas-Ayurveda-Phi3",
    filename="ayurveda_phi3_q4km.gguf",
    n_ctx=4096,         # Increase context window
    n_gpu_layers=-1     # OFF-LOAD ALL LAYERS TO M2 GPU
)

print("\n🌿 Vitas Ayurveda AI initialized. (Type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    # Generate response
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are Vitas, an expert Ayurvedic AI. Provide structured, concise, and helpful advice."},
            {"role": "user", "content": user_input}
        ],
        temperature=0.65,
        top_p=0.92,
        max_tokens=1024
    )
    
    print(f"\nVitas: {response['choices'][0]['message']['content']}\n")
