from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/phi-1_5"  # Replace with the correct Phi model name
cache_dir = "./models/phi"  # Specify a local directory for caching

# Download and cache model locally
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

input_text = "How do neural networks work in ML?"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate response
output = model.generate(**inputs, max_length=500, num_return_sequences=1)

# Decode the output
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_output)