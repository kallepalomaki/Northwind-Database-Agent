from transformers import AutoModelForCausalLM, AutoTokenizer

# Model identifier from Hugging Face
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Example input
prompt = "Generate python function which prints hello world"

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt")

# Generate response
outputs = model.generate(inputs['input_ids'], max_length=50)

# Decode and print the response
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
