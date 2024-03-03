from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model4bit = AutoModelForCausalLM.from_pretrained("google/gemma-2b", quantization_config=quantization_config)

def generate_text_gemma(prompt):
    prefix = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly. Limit the conversation to 100 characters. "
    
    input_ids = tokenizer(prefix + prompt, return_tensors="pt").to("cuda")

    outputs = model4bit.generate(**input_ids, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prefix, "").replace(prompt, "").replace("<bos>", "").replace("<eos>","").replace("\n","").replace("Answer:", "").strip()