# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
# model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto", torch_dtype=torch.float32)

# input_text = "Suppose you are walking on a beach and you see a turtle. What do you do next?"
# input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# outputs = model.generate(**input_ids, max_new_tokens=100)
# print(tokenizer.decode(outputs[0]))