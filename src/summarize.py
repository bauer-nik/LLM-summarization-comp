import requests
import os

# TODO: Update local server IP to remote server IP
SERVER_IP = os.getenv("DEEPSEEK_SERVER_IP", "127.0.0.1")
PORT = os.getenv("DEEPSEEK_PORT", "3050")

def get_t5_summary(text, model, tokenizer, device):
    prompt = f"SUMMARIZE: {text}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    inputs = inputs.input_ids.to(device)
    output_ids = model.generate(inputs, min_length=30, max_length=90)
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary

def get_gpt_summary(text, model, tokenizer, device):
    # prompt = f"SUMMARIZE the following article: {text}"
    prompt = f"ARTICLE: {text}. END OF ARTICLE. SUMMARIZE THE ARTICLE: "
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    inputs = inputs.to(device)
    outputs = model.generate(**inputs, max_new_tokens=90)
    generated_tokens = outputs[:, inputs.input_ids.size()[1]:]
    summary = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return summary


def get_deepseek_summary(text):
    prompt = f"ARTICLE: {text} ARTICLE SUMMARY: "

    url = f"http://{SERVER_IP}:{PORT}/summarize"
    data = {"prompt": prompt}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=data, headers=headers)

    out = response.json()
    return out['summary']
