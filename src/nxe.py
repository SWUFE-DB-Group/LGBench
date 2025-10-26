import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from pathlib import Path


current_dir = Path(__file__).parent
project_root = current_dir.parent
config_path = project_root / "config.json"
with open(config_path, "r") as f:
    cfg = json.load(f)
model_path = cfg["model_path"]
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# load model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=torch.float32,
    device_map=None
)

model.to(device)


def calculate_next_token_perplexity(text, model, tokenizer, top_k=5):
    """
    Calculate the perplexity of next token prediction for the given text

    Args:
        text (str): Input text
        model: Pre-trained language model
        tokenizer: Corresponding tokenizer
        top_k (int): Number of top tokens to return by probability

    Returns:
        dict: Contains perplexity, probability distribution, top-k predictions, and other information
    """
    # Encode the text
    encodings = tokenizer(text, return_tensors='pt', truncation=False)
    input_ids = encodings['input_ids']
    attention_mask = encodings.get('attention_mask', None)

    # Move data to the device where the model is located
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        # Get model output
        if attention_mask is not None:
            outputs = model(input_ids, attention_mask=attention_mask)
        else:
            outputs = model(input_ids)
        # Get logits at the last position (for predicting the next token)
        last_logits = outputs.logits[0, -1, :]  # shape: [vocab_size]
        probs = F.softmax(last_logits, dim=-1)
        # perplexity = exp(-sum(p * log(p))) where p is probability
        # This is not perplexity in the traditional sense (https://huggingface.co/docs/transformers/perplexity)
        log_probs = F.log_softmax(last_logits, dim=-1)
        entropy = -(probs * log_probs).sum()
        # print("entropy", entropy)
        perplexity = torch.exp(entropy).item()

        # Get top-k predictions
        top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)

        # Decode top-k tokens
        top_k_tokens = []
        for i in range(top_k):
            token_id = top_k_indices[i].item()
            token_text = tokenizer.decode([token_id])
            top_k_tokens.append({
                'token_id': token_id,
                'token_text': token_text,
                'probability': top_k_probs[i].item()
            })

    return {
        'input_text': text,
        'perplexity': perplexity,
        'entropy': entropy.item(),
        'vocab_size': last_logits.size(0),
        'top_k_predictions': top_k_tokens,
        'total_probability_mass': sum(token['probability'] for token in top_k_tokens)
    }

def get_nxe_ppl(text):
    messages = [
        {'role': 'system', 'content': 'rephrase what the user inputs using the language provided'},
        {"role": "user", "content": text}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    next_token_result = calculate_next_token_perplexity(text, model, tokenizer)
    return next_token_result['perplexity']

# example
if __name__ == "__main__":
    print(get_nxe_ppl("好累啊"))

