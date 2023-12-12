import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer


def load_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        add_eos_token=False,
        trust_remote_code=True,
        use_fast=False
    )

    
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            use_cache=False,
            low_cpu_mem_usage=True,
        ).to(device).eval()
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = 'left'
    return model, tokenizer

def get_nonascii_toks(tokenizer, device='cpu'):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)
    
    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)
    
    return torch.tensor(ascii_toks, device=device)

def remove_nonascii(tokenizer, output_tok, device):
    
    nonascii_toks = get_nonascii_toks(tokenizer=tokenizer, device=device)
    clean_output_tok = []
    for i in range(len(output_tok)):
        if output_tok[i] in nonascii_toks:
            continue
        else:
            clean_output_tok.append(output_tok[i])
    return clean_output_tok
    