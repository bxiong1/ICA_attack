import argparse
import json
import os
import random
import time

import shortuuid
import torch
from tqdm import tqdm
import numpy as np
from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template

model, tokenizer = load_model(
        '/root/autodl-tmp/Llama-2-7b-chat-fp16',
        device="cuda",
        num_gpus=1,
        max_gpu_memory=None,
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
)
defense_embeddings = np.load(f'/root/autodl-tmp/FastChat/fastchat/llm_judge/final_defense_embeddings_20_best_iter2.npy')
        
defense_embeddings_torch = torch.tensor(defense_embeddings, device=model.device, requires_grad = False)
f = open('/root/autodl-tmp/FastChat/fastchat/llm_judge/data/outputs.json')
data = json.load(f)
model_id='llama2'
results=[]
conv = get_conversation_template(model_id)
print('without defense')
for j in tqdm(range(len(data))):
    qs = data[j]['instruction']
    #print(qs)
    new_input_embeds=[]
    if model_id =='llama2':
        conv.messages = []
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer([prompt]).input_ids
        #assert torch.all(new_input_embeds[0,defense_slice,:]==defense_embeddings_torch)
        try:
            output_ids = model.generate(
                torch.as_tensor(input_ids).cuda(),
                temperature=0.7,
                max_new_tokens=1024,
            )
            output_ids = output_ids[0][len(input_ids[0]) :]
            if conv.stop_token_ids:
                stop_token_ids_index = [
                    i
                    for i, id in enumerate(output_ids)
                    if id in conv.stop_token_ids
                ]
                if len(stop_token_ids_index) > 0:
                    output_ids = output_ids[: stop_token_ids_index[0]]

            output = tokenizer.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )
            if conv.stop_str and output.find(conv.stop_str) > 0:
                output = output[: output.find(conv.stop_str)]
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()
            #print(output)
        except RuntimeError as e:
            print("ERROR question: ", qs)
            output = "ERROR"
                
        
        
        new_dict = {'instruction':qs, 
                    'output':output, 
                    'generator':'llama2_original', 
                    'dataset':data[j]['dataset'], 
                    'datasplit':data[j]['datasplit']}
        results.append(new_dict)
        
with open("/root/autodl-tmp/FastChat/fastchat/llm_judge/data/llama2_outputs_original.json", "w") as final:
    json.dump(results, final)