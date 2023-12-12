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
defense_embeddings = np.load(f'/root/autodl-tmp/FastChat/fastchat/llm_judge/final_defense_embeddings_iter10_1e-7.npy')
        
defense_embeddings_torch = torch.tensor(defense_embeddings, device=model.device, requires_grad = False)
f = open('/root/autodl-tmp/FastChat/fastchat/llm_judge/data/outputs.json')
data = json.load(f)
model_id='llama2'
results=[]
conv = get_conversation_template(model_id)
print('with defense')
for j in tqdm(range(len(data))):
    qs = data[j]['instruction']
    #print(qs)
    new_input_embeds=[]
    if model_id =='llama2':
        
        defense_string = '? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ?'
        conv.messages = []
        conv.append_message(conv.roles[0], None)
        toks = tokenizer([conv.get_prompt()], return_tensors="pt").input_ids
        user_role_slice = slice(None, len(toks[0]))
                    
        conv.update_last_message(f"{qs}")
        toks = tokenizer([conv.get_prompt()], return_tensors="pt").input_ids
        goal_slice = slice(user_role_slice.stop, max(user_role_slice.stop, len(toks[0]))-4)
                    
        separator = ' ' if qs else ''
        conv.update_last_message(f"{qs}{separator}{defense_string}")
        toks = tokenizer([conv.get_prompt()], return_tensors="pt").input_ids
        defense_slice = slice(goal_slice.stop, len(toks[0])-4)
        conv.append_message(conv.roles[1], None)
        toks = tokenizer([conv.get_prompt()], return_tensors="pt").input_ids
        assistant_role_slice = slice(defense_slice.stop, len(toks[0]))

        input_embeds = model.model.embed_tokens(toks.to(model.device))
        #print(input_embeds.shape)
        new_input_embeds.append(torch.cat([input_embeds[0][:defense_slice.start,:], defense_embeddings_torch, input_embeds[0][defense_slice.stop:,:]], 0))
        

        new_input_embeds = torch.stack(new_input_embeds, dim=0)
        #print(new_input_embeds.shape)
        #print(new_input_embeds[0][defense_slice])
        assert torch.all(new_input_embeds[0,defense_slice,:]==defense_embeddings_torch)
        try:
            output_ids = model.generate(
                inputs_embeds=new_input_embeds,
                temperature=0.7,
                max_new_tokens=1024,
            )
            output_ids = output_ids[0]
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
                    'generator':'llama2_defense_helpful_iter10', 
                    'dataset':data[j]['dataset'], 
                    'datasplit':data[j]['datasplit']}
        results.append(new_dict)
        
with open("/root/autodl-tmp/FastChat/fastchat/llm_judge/data/llama2_outputs_defense_helpful_iter10.json", "w") as final:
    json.dump(results, final)
        