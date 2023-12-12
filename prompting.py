import torch

class PromptManager:
    def __init__(self, tokenizer, model, instructions, targets, test_instruction, is_defense=False):
        
        self.instructions = instructions
        self.targets = targets
        self.tokenizer=tokenizer
        self.model = model
        self.user_prompt = ''#"User:"
        self.assistant_prompt = ''#"\n\nAssistant:"
        self.new_instruction = test_instruction
        if is_defense:
            self.defense = '? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ?'
        else:
            self.defense = ''
            
    def prompts(self, defense_embeddings):
        result_prompts = []
        tot_len = 0
        for i in range(len(self.instructions)):
            self.instruction = self.instructions[i]
            self.target = self.targets[i]
            
            separator = ' ' if self.instruction else ''
            
            prompt = f'{self.user_prompt}{separator}{self.instruction}{separator}{self.assistant_prompt}{separator}{self.target}'
            prompt_len = len(self.tokenizer(prompt).input_ids)
            tot_len = tot_len+prompt_len
            result_prompts.append(prompt)
            
        prompt_tmp = f'{self.user_prompt}'
        toks = self.tokenizer(prompt_tmp).input_ids
        self._user_role_slice = slice(None, len(toks))
        
        prompt_tmp = f'{self.user_prompt}{separator}{self.new_instruction}'
        toks = self.tokenizer(prompt_tmp).input_ids
        self._ins_role_slice = slice(self._user_role_slice.stop, len(toks))
        
        prompt_tmp = f'{self.user_prompt}{separator}{self.new_instruction}{separator}{self.defense}'
        toks = self.tokenizer(prompt_tmp).input_ids
        self._defense_slice = slice(self._ins_role_slice.stop, len(toks))
        
        prompt_tmp = f'{self.user_prompt}{separator}{self.new_instruction}{separator}{self.defense}{self.assistant_prompt}'
        toks = self.tokenizer(prompt_tmp).input_ids
        self._assistant_role_slice = slice(self._defense_slice.stop, len(toks))

        
        self._new_defense_slice = slice(self._defense_slice.start+tot_len, self._defense_slice.stop+tot_len)
        result_prompts.append(prompt_tmp)
        #print(self.tokenizer('\n\n').input_ids)
        result = '\n\n'.join(result_prompts)
        results_toks = self.tokenizer(result).input_ids
        results_embeds = self.model.model.embed_tokens(torch.tensor(results_toks, device = self.model.device).unsqueeze(0))

        new_results_embeds = []
        #tmp = torch.zeros((20, self.model.model.embed_tokens.embedding_dim), device=self.model.device, requires_grad=False)
        new_results_embeds.append(torch.cat([results_embeds[0,:self._new_defense_slice.start,:], defense_embeddings, results_embeds[0,self._new_defense_slice.stop:,:]], 0))
        

        new_results_embeds = torch.stack(new_results_embeds, dim=0)

        assert torch.all(new_results_embeds[0,self._new_defense_slice,:]==defense_embeddings)#defense_embeddings)
        
        return new_results_embeds