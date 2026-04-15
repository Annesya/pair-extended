from common import conv_template, extract_json
from language_models import APILiteLLM, LocalvLLM
from config import FASTCHAT_TEMPLATE_NAMES, Model


def load_attack_and_target_models(args):
    attackLM = AttackLM(model_name = args.attack_model, 
                        max_n_tokens = args.attack_max_n_tokens, 
                        max_n_attack_attempts = args.max_n_attack_attempts, 
                        category = args.category,
                        evaluate_locally = args.evaluate_locally
                        )
    
    share_model = (args.evaluate_locally 
                   and args.attack_model == args.target_model)
    
    targetLM = TargetLM(model_name = args.target_model,
                        category = args.category,
                        max_n_tokens = args.target_max_n_tokens,
                        evaluate_locally = args.evaluate_locally,
                        shared_model = attackLM.model if share_model else None,
                        )
    
    return attackLM, targetLM

def load_indiv_model(model_name, local=False):
    if local:
        return LocalvLLM(model_name)
    else:
        return APILiteLLM(model_name)

class AttackLM():
    """
        Base class for attacker language models.
        
        Generates attacks for conversations using a language model. The self.model attribute contains the underlying generation model.
    """
    def __init__(self, 
                model_name: str, 
                max_n_tokens: int, 
                max_n_attack_attempts: int, 
                category: str,
                evaluate_locally: bool):
        
        self.model_name = Model(model_name)
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts

        from config import ATTACK_TEMP, ATTACK_TOP_P
        self.temperature = ATTACK_TEMP
        self.top_p = ATTACK_TOP_P

        self.category = category
        self.evaluate_locally = evaluate_locally
        self.model = load_indiv_model(model_name, local=evaluate_locally)
        self.initialize_output = self.model.use_open_source_model
        self.template = FASTCHAT_TEMPLATE_NAMES[self.model_name]

    def preprocess_conversation(self, convs_list: list, prompts_list: list[str]):
        init_message = ""
        if self.initialize_output:
            if len(convs_list[0].messages) == 0:
                init_message = '{"improvement": "","prompt": "'
            else:
                init_message = '{"improvement": "'
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            if self.initialize_output:
                conv.append_message(conv.roles[1], init_message)
        openai_convs_list = [conv.to_openai_api_messages() for conv in convs_list]
        return openai_convs_list, init_message
        
    def _generate_attack(self, openai_conv_list: list[list[dict]], init_message: str):
        batchsize = len(openai_conv_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize
        new_adv_prompts = [None] * batchsize
        
        # Only use "}" as a stop token for open-source models;
        # API models (GPT, Claude, etc.) handle JSON termination on their own.
        eos = ["}"] if self.initialize_output else None
        
        for attempt in range(self.max_n_attack_attempts):
            convs_subset = [openai_conv_list[i] for i in indices_to_regenerate]
            outputs_list = self.model.batched_generate(convs_subset,
                                                        max_n_tokens = self.max_n_tokens,  
                                                        temperature = self.temperature,
                                                        top_p = self.top_p,
                                                        extra_eos_tokens=eos
                                                    )
            
            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]
                full_output = init_message + full_output + "}"
                attack_dict, json_str = extract_json(full_output)
                if attack_dict is not None:
                    valid_outputs[orig_index] = attack_dict
                    new_adv_prompts[orig_index] = json_str
                else:
                    new_indices_to_regenerate.append(orig_index)
            
            indices_to_regenerate = new_indices_to_regenerate
            if not indices_to_regenerate:
                break

        if any([output is None for output in valid_outputs]):
            raise ValueError(f"Failed to generate valid output after {self.max_n_attack_attempts} attempts. Terminating.")
        return valid_outputs, new_adv_prompts

    def get_attack(self, convs_list, prompts_list):
        """
        Generates responses for a batch of conversations and prompts using a language model. 
        Only valid outputs in proper JSON format are returned. If an output isn't generated 
        successfully after max_n_attack_attempts, it's returned as None.
        
        Parameters:
        - convs_list: List of conversation objects.
        - prompts_list: List of prompts corresponding to each conversation.
        
        Returns:
        - List of generated outputs (dictionaries) or None for failed generations.
        """
        assert len(convs_list) == len(prompts_list), "Mismatch between number of conversations and prompts."
        
        processed_convs_list, init_message = self.preprocess_conversation(convs_list, prompts_list)
        valid_outputs, new_adv_prompts = self._generate_attack(processed_convs_list, init_message)

        for jailbreak_prompt, conv in zip(new_adv_prompts, convs_list):
            if self.initialize_output:
                jailbreak_prompt += self.model.post_message
            conv.update_last_message(jailbreak_prompt)
        
        return valid_outputs

class TargetLM():
    """
        Target language model wrapper. Uses APILiteLLM for API-based models
        and LocalvLLM for local evaluation.
    """
    def __init__(self, 
            model_name: str, 
            category: str,
            max_n_tokens: int,
            evaluate_locally: bool = False,
            shared_model = None,
            ):
        
        self.model_name = Model(model_name)
        self.max_n_tokens = max_n_tokens
        self.evaluate_locally = evaluate_locally
        self.template = FASTCHAT_TEMPLATE_NAMES[self.model_name]

        from config import TARGET_TEMP, TARGET_TOP_P   
        self.temperature = TARGET_TEMP
        self.top_p = TARGET_TOP_P

        if shared_model is not None:
            self.model = shared_model
        else:
            self.model = load_indiv_model(model_name, local=evaluate_locally)
        self.category = category

    def get_response(self, prompts_list):
        batchsize = len(prompts_list)
        convs_list = [conv_template(self.template) for _ in range(batchsize)]
        full_prompts = []
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            full_prompts.append(conv.to_openai_api_messages())

        responses = self.model.batched_generate(full_prompts, 
                                                max_n_tokens = self.max_n_tokens,  
                                                temperature = self.temperature,
                                                top_p = self.top_p
                                                )
        return responses
