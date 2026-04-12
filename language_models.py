import os 
import litellm
from config import OPEN_SOURCE_MODELS, HF_MODEL_NAMES, LITELLM_TEMPLATES, Model
from loggers import logger
from common import get_api_key

class LanguageModel():
    def __init__(self, model_name):
        self.model_name = Model(model_name)
    
    def batched_generate(self, prompts_list: list, max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError
    
class APILiteLLM(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "ERROR: API CALL FAILED."
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20

    def __init__(self, model_name):
        super().__init__(model_name)
        self.api_key = get_api_key(self.model_name)
        self.litellm_model_name = self.get_litellm_model_name(self.model_name)
        litellm.drop_params = True
        self.set_eos_tokens(self.model_name)
        
    def get_litellm_model_name(self, model_name):
        if model_name in OPEN_SOURCE_MODELS:
            hf_name = HF_MODEL_NAMES[model_name]
            litellm_name = f"huggingface/{hf_name}"
            self.use_open_source_model = True
        else:
            self.use_open_source_model = False
            litellm_name = model_name.value 
        return litellm_name
    
    def set_eos_tokens(self, model_name):
        if self.use_open_source_model:
            self.eos_tokens = LITELLM_TEMPLATES[model_name]["eos_tokens"]     
        else:
            self.eos_tokens = []

    def _update_prompt_template(self):
        if self.model_name in LITELLM_TEMPLATES:
            litellm.register_prompt_template(
                initial_prompt_value=LITELLM_TEMPLATES[self.model_name]["initial_prompt_value"],
                model=self.litellm_model_name,
                roles=LITELLM_TEMPLATES[self.model_name]["roles"]
            )
            self.post_message = LITELLM_TEMPLATES[self.model_name]["post_message"]
        else:
            self.post_message = ""
        
    def batched_generate(self, convs_list: list[list[dict]], 
                         max_n_tokens: int, 
                         temperature: float, 
                         top_p: float,
                         extra_eos_tokens: list[str] = None) -> list[str]: 
        
        eos_tokens = self.eos_tokens.copy()

        if extra_eos_tokens:
            eos_tokens.extend(extra_eos_tokens)
        if self.use_open_source_model:
            self._update_prompt_template()
        
        outputs = litellm.batch_completion(
            model=self.litellm_model_name, 
            messages=convs_list,
            api_key=self.api_key,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_n_tokens,
            num_retries=self.API_MAX_RETRY,
            seed=0,
            stop=eos_tokens,
        )
        
        responses = []
        for output in outputs:
            if isinstance(output, Exception):
                logger.error(f"API call failed: {output}")
                responses.append(self.API_ERROR_OUTPUT)
            else:
                responses.append(output["choices"][0]["message"].content)

        return responses


class LocalvLLM(LanguageModel):
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        import vllm
        from vllm.distributed.parallel_state import destroy_model_parallel
        self.hf_model_name = HF_MODEL_NAMES[self.model_name]
        destroy_model_parallel()
        self.model = vllm.LLM(model=self.hf_model_name)
        self.use_open_source_model = True
        self.post_message = LITELLM_TEMPLATES.get(self.model_name, {}).get("post_message", "")

    def batched_generate(self, convs_list: list[list[dict]], 
                         max_n_tokens: int, 
                         temperature: float, 
                         top_p: float,
                         extra_eos_tokens: list[str] = None) -> list[str]:
        import vllm
        from common import conv_template
        from config import FASTCHAT_TEMPLATE_NAMES

        template_name = FASTCHAT_TEMPLATE_NAMES[self.model_name]
        full_prompts = []
        for conv_messages in convs_list:
            conv = conv_template(template_name)
            for msg in conv_messages:
                if msg["role"] == "system":
                    conv.set_system_message(msg["content"])
                elif msg["role"] == "user":
                    conv.append_message(conv.roles[0], msg["content"])
                elif msg["role"] == "assistant":
                    conv.append_message(conv.roles[1], msg["content"])
            conv.append_message(conv.roles[1], None)
            full_prompt = conv.get_prompt()
            if "llama-2" in self.hf_model_name.lower():
                full_prompt += " "
            full_prompts.append(full_prompt)

        if temperature > 0:
            sampling_params = vllm.SamplingParams(
                temperature=temperature, top_p=top_p, max_tokens=max_n_tokens
            )
        else:
            sampling_params = vllm.SamplingParams(temperature=0, max_tokens=max_n_tokens)

        outputs = self.model.generate(full_prompts, sampling_params)
        outputs_list = [output.outputs[0].text.lstrip() for output in outputs]
        return outputs_list
