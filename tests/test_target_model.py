import unittest
from conversers import TargetLM



class TestTargetVicuna(unittest.TestCase):
    def setUp(self):
        self.lm = TargetLM(model_name="vicuna-13b-v1.5",
            category="unittest",
            max_n_tokens=10,
            evaluate_locally=False)

    def test_get_response(self):
        prompts_list=["What is the meaning of life?", "Write a poem about the ocean."]
        num_prompts = len(prompts_list)
        responses = self.lm.get_response(prompts_list)
                                          
        self.assertEqual(len(responses), num_prompts)

class TestTargetGPT35(unittest.TestCase):
    def setUp(self):
        self.lm = TargetLM(model_name="gpt-3.5-turbo-1106",
            category="unittest",
            max_n_tokens=10,
            evaluate_locally=False)

    def test_get_response(self):
        prompts_list=["What is the meaning of life?", "Write a poem about the ocean."]
        num_prompts = len(prompts_list)
        responses = self.lm.get_response(prompts_list)
                                          
        self.assertEqual(len(responses), num_prompts)
