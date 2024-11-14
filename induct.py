import argparse
import json
import numpy as np
import pandas as pd
import torch

from PIL import Image
from transformers import pipeline, AutoProcessor, MllamaForConditionalGeneration, BitsAndBytesConfig

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='SHTech', choices=['SHTech', 'avenue', 'ped2', 'UBNormal'], help='Dataset name')
    parser.add_argument('--root', type=str, default='/data/tjf5667/datasets/', help='Root directory for datasets')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--activity_limit', type=int, default=10, help='Maximum amount of activities to derive')
    parser.add_argument('--object_limit', type=int, default=10, help='Maximum amount of objects to derive')
    return parser.parse_args()

class Induction:
    def __init__(self, vlm_model, instruct_model, processor, args):
        self.vlm_model = vlm_model
        self.instruct_model = instruct_model
        self.processor = processor
        self.args = args
        self.frame_paths = None
        self.vlm_message = None
        self.frame_descriptions = []
        self.normal_activities = None
        self.abnormal_activities = None
        self.normal_objects = None
        self.abnormal_objects = None

    def select_frames(self):
        df = pd.read_csv(f'{self.args.data}/train.csv')
        image_file_paths = list(df.loc[df['label'] == 0, 'image_path'].values)
        random_frame_paths = np.random.choice(image_file_paths, self.args.batch_size, replace=False)
        self.frame_paths = [f"{self.args.root}{path}" for path in random_frame_paths]

    def process_vlm_message(self):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "How many people are in the image and what is each of them doing? What are in the images other than people? Think step by step."}]}]
        self.vlm_message = self.processor.apply_chat_template(messages, add_generation_prompt=True)

    def generate_frame_descriptions(self):
        for frame_path in self.frame_paths:
            image = Image.open(frame_path).convert('RGB')
            input = self.processor(
                image,
                self.vlm_message,
                add_special_tokens=False,
                return_tensors="pt"
            ).to('cuda')

            print('Generating frame_description for:', frame_path)
            with torch.no_grad():
                output = self.vlm_model.generate(**input, max_new_tokens=128)
                decoded_output = self.processor.decode(output[0])
                self.frame_descriptions.append(decoded_output)

    def extract_rules(self, unparsed_rules):
        rules_set = set()
        for line in unparsed_rules['content'].split('\n'):
            if line.strip() and line[0].isdigit():
                if '. ' in line:
                    rule = line.split('. ', 1)[1].lower()
                    rules_set.add(rule)

        rules = list(rules_set)
        return rules

    def generate_rules(self, prompt):
        messages = [{"role": "system", "content": "You are a surveillance monitor for urban safety"}, {"role": "user", "content": {prompt}},]
        output = self.instruct_model(
            messages, 
            max_new_tokens=128, 
            pad_token_id=50256
        )

        unparsed_rules = output[0]["generated_text"][-1]
        rules = self.extract_rules(unparsed_rules)
        return rules
    
    def save_rules(self):
        output_data = {
            "normal_activities": self.normal_activities,
            "abnormal_activities": self.abnormal_activities,
            "normal_objects": self.normal_objects,
            "abnormal_objects": self.abnormal_objects
        }

        output_file = f"{self.args.data}_rules.json"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=4)

        print(f"Rules saved to {output_file}")

class Prompts:
    def __init__(self, frame_descriptions, activity_limit, object_limit):
            self.frame_descriptions = frame_descriptions
            self.activity_limit = activity_limit
            self.object_limit = object_limit

    def normal_activities(self):
            return f"""Given the frame descriptions {self.frame_descriptions}, please list at most {self.activity_limit} unique human activities from these frame descriptions.
            List them using short terms, not an entire sentence. Output the information in the following format, with no deviations:
            Example:
            1. Walking
            2. Standing
            
            Answer:
            """
    
    def normal_objects(self):
            return f"""Given the frame descriptions {self.frame_descriptions}, please list at most {self.object_limit} unique environmental objects from these frame descriptions.
            List them using short terms, not an entire sentence. Output the information in the following format, with no deviations:
            Example: 
            1. Pathway
            2. Building
            
            Answer:
            """

    def abnormal_activities(self, normal_activities):
            return f"""Given these normal activities {normal_activities}, please list at most {self.activity_limit} potential abnormal human activities from the context of these normal activites.
            List them using short terms, not an entire sentence. Output the information in the following format, with no deviations:
            Example:
            1. Fighting
            2. Running

            Answer:
            """
    
    def abnormal_objects(self, normal_objects):
            return f"""Given these normal objects {normal_objects}, please list at most {self.object_limit} potential abnormal environmental objects from the context of these normal objects.
            List them using short terms, not an entire sentence. Output the information in the following format, with no deviations:
            Example:
            1. Car
            2. Weapon
            
            Answer:
            """
        
if __name__ == "__main__":
    args = parse_arguments()
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    vlm_model = MllamaForConditionalGeneration.from_pretrained(
        'meta-llama/Llama-3.2-11B-Vision-Instruct',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        device_map='auto'
    )

    instruct_model = pipeline(
        task="text-generation",
        model="meta-llama/Llama-3.2-3B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained('meta-llama/Llama-3.2-11B-Vision-Instruct')

    inductor = Induction(vlm_model, instruct_model, processor, args)
    inductor.select_frames()
    inductor.process_vlm_message()
    inductor.generate_frame_descriptions()

    prompts = Prompts(inductor.frame_descriptions, args.activity_limit, args.object_limit)
    inductor.normal_activities = inductor.generate_rules(prompts.normal_activities())
    inductor.abnormal_activities = inductor.generate_rules(prompts.abnormal_activities(inductor.normal_activities)) if inductor.normal_activities else None
    inductor.normal_objects = inductor.generate_rules(prompts.normal_objects())
    inductor.abnormal_objects = inductor.generate_rules(prompts.abnormal_objects(inductor.normal_objects)) if inductor.normal_objects else None

    inductor.save_rules()