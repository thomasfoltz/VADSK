import argparse
import json
import pandas as pd
import torch
import re

from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration, BitsAndBytesConfig

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='SHTech', choices=['SHTech', 'avenue', 'ped2', 'UBNormal'], help='Dataset name')
    parser.add_argument('--root', type=str, default='/data/tjf5667/datasets/', help='Root directory for datasets')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--video_name', type=str, default='test_01_0014', help='Name of the clip for deduction')

    return parser.parse_args()

class Deduction:
    def __init__(self, vlm_model, processor, args):
        self.vlm_model = vlm_model
        self.processor = processor
        self.args = args
        self.labels = None
        self.frame_paths = None
        self.vlm_message = None
        self.video_name = self.args.video_name
        self.continuous_frame_description = []
        self.keywords = None
        self.feature_input = []

    def set_frame_paths(self):
        self.labels = pd.read_csv(f'{self.args.data}/test_frame/{self.video_name}.csv').iloc[:, 1].tolist()
        frame_paths = pd.read_csv(f'{self.args.data}/test_frame/{self.video_name}.csv').iloc[:, 0].tolist()
        self.frame_paths = [f"{self.args.root}{path}" for path in frame_paths]

    def process_vlm_message(self):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "How many people are in the image and what is each of them doing? What are in the images other than people? Think step by step."}]}]
        self.vlm_message = self.processor.apply_chat_template(messages, add_generation_prompt=True)

    def extract_frame_description(self, unparsed_frame_description):
        unparsed_frame_description = unparsed_frame_description.split('<|end_header_id|>\n\n')[2]
        frame_description = unparsed_frame_description.replace('.', ' ').replace('\n', '').replace('<|eot_id|>', '').lower()
        self.continuous_frame_description.append(frame_description)

    def generate_continuous_frame_description(self):
        print(f'Generating continuous_frame_description for {self.video_name}')
        for frame_path in self.frame_paths:
            image = Image.open(frame_path).convert('RGB')
            input = self.processor(
                image,
                self.vlm_message,
                add_special_tokens=False,
                return_tensors="pt"
            ).to('cuda')

            print(frame_path)
            with torch.no_grad():
                output = self.vlm_model.generate(**input, max_new_tokens=128)
                unparsed_frame_description = self.processor.decode(output[0])
                self.extract_frame_description(unparsed_frame_description)

    def set_keywords(self):
        print(self.args.data)
        with open(f'{self.args.data}_rules.json', 'r') as f:
            rules = json.load(f)
        self.keywords = rules['normal_activities'] + rules['abnormal_activities'] + rules['normal_objects'] + rules['abnormal_objects']

    def continuous_frame_descriptions_to_features(self):
        num_keywords = len(self.keywords)
        num_frames = len(self.continuous_frame_description)
        self.feature_input = torch.zeros((num_keywords, num_frames), dtype=torch.float32)

        for frame_idx, frame_description in enumerate(self.continuous_frame_description):
            for keyword_idx, keyword in enumerate(self.keywords):
                if re.search(r'\b' + keyword + r'\b', frame_description):
                    self.feature_input[keyword_idx, frame_idx] = 1.0

    def show_frame_feature_matches(self, frame_idx):
        match_idx = torch.where(self.feature_input[:, frame_idx] == 1.0)[0].tolist()
        print(f'Frame {frame_idx}: {[self.keywords[idx] for idx in match_idx]}')

    def accuracy_score():
        pass

    def precision_score():
        pass

    def recall_score():
        pass

    def roc_auc_score():
        pass

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

    processor = AutoProcessor.from_pretrained('meta-llama/Llama-3.2-11B-Vision-Instruct')

    deductor = Deduction(vlm_model, processor, args)
    deductor.set_frame_paths()
    deductor.process_vlm_message()
    deductor.generate_continuous_frame_description()

    with open(f"{args.video_name}.json", 'w') as f:
        json.dump(deductor.continuous_frame_description, f)

    # with open(f"{args.video_name}.json", 'r') as f:
    #     deductor.continuous_frame_description = json.load(f)

    deductor.set_keywords()
    deductor.continuous_frame_descriptions_to_features()

    for frame_idx in range(deductor.feature_input.shape[1]):
        deductor.show_frame_feature_matches(frame_idx)

    # TODO: organize file paths for rules and frame descriptions