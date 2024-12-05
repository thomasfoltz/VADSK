import argparse
import csv
import json
import pandas as pd
import torch
import re
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration, BitsAndBytesConfig
from shield_layer import ShieldLayer
from vadsr import VADSR
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='SHTech', choices=['SHTech', 'avenue', 'ped2', 'UBNormal'], help='Dataset name')
    parser.add_argument('--root', type=str, default='/data/tjf5667/datasets/', help='Root directory for datasets')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--train', action='store_true')

    return parser.parse_args()

class Deduction:
    def __init__(self, vlm_model, processor, args):
        self.vlm_model = vlm_model
        self.processor = processor
        self.args = args
        self.dataset_split = 'train' if args.train else 'test'
        self.labels = None
        self.frame_paths = None
        self.vlm_message = None
        self.keywords = None
        self.grouped_frames = None
        self.classification_model = None
        self.criterion = None
        self.optimizer = None

    def set_frame_paths(self):
        self.labels = pd.read_csv(f'{self.args.data}/{self.dataset_split}.csv').iloc[:, 1].tolist()
        self.frame_paths = pd.read_csv(f'{self.args.data}/{self.dataset_split}.csv').iloc[:, 0].tolist()

    def process_vlm_message(self):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "How many people are in the image and what is each of them doing? What are in the images other than people? Think step by step."}]}]
        self.vlm_message = self.processor.apply_chat_template(messages, add_generation_prompt=True)

    def parse_frame_description(self, unparsed_frame_description):
        split_frame_description = unparsed_frame_description.split('<|end_header_id|>\n\n')[2]
        frame_description = split_frame_description.replace('.', ' ').replace('\n', '').replace('<|eot_id|>', '').lower()
        return frame_description

    def generate_frame_descriptions(self):
        for frame_path, label in zip(self.frame_paths, self.labels):
            print(frame_path)
            image = Image.open(f"{self.args.root}{frame_path}").convert('RGB')
            input = self.processor(
                image,
                self.vlm_message,
                add_special_tokens=False,
                return_tensors="pt"
            ).to('cuda')

            with torch.no_grad():
                output = self.vlm_model.generate(**input, max_new_tokens=128)
                unparsed_frame_description = self.processor.decode(output[0])
            
            frame_description = self.parse_frame_description(unparsed_frame_description)

            with open(f"{self.args.data}/{self.dataset_split}_descriptions.csv", 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([frame_path, label, frame_description])

    def set_keywords(self):
        with open(f'{self.args.data}/rules.json', 'r') as f:
            rules = json.load(f)
        self.keywords = [None] + rules['normal_activities'] + [None] + rules['abnormal_activities'] + [None] + rules['normal_objects'] + [None] + rules['abnormal_objects']

    def group_frames(self):
        with open(f'{args.data}/{deductor.dataset_split}_descriptions.csv', 'r') as f:
            df = pd.read_csv(f, header=None, names=['frame_path', 'label', 'description'])
            if args.train:
                df['video_id'] = df['frame_path'].apply(lambda x: '_'.join(x.split('/')[-1].split('_')[:2]))
            else:
                df['video_id'] = df['frame_path'].apply(lambda x: x.split('/')[-2])
            self.grouped_frames = df.groupby('video_id')

    def frame_descriptions_to_features(self, frame_descriptions):
        num_keywords = len(self.keywords)
        num_frames = len(frame_descriptions)
        self.feature_input = torch.zeros((num_frames, num_keywords), dtype=torch.float32)

        for frame_idx, description in enumerate(frame_descriptions):
            for keyword_idx, keyword in enumerate(self.keywords):
                if keyword is not None and re.search(r'\b' + keyword + r'\b', description):
                    self.feature_input[frame_idx, keyword_idx] = 1.0

        return self.feature_input

    def show_frame_feature_matches(self, frame_idx):
        match_idx = torch.where(self.feature_input[frame_idx, :] == 1.0)[0].tolist()
        print(f'Frame {frame_idx}: {[self.keywords[idx] for idx in match_idx]}')

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
    deductor.generate_frame_descriptions()
    # deductor = Deduction(None, None, args)
    # deductor.set_keywords()
    # deductor.group_frames()

    # for video_id, group in deductor.grouped_frames:
    #     print(f"Video ID: {video_id}")
    #     feature_input = deductor.frame_descriptions_to_features(group['description'].tolist())

    #     for frame_idx in range(feature_input.shape[0]):
    #         deductor.show_frame_feature_matches(frame_idx)
    
    #     feature_num = feature_input.shape[-1]
    #     shield_layer = ShieldLayer('SHTech', feature_num)
    #     corrected_feature_input = shield_layer.correct_features(feature_input.clone())

    #     if not deductor.classification_model:
    #         print('loading classification model')
    #         deductor.classification_model = VADSR(input_size=feature_num, n=corrected_feature_input.shape[0])
    #         deductor.criterion = nn.BCELoss()
    #         deductor.optimizer = optim.Adam(deductor.classification_model.parameters(), lr=0.001)

    #     labels = group['label'].tolist()
    #     labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(0)
    
    #     num_epochs = 200
    #     for epoch in range(num_epochs):
    #         deductor.optimizer.zero_grad()
    #         outputs = deductor.classification_model(corrected_feature_input)
    #         loss = deductor.criterion(outputs, labels)
    #         loss.backward()
    #         deductor.optimizer.step()
                
    #         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    #     predictions = outputs.round().int().squeeze()
    #     true_labels = labels.round().int().squeeze()

    #     accuracy = accuracy_score(true_labels, predictions)
    #     precision = precision_score(true_labels, predictions)
    #     recall = recall_score(true_labels, predictions)
    #     roc_auc = roc_auc_score(true_labels, predictions)

    #     print(f'Accuracy: {accuracy:.4f}')
    #     print(f'Precision: {precision:.4f}')
    #     print(f'Recall: {recall:.4f}')
    #     print(f'ROC AUC: {roc_auc:.4f}')