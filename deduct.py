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
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from pishield.shield_layer import ShieldLayer
from models import VADSR_CNN as VADSR

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ped2', choices=['SHTech', 'avenue', 'ped2', 'UBNormal'], help='Dataset name')
    parser.add_argument('--root', type=str, default='/data/tjf5667/datasets/', help='Root directory for datasets')
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--num_epochs', type=int, default=5)
    return parser.parse_args()

class Deduction:
    def __init__(self, vlm_model, processor, args):
        self.vlm_model = vlm_model
        self.processor = processor
        self.args = args
        self.labels = None
        self.frame_paths = None
        self.vlm_message = None
        self.keywords = None
        self.grouped_frames = None
        self.rules = {}
        self.rule_num = 0
        self.shield_layer = None
        self.VADSR = None
        self.criterion = None
        self.optimizer = None

    def set_frame_paths(self):
        self.labels = pd.read_csv(f'benchmarks/{self.args.data}/test.csv').iloc[:, 1].tolist()
        self.frame_paths = pd.read_csv(f'benchmarks/{self.args.data}/test.csv').iloc[:, 0].tolist()

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

            with open(f"benchmarks/{self.args.data}/test_descriptions.csv", 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([frame_path, label, frame_description])

    def set_keywords(self):
        with open(f'benchmarks/{self.args.data}/rules.json', 'r') as f:
            rules = json.load(f)
        self.keywords = [None] + rules['normal_activities'] + [None] + rules['abnormal_activities'] + [None] + rules['normal_objects'] + [None] + rules['abnormal_objects']

    def group_frames(self):
        with open(f'benchmarks/{self.args.data}/test_descriptions.csv', 'r') as f:
            df = pd.read_csv(f, header=None, names=['frame_path', 'label', 'description'])
            # df['video_id'] = df['frame_path'].apply(lambda x: '_'.join(x.split('/')[-1].split('_')[:2])) # TODO: if extracting from train_descriptions
            df['video_id'] = df['frame_path'].apply(lambda x: x.split('/')[-2])
            self.grouped_frames = df.groupby('video_id')

    def frame_descriptions_to_features(self, frame_descriptions):
        num_keywords = len(self.keywords)
        num_frames = len(frame_descriptions)
        feature_input = torch.zeros((num_frames, num_keywords), dtype=torch.float32)

        for frame_idx, description in enumerate(frame_descriptions):
            for keyword_idx, keyword in enumerate(self.keywords):
                if keyword is not None and re.search(r'\b' + keyword + r'\b', description):
                    feature_input[frame_idx, keyword_idx] = 1.0

        return feature_input
    
    def construct_rule_statements(self):
        with open(f'benchmarks/{self.args.data}/rules.json', 'r') as file:
            rules_dict = json.load(file)

        for key, values in rules_dict.items():
            self.rules[key] = [f'y_{self.rule_num}']
            self.rule_num += 1
            self.rules[key].extend(f'y_{self.rule_num + i}' for i in range(len(values)))
            self.rule_num += len(values)

        with open(f'benchmarks/{self.args.data}/propositional_statements.txt', 'w') as file:
            for values in self.rules.values():
                file.write(' or '.join(values) + '\n')

    def init_shield_layer(self, feature_num):
        self.shield_layer = ShieldLayer(
            feature_num,
            f'benchmarks/{self.args.data}/propositional_statements.txt', 
            ordering=list(range(self.rule_num)), 
        )

    def show_frame_feature_matches(self, feature_input, frame_idx):
        match_idx = torch.where(feature_input[frame_idx, :] == 1.0)[0].tolist()
        print(f'Frame {frame_idx}: {[self.keywords[idx] for idx in match_idx]}')

if __name__ == "__main__":
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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

    deductor.set_keywords()
    deductor.group_frames()
    deductor.construct_rule_statements()
    feature_num = len(deductor.keywords)
    deductor.init_shield_layer(feature_num)
    deductor.VADSR = VADSR(k=feature_num).to(device)

    df = pd.read_csv(f'benchmarks/{args.data}/test_descriptions.csv', header=None, names=['frame_path', 'label', 'description'])
    descriptions = df['description'].tolist()
    labels = df['label'].tolist()

    train_descriptions, eval_descriptions, train_labels, eval_labels = train_test_split(
        descriptions, labels, test_size=0.3, random_state=42
    )
    val_descriptions, test_descriptions, val_labels, test_labels = train_test_split(
        eval_descriptions, eval_labels, test_size=0.5, random_state=42
    )
    num_train_batches = len(train_descriptions) // args.batch_size
    num_val_batches = len(val_descriptions) // args.batch_size
    num_test_batches = len(test_descriptions) // args.batch_size

    neg_class_total = sum(1 for label in train_labels if label == 0)
    neg_class_prop = neg_class_total / len(train_labels)
    
    # SHTech: neg class (0) - 0.5739, pos class (1) - 0.4261
    # ped2: neg class (0) - 0.1859, pos class (1) - 0.8141
    # avenue: neg class (0) - 0.747, pos class (1) - 0.253

    # TODO: find way to do this without hard-coding
    if args.data == 'SHTech':
        pos_weight = torch.tensor([2.0], dtype=torch.float32, device=device)
    elif args.data == 'avenue':
        pos_weight = torch.tensor([4.0], dtype=torch.float32, device=device)
    elif args.data == 'ped2':
        pos_weight = torch.tensor([0.4], dtype=torch.float32, device=device)

    deductor.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # TODO: need to explore different weightings for class imbalance
    deductor.optimizer = optim.AdamW(deductor.VADSR.parameters(), lr=0.001, weight_decay=1e-5)

    for epoch in range(args.num_epochs):
        train_loss, val_loss = 0.0, 0.0

        # TRAINING
        deductor.VADSR.train()
        for i in range(num_train_batches):
            batch_descriptions = train_descriptions[i * args.batch_size:(i + 1) * args.batch_size]
            batch_labels = train_labels[i * args.batch_size:(i + 1) * args.batch_size]
            labels_tensor = torch.tensor(batch_labels, dtype=torch.float32).unsqueeze(0).to(device)
            
            feature_input = deductor.frame_descriptions_to_features(batch_descriptions)
            shielded_feature_input = deductor.shield_layer(feature_input).to(device)

            deductor.optimizer.zero_grad()

            outputs = deductor.VADSR(shielded_feature_input)
            loss = deductor.criterion(outputs, labels_tensor)
            train_loss += loss.item()
            
            loss.backward()
            deductor.optimizer.step()

        train_loss /= num_train_batches

        # VALIDATION
        deductor.VADSR.eval()
        with torch.no_grad():
            for i in range(num_val_batches):
                batch_descriptions = val_descriptions[i * args.batch_size:(i + 1) * args.batch_size]
                batch_labels = val_labels[i * args.batch_size:(i + 1) * args.batch_size]
                labels_tensor = torch.tensor(batch_labels, dtype=torch.float32).unsqueeze(0).to(device)
                
                feature_input = deductor.frame_descriptions_to_features(batch_descriptions)
                shielded_feature_input = deductor.shield_layer(feature_input).to(device)
                
                outputs = deductor.VADSR(shielded_feature_input)
                loss = deductor.criterion(outputs, labels_tensor)
                val_loss += loss.item()

        val_loss /= num_val_batches

        print(f'Epoch [{epoch+1}/{args.num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
    
    torch.save(deductor.VADSR.state_dict(), f'benchmarks/{args.data}/vadsr_{args.data}.pth')
    # deductor.VADSR.load_state_dict(torch.load(f'benchmarks/{args.data}/vadsr_{args.data}.pth', weights_only=True))

    # TESTING
    deductor.VADSR.eval()
    with torch.no_grad():
        labels_tensor = torch.tensor(test_labels, dtype=torch.float32).to(device)
        
        feature_input = deductor.frame_descriptions_to_features(test_descriptions)
        shielded_feature_input = deductor.shield_layer(feature_input).to(device)
        
        outputs = deductor.VADSR(shielded_feature_input)

        predictions = torch.round(outputs).squeeze(0).tolist() # TODO: better thresholding
        accuracy = accuracy_score(labels_tensor.tolist(), predictions)
        precision = precision_score(labels_tensor.tolist(), predictions)
        recall = recall_score(labels_tensor.tolist(), predictions)
        roc_auc = roc_auc_score(labels_tensor.tolist(), predictions)

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'ROC AUC: {roc_auc:.4f}')
