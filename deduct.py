import argparse, csv, json, torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd

from PIL import Image
from model import VADSR
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from transformers import AutoProcessor, MllamaForConditionalGeneration, BitsAndBytesConfig

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ped2', choices=['SHTech', 'avenue', 'ped2', 'UBNormal'], help='Dataset name')
    parser.add_argument('--root', type=str, default='/data/tjf5667/datasets/', help='Root directory for datasets')
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--kfolds', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--decay', type=float, default=0.01)
    return parser.parse_args()

def calculate_cls_weight(labels):
    anomaly_count = sum(1 for label in labels if label == 1)
    anomaly_prop = anomaly_count / len(labels)
    cls_weight = torch.tensor([(1-anomaly_prop)/anomaly_prop], dtype=torch.float32, device=device)
    return cls_weight

class Deduction:
    def __init__(self, vlm_model, processor, args):
        self.vlm_model = vlm_model
        self.processor = processor
        self.args = args
        self.labels = None
        self.frame_paths = None
        self.vlm_message = None
        self.keywords = None
        self.feature_dim = None
        self.cls_weight = None
        self.rule_num = 0
        self.shield_layer = None
        self.VADSR = None
        self.criterion = None
        self.optimizer = None

    def init_frame_paths(self):
        self.labels = pd.read_csv(f'benchmarks/{self.args.data}/test.csv').iloc[:, 1].tolist()
        self.frame_paths = pd.read_csv(f'benchmarks/{self.args.data}/test.csv').iloc[:, 0].tolist()

    def setup_vlm_prompt(self):
        messages = [{"role": "system", "content": "You are a surveillance monitor for urban safety"},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the activities and objects present in this scene."}]}]
        self.vlm_message = self.processor.apply_chat_template(messages, add_generation_prompt=True)

    def generate_frame_descriptions(self):
        def setup_input(frame_path):
            image = Image.open(f"{self.args.root}{frame_path}").convert('RGB')
            input = self.processor(
                image,
                self.vlm_message,
                add_special_tokens=False,
                return_tensors="pt"
            ).to('cuda')
            return input
        
        def generate_output(input):
            with torch.no_grad():
                output = self.vlm_model.generate(**input, max_new_tokens=128)
            return self.processor.decode(output[0])

        for frame_path, label in zip(self.frame_paths, self.labels):
            print('Generating frame description:', frame_path)
            input = setup_input(frame_path)
            output = generate_output(input)

            content = output.split('assistant<|end_header_id|>')[1].strip('<|eot_id|>')
            frame_description = content.strip().lower().replace('\n', '')

            with open(f"benchmarks/{self.args.data}/descriptions.csv", 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([frame_path, label, frame_description])

    def init_keywords(self):        
        with open(f'benchmarks/{self.args.data}/keywords.json', 'r') as f:
            keyword_data = json.load(f)

        self.keywords = keyword_data['keywords']
        self.used_frame_paths = keyword_data['normal_frame_paths'] + keyword_data['abnormal_frame_paths']
        self.feature_dim = len(self.keywords)

    def init_classifier(self):
        self.VADSR = VADSR(feature_dim=self.feature_dim).to(device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.cls_weight)
        self.optimizer = optim.AdamW(self.VADSR.parameters(), lr=self.args.lr, weight_decay=self.args.decay)

    def frame_descriptions_to_features(self, frame_descriptions):
        feature_input = torch.zeros((len(frame_descriptions), self.feature_dim), dtype=torch.float32)
        for j, keyword in enumerate(list(self.keywords.keys())):
            weight = self.keywords[keyword]
            matches = [weight if keyword in desc else 0 for desc in frame_descriptions]
            feature_input[:, j] = torch.tensor(matches, dtype=torch.float32)
        return feature_input

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
    )

    processor = AutoProcessor.from_pretrained('meta-llama/Llama-3.2-11B-Vision-Instruct')

    deductor = Deduction(vlm_model, processor, args)

    deductor.init_frame_paths()
    deductor.setup_vlm_prompt()
    deductor.generate_frame_descriptions()
    deductor.init_keywords()

    df = pd.read_csv(f'benchmarks/{args.data}/descriptions.csv', header=None, names=['frame_path', 'label', 'description'])
    df = df[~df['frame_path'].isin(deductor.used_frame_paths)]

    train_paths, test_paths, train_descriptions, test_descriptions, train_labels, test_labels = train_test_split(
        df['frame_path'].tolist(), df['description'].tolist(), df['label'].tolist(), 
        test_size=0.2, random_state=42
    )

    train_losses, val_losses = [], []
    best_val_loss, best_model_state = float('inf'), None
    deductor.cls_weight = calculate_cls_weight(train_labels)
    kf = KFold(n_splits=args.kfolds, shuffle=True, random_state=42)

    for fold, (train_index, val_index) in enumerate(kf.split(list(range(len(train_paths))))):
        print(f'Fold {fold + 1}/{args.kfolds}')

        train_descriptions_fold = [train_descriptions[i] for i in train_index]
        train_labels_fold = [train_labels[i] for i in train_index]
        train_paths_fold = [train_paths[i] for i in train_index]
        
        val_descriptions_fold = [train_descriptions[i] for i in val_index]
        val_labels_fold = [train_labels[i] for i in val_index]
        val_paths_fold = [train_paths[i] for i in val_index]

        train_batches = len(train_paths_fold) // args.batch_size
        val_batches = len(val_paths_fold) // args.batch_size

        deductor.init_classifier()

        for epoch in range(args.epochs):
            train_loss, val_loss = 0.0, 0.0

            deductor.VADSR.train()
            for i in range(train_batches):
                deductor.optimizer.zero_grad()

                batch_descriptions = train_descriptions_fold[i * args.batch_size:(i + 1) * args.batch_size]
                batch_labels = train_labels_fold[i * args.batch_size:(i + 1) * args.batch_size]
                batch_images = train_paths_fold[i * args.batch_size:(i + 1) * args.batch_size]

                labels_tensor = torch.tensor(batch_labels, dtype=torch.float32).unsqueeze(0).to(device)
                feature_input = deductor.frame_descriptions_to_features(batch_descriptions)
                outputs = deductor.VADSR(feature_input.to(device))

                loss = deductor.criterion(outputs, labels_tensor)
                train_loss += loss.item()
                
                loss.backward()
                deductor.optimizer.step()

            train_loss /= train_batches
            train_losses.append(train_loss)

            deductor.VADSR.eval()
            with torch.no_grad():
                for i in range(val_batches):
                    batch_descriptions = val_descriptions_fold[i * args.batch_size:(i + 1) * args.batch_size]
                    batch_labels = val_labels_fold[i * args.batch_size:(i + 1) * args.batch_size]
                    batch_images = val_paths_fold[i * args.batch_size:(i + 1) * args.batch_size]

                    labels_tensor = torch.tensor(batch_labels, dtype=torch.float32).unsqueeze(0).to(device)
                    feature_input = deductor.frame_descriptions_to_features(batch_descriptions)
                    outputs = deductor.VADSR(feature_input.to(device))

                    loss = deductor.criterion(outputs, labels_tensor)
                    val_loss += loss.item()

            val_loss /= val_batches
            val_losses.append(val_loss)

            print(f'Epoch {epoch + 1}/{args.epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = deductor.VADSR.state_dict()
    
    average_train_loss = sum(train_losses) / len(train_losses)
    average_val_loss = sum(val_losses) / len(val_losses)

    print(f'Average Train Loss: {average_train_loss:.4f}')
    print(f'Average Val Loss: {average_val_loss:.4f}')

    torch.save(best_model_state, f'benchmarks/{args.data}/vadsr.pth')
    deductor.VADSR.load_state_dict(torch.load(f'benchmarks/{args.data}/vadsr.pth', weights_only=True))

    deductor.VADSR.eval()
    with torch.no_grad():
        feature_input = deductor.frame_descriptions_to_features(test_descriptions)
        outputs = deductor.VADSR(feature_input.to(device))
        probs = torch.sigmoid(outputs)
        predictions = torch.round(probs).squeeze(0).tolist() # TODO: experiment with thresholding

        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions)
        recall = recall_score(test_labels, predictions)
        roc_auc = roc_auc_score(test_labels, predictions)

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'ROC AUC: {roc_auc:.4f}')