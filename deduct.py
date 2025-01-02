import argparse, csv, json, os, seaborn, torch
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import pandas as pd

from PIL import Image
from model import VADSK
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from transformers import AutoProcessor, MllamaForConditionalGeneration, BitsAndBytesConfig

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ped2', choices=['SHTech', 'avenue', 'ped2'])
    parser.add_argument('--root', type=str, help='Root directory for datasets')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training and validation')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--kfolds', type=int, default=5, help='Number of folds for K-Fold cross-validation')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--decay', type=float, default=0.001, help='Weight decay for optimizer')
    parser.add_argument('--early_stop', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for classification')
    parser.add_argument('--train', action='store_true', help='Flag to indicate training mode')
    parser.add_argument('--test', action='store_true', help='Flag to indicate testing mode')
    parser.add_argument('--interpret', action='store_true', help='Flag to generate feature input heatmap')
    return parser.parse_args()

class Deduction:
    def __init__(self, args):
        self.dataset = args.dataset
        self.root = args.root
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = args.lr
        self.decay = args.decay
        self.cls_weight = None

    def init_vlm(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.vlm_model = MllamaForConditionalGeneration.from_pretrained(
            'meta-llama/Llama-3.2-11B-Vision-Instruct',
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            quantization_config=bnb_config,
        )
        self.processor = AutoProcessor.from_pretrained('meta-llama/Llama-3.2-11B-Vision-Instruct')

    def init_frame_paths(self):
        self.labels = pd.read_csv(f'benchmarks/{self.dataset}/test.csv').iloc[:, 1].tolist()
        self.frame_paths = pd.read_csv(f'benchmarks/{self.dataset}/test.csv').iloc[:, 0].tolist()

    def init_vlm_prompt(self):
        message = [{"role": "system", "content": "You are a surveillance monitor for urban safety"},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the activities and objects present in this scene."}]}]
        self.vlm_prompt = self.processor.apply_chat_template(message, add_generation_prompt=True)

    def generate_frame_descriptions(self):
        def setup_input(frame_path):
            image = Image.open(f"{self.root}{frame_path}").convert('RGB')
            input = self.processor(
                image,
                self.vlm_prompt,
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

            with open(f"benchmarks/{self.dataset}/descriptions.csv", 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([frame_path, label, frame_description])

    def init_keywords(self):        
        with open(f'benchmarks/{self.dataset}/keywords.json', 'r') as f:
            keyword_data = json.load(f)

        self.keywords = keyword_data['keywords']
        self.used_frame_paths = keyword_data['normal_frame_paths'] + keyword_data['anomaly_frame_paths']
        self.feature_dim = len(self.keywords)
    
    def calculate_cls_weight(self, labels):
        anomaly_count = sum(1 for label in labels if label == 1)
        anomaly_prop = anomaly_count / len(labels)
        self.cls_weight = torch.tensor([(1-anomaly_prop)/anomaly_prop], dtype=torch.float32, device=self.device)

    def init_classifier(self):
        self.VADSK = VADSK(feature_dim=self.feature_dim).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.cls_weight)
        self.optimizer = optim.AdamW(self.VADSK.parameters(), lr=self.lr, weight_decay=self.decay)

    def frame_descriptions_to_features(self, frame_descriptions):
        feature_input = torch.zeros((len(frame_descriptions), self.feature_dim), dtype=torch.float32)
        for j, keyword in enumerate(list(self.keywords.keys())):
            weight = self.keywords[keyword]
            matches = [weight if keyword in desc else 0 for desc in frame_descriptions]
            feature_input[:, j] = torch.tensor(matches, dtype=torch.float32)
        return feature_input

if __name__ == "__main__":
    args = parse_arguments()
    deductor = Deduction(args)

    descriptions_path = f'benchmarks/{args.dataset}/descriptions.csv'
    vadsk_path = f'benchmarks/{args.dataset}/vadsk.pth'

    if not os.path.exists(descriptions_path):
        deductor.init_vlm()
        deductor.init_vlm_prompt()
        deductor.init_frame_paths()
        deductor.generate_frame_descriptions()

    deductor.init_keywords()
    df = pd.read_csv(descriptions_path, header=None, names=['frame_path', 'label', 'description'])
    df = df[~df['frame_path'].isin(deductor.used_frame_paths)]

    train_paths, test_paths, train_descriptions, test_descriptions, train_labels, test_labels = train_test_split(
        df['frame_path'].tolist(), df['description'].tolist(), df['label'].tolist(), 
        test_size=0.2, random_state=42
    )

    if args.train:
        lowest_val_loss = float('inf')
        train_losses, val_losses = [], []

        deductor.calculate_cls_weight(train_labels)
        kf = KFold(n_splits=args.kfolds, shuffle=True, random_state=42)

        for fold, (train_index, val_index) in enumerate(kf.split(list(range(len(train_paths))))):
            fold_lowest_val_loss = float('inf')
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

                deductor.VADSK.train()
                for i in range(train_batches):
                    deductor.optimizer.zero_grad()

                    batch_descriptions = train_descriptions_fold[i * args.batch_size:(i + 1) * args.batch_size]
                    batch_labels = train_labels_fold[i * args.batch_size:(i + 1) * args.batch_size]
                    batch_images = train_paths_fold[i * args.batch_size:(i + 1) * args.batch_size]

                    labels_tensor = torch.tensor(batch_labels, dtype=torch.float32).unsqueeze(0).to(deductor.device)
                    feature_input = deductor.frame_descriptions_to_features(batch_descriptions)
                    outputs = deductor.VADSK(feature_input.to(deductor.device))

                    loss = deductor.criterion(outputs, labels_tensor)
                    train_loss += loss.item()
                    
                    loss.backward()
                    deductor.optimizer.step()

                train_loss /= train_batches
                train_losses.append(train_loss)

                deductor.VADSK.eval()
                for i in range(val_batches):
                    batch_descriptions = val_descriptions_fold[i * args.batch_size:(i + 1) * args.batch_size]
                    batch_labels = val_labels_fold[i * args.batch_size:(i + 1) * args.batch_size]
                    batch_images = val_paths_fold[i * args.batch_size:(i + 1) * args.batch_size]

                    labels_tensor = torch.tensor(batch_labels, dtype=torch.float32).unsqueeze(0).to(deductor.device)
                    feature_input = deductor.frame_descriptions_to_features(batch_descriptions)

                    with torch.no_grad():
                        outputs = deductor.VADSK(feature_input.to(deductor.device))

                    loss = deductor.criterion(outputs, labels_tensor)
                    val_loss += loss.item()

                val_loss /= val_batches
                val_losses.append(val_loss)

                print(f'Epoch {epoch + 1}/{args.epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

                if val_loss < fold_lowest_val_loss:
                    fold_lowest_val_loss = val_loss
                    early_stop_count = 0
                    if fold_lowest_val_loss < lowest_val_loss:
                        best_model_state = deductor.VADSK.state_dict()
                else:
                    early_stop_count += 1

                if early_stop_count >= args.early_stop:
                    print('Early stopping triggered')
                    break
        
        average_train_loss = sum(train_losses) / len(train_losses)
        average_val_loss = sum(val_losses) / len(val_losses)

        print(f'Average Train Loss: {average_train_loss:.4f}')
        print(f'Average Val Loss: {average_val_loss:.4f}')

        torch.save(best_model_state, vadsk_path)

    if args.test:
        feature_input = deductor.frame_descriptions_to_features(test_descriptions)
        if args.interpret:
            plt.figure(figsize=(15, 15))
            seaborn.heatmap(feature_input.T.cpu().numpy(), cmap='viridis', cbar=True)
            plt.title('Feature Input Heatmap')
            plt.xlabel('Frame')
            plt.ylabel('Feature')
            plt.yticks(ticks=range(len(deductor.keywords)), labels=deductor.keywords, rotation=0)
            plt.savefig(f'benchmarks/{args.dataset}/feature_input.png')

        deductor.init_classifier()
        deductor.VADSK.load_state_dict(torch.load(vadsk_path, weights_only=True))
        deductor.VADSK.eval()

        with torch.no_grad():
            outputs = deductor.VADSK(feature_input.to(deductor.device))
            
        probs = torch.sigmoid(outputs)
        predictions = (probs >= args.threshold).squeeze(0).tolist()

        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions)
        recall = recall_score(test_labels, predictions)
        roc_auc = roc_auc_score(test_labels, predictions)

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'ROC AUC: {roc_auc:.4f}')