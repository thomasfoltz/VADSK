import argparse, json, os, time, torch
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import pandas as pd

from PIL import Image
from model import VADSK
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from transformers import AutoProcessor, MllamaForConditionalGeneration, BitsAndBytesConfig
from transformers import AutoModel, AutoTokenizer

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ped2', choices=['SHTech', 'avenue', 'ped2'])
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--train', action='store_true', help='Flag to indicate training mode')
    parser.add_argument('--test', action='store_true', help='Flag to indicate testing mode')
    parser.add_argument('--interpret', action='store_true', help='Flag to generate feature input heatmap')
    parser.add_argument('--live', action='store_true', help='Predict on live frame descriptions')
    parser.add_argument('--root', type=str, help='Root directory for datasets')
    parser.add_argument('--batch_size', type=int, help='Batch size for training and validation')
    parser.add_argument('--epochs', type=int, help='Number of epochs for training')
    parser.add_argument('--k_folds', type=int, help='Number of folds for K-Fold cross-validation')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for optimizer')
    parser.add_argument('--decay', type=float, help='Weight decay for optimizer')
    parser.add_argument('--early_stopping', type=int, help='Early stopping patience')
    parser.add_argument('--pred_threshold', type=float, help='Threshold for classification')
    return parser.parse_args()

def plot_feature_heatmap(features, feature_dim, keywords, dataset, filename='feature_input'):
    plt.figure(figsize=(6, 10))
    plt.imshow(features.T.cpu().numpy(), cmap='viridis', aspect='auto')
    plt.title(f'Feature Input Heatmap')
    plt.xticks([])
    plt.yticks(ticks=range(feature_dim), labels=keywords, fontsize=8)
    plt.colorbar(label='Anomaly Weight')
    plt.tight_layout()
    plt.savefig(f'{dataset}/{filename}.png')

class Deduction:
    """
    This class is designed to facilitate the process of generating frame descriptions from video frames, 
    encoding those frame descriptions into keyword weights, and setting up anomaly classification.

    Attributes:
        device (torch.device): The device to run the computations on (CPU or GPU).
        dataset (str): The path to the dataset.
        root (str): The root directory for the dataset.
        model_type (str): The type of foundational model used for vision-language processing.
        learning_rate (float): The learning rate for the classification optimizer.
        decay (float): The weight decay for the classification optimizer.
        keywords (dict): A dictionary of keywords and their associated weights.
        used_frame_paths (list): A list of paths to the frames used previously in the induction set.
        feature_dim (int): The dimension of the feature vectors.

    Methods:
        init_foundational_model(): Initializes the foundational model for vision-language processing.
        generate_frame_descriptions(): Generates descriptions for the given frames.
        calculate_cls_weight(): Calculates the loss weighting for adjusted for class imbalance.
        init_classifier(): Initializes the classifier for video anomaly detection.
        frame_descriptions_to_features(): Encodes the frame descriptions into weighted feature vectors.
    """
    def __init__(self, dataset, root, model_type, learning_rate, decay, keywords, used_frame_paths, feature_dim):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.root = root
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.decay = decay
        self.keywords = keywords
        self.used_frame_paths = used_frame_paths
        self.feature_dim = feature_dim

    def init_foundational_model(self):
        if args.model_type == "meta-llama/Llama-3.2-11B-Vision-Instruct":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.foundational_model = MllamaForConditionalGeneration.from_pretrained(
                'meta-llama/Llama-3.2-11B-Vision-Instruct',
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                quantization_config=bnb_config,
            )

            self.processor = AutoProcessor.from_pretrained('meta-llama/Llama-3.2-11B-Vision-Instruct')
            message = [{"role": "system", "content": "You are a surveillance monitor for urban safety"},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the activities and objects present in this scene."}]}]
            self.foundational_model_prompt = self.processor.apply_chat_template(message, add_generation_prompt=True)

        elif args.model_type == "openbmb/MiniCPM-V-2_6-int4":
            self.foundational_model = AutoModel.from_pretrained(
                'openbmb/MiniCPM-V-2_6-int4',
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).eval()

            self.tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6-int4', trust_remote_code=True)
            self.foundational_model_prompt = 'You are a surveillance monitor for urban safety. Describe the activities and objects present in this scene.'
        
        else:
            raise ValueError('Invalid model type')

    def generate_frame_descriptions(self, frame_paths, mode='train'):
        def setup_input(frame_path):
            image = Image.open(f"{self.root}{frame_path}").convert('RGB')
            if args.model_type == "meta-llama/Llama-3.2-11B-Vision-Instruct":
                input = self.processor(
                    image,
                    self.foundational_model_prompt,
                    add_special_tokens=False,
                    return_tensors="pt"
                ).to('cuda')

            elif args.model_type == "openbmb/MiniCPM-V-2_6-int4":
                input = [{'role': 'user', 'content': [image, self.foundational_model_prompt]}]

            else:
                raise ValueError('Invalid model type')
            
            return input
        
        def generate_output(input):
            if args.model_type == "meta-llama/Llama-3.2-11B-Vision-Instruct":
                with torch.no_grad():
                    output = self.foundational_model.generate(**input, max_new_tokens=128)
                decoded_output = self.processor.decode(output[0])
                content = decoded_output.split('<|end_header_id|>')[3].strip('<|eot_id|>')

            elif args.model_type == "openbmb/MiniCPM-V-2_6-int4":
                content = self.foundational_model.chat(
                    image=None,
                    msgs=input,
                    tokenizer=self.tokenizer
                )
                
            else:
                raise ValueError('Invalid model type')
            
            return ' '.join(content.replace('\n', ' ').split()).lower()
        
        if len(frame_paths) == 1:
            input = setup_input(frame_paths[0])
            frame_description = generate_output(input)
            return frame_description
        else:
            for frame_path in frame_paths:
                print('Generating frame description:', frame_path)
                input = setup_input(frame_path)
                frame_description = generate_output(input)
                del input
                torch.cuda.empty_cache()
                with open(f'{self.dataset}/{mode}_descriptions.txt', 'a') as f:
                    f.write(f'{frame_description}\n')
    
    def calculate_cls_weight(self, labels):
        anomaly_count = sum(1 for label in labels if label == 1)
        anomaly_prop = anomaly_count / len(labels)
        self.cls_weight = torch.tensor([(1-anomaly_prop)/anomaly_prop], dtype=torch.float32, device=self.device)

    def init_classifier(self):
        self.VADSK = VADSK(feature_dim=self.feature_dim).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.cls_weight)
        self.optimizer = optim.AdamW(self.VADSK.parameters(), lr=self.learning_rate, weight_decay=self.decay)

    def frame_descriptions_to_features(self, frame_descriptions):
        feature_input = torch.zeros((len(frame_descriptions), self.feature_dim), dtype=torch.float32)
        for j, keyword in enumerate(list(self.keywords.keys())):
            weight = self.keywords[keyword]
            matches = [weight if keyword in desc else 0 for desc in frame_descriptions]
            feature_input[:, j] = torch.tensor(matches, dtype=torch.float32)
        return feature_input

if __name__ == "__main__":
    args = parse_arguments()

    with open(f"{args.dataset}/config.json", 'r') as f:
        config = json.load(f)

    args.root = config['root']
    args.model_type = config['model_type']
    for key, value in config['deduction'].items():
        setattr(args, key, value)

    with open(f'{args.dataset}/keywords.json', 'r') as f:
        keyword_data = json.load(f)

    keywords = keyword_data['keywords']
    used_frame_paths = keyword_data['frame_paths']
    feature_dim = len(keywords)

    deductor = Deduction(
        args.dataset, 
        args.root,
        args.model_type, 
        args.learning_rate, 
        args.decay,
        keywords, 
        used_frame_paths, 
        feature_dim
    )

    df = pd.read_csv(f'{args.dataset}/labels.csv', dtype={'frame_path': str, 'label': int})
    df = df[~df['frame_path'].isin(deductor.used_frame_paths) & df['frame_path'].str.contains('test')]
    frame_paths = df.iloc[:, 0].tolist()
    labels = df.iloc[:, 1].astype(int).tolist()

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        frame_paths, labels, test_size=0.2, random_state=args.seed
    )

    if args.train:
        description_path = f'{args.dataset}/train_descriptions.txt'
        if not os.path.exists(description_path):
            deductor.init_foundational_model()
            deductor.generate_frame_descriptions(train_paths, mode='train')
        
        with open(description_path, 'r') as f:
            train_descriptions = [line.strip() for line in f.readlines()]

        lowest_val_loss = float('inf')
        train_losses, val_losses = [], []

        deductor.calculate_cls_weight(train_labels)
        kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

        for fold, (train_index, val_index) in enumerate(kf.split(list(range(len(train_paths))))):
            fold_lowest_val_loss = float('inf')
            print(f'Fold {fold + 1}/{args.k_folds}')

            train_descriptions_fold = [train_descriptions[i] for i in train_index]
            train_labels_fold = [train_labels[i] for i in train_index]
            
            val_descriptions_fold = [train_descriptions[i] for i in val_index]
            val_labels_fold = [train_labels[i] for i in val_index]

            train_batches = len(train_descriptions_fold) // args.batch_size
            val_batches = len(val_descriptions_fold) // args.batch_size

            deductor.init_classifier()
            
            for epoch in range(args.epochs):
                train_loss, val_loss = 0.0, 0.0

                deductor.VADSK.train()
                for i in range(train_batches):
                    deductor.optimizer.zero_grad()

                    batch_descriptions = train_descriptions_fold[i * args.batch_size:(i + 1) * args.batch_size]
                    batch_labels = train_labels_fold[i * args.batch_size:(i + 1) * args.batch_size]

                    labels_tensor = torch.tensor(batch_labels, dtype=torch.float32).unsqueeze(0).to(deductor.device)
                    feature_input = deductor.frame_descriptions_to_features(batch_descriptions).to(deductor.device)
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

                    labels_tensor = torch.tensor(batch_labels, dtype=torch.float32).unsqueeze(0).to(deductor.device)
                    feature_input = deductor.frame_descriptions_to_features(batch_descriptions).to(deductor.device)

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

                if early_stop_count >= args.early_stopping:
                    print('Early stopping triggered')
                    break
        
        average_train_loss = sum(train_losses) / len(train_losses)
        average_val_loss = sum(val_losses) / len(val_losses)

        print(f'Average Train Loss: {average_train_loss:.4f}')
        print(f'Average Val Loss: {average_val_loss:.4f}')

        torch.save(best_model_state, f'{args.dataset}/vadsk.pth')

    if args.test:
        deductor.VADSK = VADSK(feature_dim=feature_dim).to(deductor.device)
        deductor.VADSK.load_state_dict(torch.load(f'{args.dataset}/vadsk.pth', weights_only=True))
        deductor.VADSK.eval()

        if args.live:
            deductor.init_foundational_model()

            total_time = 0
            num_iterations = len(test_paths)
            predictions = []

            for test_path, test_label in zip(test_paths, test_labels):
                start_time = time.time()
                
                test_description = deductor.generate_frame_descriptions([test_path], mode='test')
                feature_input = deductor.frame_descriptions_to_features([test_description])
                
                if args.interpret:
                    filename = ''.join(test_path.replace('/', '_').split('.')[0])
                    plot_feature_heatmap(
                        feature_input, 
                        feature_dim, 
                        deductor.keywords, 
                        args.dataset, 
                        filename
                    )

                with torch.no_grad():
                    output = deductor.VADSK(feature_input.to(deductor.device))
                
                prediction = 1 if output >= args.pred_threshold else 0
                predictions.append(prediction)
                print(f'Predicted Probability: {round(output, 4)}, Ground Truth Label: {test_label}')
                
                end_time = time.time()
                iteration_time = end_time - start_time
                print(f'Iteration time: {iteration_time:.4f} seconds')
                total_time += iteration_time

            average_time = total_time / num_iterations
            print(f'Average iteration time: {average_time:.4f} seconds')
                
        else:
            description_path = f'{args.dataset}/test_descriptions.txt'
            if not os.path.exists(description_path):
                deductor.init_foundational_model()
                deductor.generate_frame_descriptions(test_paths, mode='test')
            
            with open(description_path, 'r') as f:
                test_descriptions = [line.strip() for line in f.readlines()]

            feature_input = deductor.frame_descriptions_to_features(test_descriptions)
            if args.interpret: 
                plot_feature_heatmap(
                    feature_input, 
                    feature_dim, 
                    deductor.keywords, 
                    args.dataset
                )

            with torch.no_grad():
                outputs = deductor.VADSK(feature_input.to(deductor.device))

            predictions = (outputs >= args.pred_threshold).squeeze(0).tolist()

        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions)
        recall = recall_score(test_labels, predictions)
        roc_auc = roc_auc_score(test_labels, predictions)

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'ROC AUC: {roc_auc:.4f}')