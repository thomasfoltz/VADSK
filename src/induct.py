import argparse, json, torch
import numpy as np
import pandas as pd

from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoProcessor, MllamaForConditionalGeneration, BitsAndBytesConfig

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ped2', choices=['SHTech', 'avenue', 'ped2'])
    parser.add_argument('--root', type=str, help='Root directory for datasets')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--sample_size', type=int, help='Number of normal and anomaly samples to process')
    parser.add_argument('--keyword_limit', type=int, help='Maximum number of keywords for TF-IDF')
    parser.add_argument('--n_value', type=int, help='N-gram range for TF-IDF')
    parser.add_argument('--df_threshold', type=float, help='Document frequency threshold for TF-IDF')
    return parser.parse_args()

def tfidf(corpus, keyword_limit, n_value, df_threshold):
    if df_threshold>=0.5 or df_threshold<0:
        raise ValueError('Document frequency threshold must be between 0 and 0.5')
    
    vectorizer = TfidfVectorizer(
        decode_error = 'ignore',
        stop_words = 'english',
        ngram_range = (1, n_value),
        max_features = keyword_limit,
        min_df = df_threshold,
        max_df = 1 - df_threshold
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    keywords = vectorizer.get_feature_names_out()
    return tfidf_matrix, keywords

def tfidf_normalized_diff(tfidf_matrix):
    tfidf_diff = (tfidf_matrix[1] - tfidf_matrix[0]).toarray().flatten()
    tfidf_min, tf_idf_max = np.min(tfidf_diff), np.max(tfidf_diff)
    return np.round((tfidf_diff - tfidf_min) / (tf_idf_max - tfidf_min), 4)

class Induction:
    def __init__(self, dataset, root, sample_size, seed):
        self.dataset = dataset
        self.root = root
        self.sample_size = sample_size
        self.seed = seed

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

    def init_vlm_prompt(self):
        message = [{"role": "system", "content": "You are a surveillance monitor for urban safety"},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the activities and objects present in this scene."}]}]
        self.vlm_prompt = self.processor.apply_chat_template(message)

    def init_frame_paths(self):
        np.random.seed(self.seed)
        def select_frames(frame_paths):
            return np.random.choice(frame_paths, self.sample_size, replace=False).tolist()
        
        df = pd.read_csv(f'{self.dataset}/labels.csv')
        train_df = df[df['frame_path'].str.contains('train')]
        test_df = df[df['frame_path'].str.contains('test')]

        normal_frame_paths = select_frames(train_df['frame_path'].values)
        anomaly_frame_paths = select_frames(test_df.loc[test_df['label'] == 1, 'frame_path'].values)

        return normal_frame_paths, anomaly_frame_paths

    def generate_frame_descriptions(self, frame_paths):
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
            decoded_output = self.processor.decode(output[0])
            content = decoded_output.split('<|end_header_id|>')[3].strip('<|eot_id|>')
            return ' '.join(content.replace('\n', ' ').split()).lower()

        frame_descriptions = ''
        for frame_path in frame_paths:
            print('Generating frame description:', frame_path)
            input = setup_input(frame_path)
            frame_description = generate_output(input)
            frame_descriptions += f'{frame_description} '

        return frame_descriptions
    
    def generate_corpus(self, normal_frame_paths, anomaly_frame_paths):
        normal_frame_descriptions = self.generate_frame_descriptions(normal_frame_paths)
        anomaly_frame_descriptions = self.generate_frame_descriptions(anomaly_frame_paths)
        return (normal_frame_descriptions, anomaly_frame_descriptions)
        
if __name__ == "__main__":
    args = parse_arguments()

    with open(f"{args.dataset}/config.json", 'r') as f:
        config = json.load(f)
        for key, value in config['induction'].items():
            setattr(args, key, value)

    inductor = Induction(
        args.dataset, 
        args.root, 
        args.sample_size,
        args.seed
    )

    inductor.init_vlm()
    inductor.init_vlm_prompt()

    normal_frame_paths, anomaly_frame_paths = inductor.init_frame_paths()
    corpus = inductor.generate_corpus(normal_frame_paths, anomaly_frame_paths)

    tfidf_matrix, keywords = tfidf(corpus, args.keyword_limit, args.n_value, args.df_threshold)
    weights = tfidf_normalized_diff(tfidf_matrix)

    valid_keywords, valid_weights = zip(*[(keyword, weight) for keyword, weight in zip(keywords, weights) if keyword.isalpha()])

    keyword_data = {
        "frame_paths": normal_frame_paths + anomaly_frame_paths,
        "keywords": {valid_keywords[i]: valid_weights[i] for i in np.argsort(valid_weights)}
    }

    with open(f"{args.dataset}/keywords.json", 'w') as f:
        json.dump(keyword_data, f, indent=4)