import argparse, json, torch
import numpy as np
import pandas as pd

from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoProcessor, MllamaForConditionalGeneration, BitsAndBytesConfig

# import nltk
# from nltk.corpus import wordnet as wn

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ped2', choices=['SHTech', 'avenue', 'ped2'])
    parser.add_argument('--root', type=str, help='Root directory for datasets')
    parser.add_argument('--sample_size', type=int, default=20, help='Number of normal and anomaly samples to process')
    parser.add_argument('--keyword_limit', type=int, default=100, help='Maximum number of keywords for TF-IDF')
    parser.add_argument('--n_value', type=int, default=1, help='N-gram range for TF-IDF')
    parser.add_argument('--df_threshold', type=float, default=0.05, help='Document frequency threshold for TF-IDF')
    return parser.parse_args()

def tfidf(corpus, keyword_limit, n_value, df_threshold):
    vectorizer = TfidfVectorizer(
        decode_error = 'ignore',
        stop_words = 'english',
        ngram_range = (1, n_value),
        max_features = keyword_limit,
        min_df = df_threshold,
        max_df = 1 - df_threshold
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix, vectorizer

def tfidf_normalized_diff(tfidf_matrix):
    tfidf_diff = (tfidf_matrix[1] - tfidf_matrix[0]).toarray().flatten()
    tfidf_min, tf_idf_max = np.min(tfidf_diff), np.max(tfidf_diff)
    return (tfidf_diff - tfidf_min) / (tf_idf_max - tfidf_min)

# def get_synonyms(word):
#     synonyms = set()
#     synsets = wn.synsets(word)
#     if synsets == []:
#         return None
#     original_synset = synsets[0]
    
#     for syn in wn.synsets(word):
#         for lemma in syn.lemmas():
#             synonym_synset = lemma.synset()
#             similarity = original_synset.path_similarity(synonym_synset)
#             lemma_name = lemma.name().replace('_', ' ')
#             if similarity == 1.0 and lemma_name != word:
#                 synonyms.add(lemma_name)

#     return list(synonyms)

# def include_synonyms(keywords, weights):
#     nltk.download('wordnet')
#     for keyword, weight in zip(keywords, weights):
#         synonyms = get_synonyms(keyword)
#         if synonyms is None:
#             continue
#         for synonym in synonyms:
#             if synonym not in keywords:
#                 keywords = np.append(keywords, synonym)
#                 weights = np.append(weights, weight)
#             else:
#                 index = np.where(keywords == synonym)[0][0]
#                 weights[index] = weight
#     return keywords, weights

class Induction:
    def __init__(self, args):
        self.dataset = args.dataset
        self.root = args.root
        self.sample_size = args.sample_size
        self.normal_frame_paths = None
        self.anomaly_frame_paths = None
        self.vlm_prompt = None

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
        def load_frame_paths(file_path, label):
            np.random.seed(42)
            df = pd.read_csv(file_path)
            frame_paths = df.loc[df['label'] == label, 'image_path'].values
            selected_frame_paths = np.random.choice(frame_paths, self.sample_size, replace=False)
            return selected_frame_paths.tolist()

        self.normal_frame_paths = load_frame_paths(f'benchmarks/{self.dataset}/train.csv', 0)
        self.anomaly_frame_paths = load_frame_paths(f'benchmarks/{self.dataset}/test.csv', 1)

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
            return self.processor.decode(output[0])

        frame_descriptions = ''
        for frame_path in frame_paths:
            print('Generating frame description:', frame_path)
            input = setup_input(frame_path)
            output = generate_output(input)

            content = output.split('assistant<|end_header_id|>')[1].strip('<|eot_id|>')
            frame_description = content.strip().lower().replace('\n', '')
            frame_descriptions += f'{frame_description} '

        return frame_descriptions
    
    def generate_corpus(self):
        normal_frame_descriptions = self.generate_frame_descriptions(self.normal_frame_paths)
        anomaly_frame_descriptions = self.generate_frame_descriptions(self.anomaly_frame_paths)
        return [normal_frame_descriptions, anomaly_frame_descriptions]
        
if __name__ == "__main__":
    args = parse_arguments()
    inductor = Induction(args)

    inductor.init_vlm()
    inductor.init_vlm_prompt()
    inductor.init_frame_paths()

    corpus = inductor.generate_corpus()
    tfidf_matrix, vectorizer = tfidf(corpus, args.keyword_limit, args.n_value, args.df_threshold)
    keywords = vectorizer.get_feature_names_out()
    weights = tfidf_normalized_diff(tfidf_matrix)
    
    # keywords, weights = include_synonyms(keywords, weights) # TODO: experiment if this is needed
    # TODO: experiment to see if I should use instruct_model for suggesting other anomalous keywords based off of high-anomaly score keywords
    # extra_keywords = inductor.generate_response(f"Please suggest other anomalous keywords based off of these keywords: {keywords}")

    output_data = {
        "normal_frame_paths": inductor.normal_frame_paths,
        "anomaly_frame_paths": inductor.anomaly_frame_paths,
        "keywords": {keywords[i]: weights[i] for i in np.argsort(weights)}
    }

    with open(f"benchmarks/{args.dataset}/keywords.json", 'w') as f:
        json.dump(output_data, f, indent=4)