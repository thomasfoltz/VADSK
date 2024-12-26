import argparse, json, nltk, torch
import numpy as np
import pandas as pd

from PIL import Image
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, AutoProcessor, MllamaForConditionalGeneration, BitsAndBytesConfig

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ped2', choices=['SHTech', 'avenue', 'ped2'], help='Dataset name')
    parser.add_argument('--root', type=str, default='/data/tjf5667/datasets/', help='Root directory for datasets')
    parser.add_argument('--batch_size', type=int, default=20)
    return parser.parse_args()

def tfidf(corpus):
    vectorizer = TfidfVectorizer(decode_error='ignore', ngram_range=(1, 2), max_features=500)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix, vectorizer

def tfidf_normalized_diff(tfidf_matrix):
    tfidf_diff = (tfidf_matrix[1] - tfidf_matrix[0]).toarray().flatten()
    tfidf_min, tf_idf_max = np.min(tfidf_diff), np.max(tfidf_diff)
    return (tfidf_diff - tfidf_min) / (tf_idf_max - tfidf_min)

def identify_synsets(keywords):
    nltk.download('wordnet', download_dir='./wordnet/')
    print(nltk.corpus.wordnet.synonyms('biking'))
    print(wn.synsets('biking'))
    print(wn.synsets(keywords[0]))
    print(wn.synsets('dog', pos=wn.VERB))
    breakpoint()

class Induction:
    def __init__(self, args, vlm_model, processor, instruct_model):
        self.args = args
        self.vlm_model = vlm_model
        self.processor = processor
        self.instruct_model = instruct_model
        self.normal_frame_paths = None
        self.abnormal_frame_paths = None
        self.vlm_message = None

    def init_vlm_message(self):
        message = [{"role": "system", "content": "You are a surveillance monitor for urban safety"},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the activities and objects present in this scene."}]}]
        self.vlm_message = self.processor.apply_chat_template(message)

    def init_frame_paths(self):
        def load_frame_paths(file_path, label, batch_size):
            df = pd.read_csv(file_path)
            image_file_paths = df.loc[df['label'] == label, 'image_path'].values
            selected_frame_paths = np.random.choice(image_file_paths, batch_size, replace=False)
            return selected_frame_paths.tolist()

        self.normal_frame_paths = load_frame_paths(f'benchmarks/{self.args.data}/train.csv', 0, self.args.batch_size)
        self.abnormal_frame_paths = load_frame_paths(f'benchmarks/{self.args.data}/test.csv', 1, self.args.batch_size)

    def generate_frame_descriptions(self, frame_paths):
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
        abnormal_frame_descriptions = self.generate_frame_descriptions(self.abnormal_frame_paths)
        return [normal_frame_descriptions, abnormal_frame_descriptions]

    def generate_response(self, prompt):
        message = [{"role": "system", "content": "You are a surveillance monitor for urban safety"}, 
                    {"role": "user", "content": {prompt}},]
        output = self.instruct_model(
            message, 
            max_new_tokens=128, 
            pad_token_id=50256
        )
        return output[0]["generated_text"][-1]
        
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
    )

    # instruct_model = pipeline(
    #     task="text-generation",
    #     model="meta-llama/Llama-3.2-3B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    # )

    processor = AutoProcessor.from_pretrained('meta-llama/Llama-3.2-11B-Vision-Instruct')
    inductor = Induction(args, vlm_model, processor, instruct_model=None)

    inductor.init_vlm_message()
    inductor.init_frame_paths()
   
    corpus = inductor.generate_corpus()
    tfidf_matrix, vectorizer = tfidf(corpus)
    keywords = vectorizer.get_feature_names_out()
    keyword_weights = tfidf_normalized_diff(tfidf_matrix)
    synsets = identify_synsets(keywords)

    output_data = {
        "normal_frame_paths": inductor.normal_frame_paths,
        "abnormal_frame_paths": inductor.abnormal_frame_paths,
        "keywords": {keywords[i]: keyword_weights[i] for i in np.argsort(keyword_weights)}
    }

    with open(f"benchmarks/{args.data}/keywords.json", 'w') as f:
        json.dump(output_data, f, indent=4)