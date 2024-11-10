import argparse
import json
import numpy as np
import pandas as pd
import torch

from PIL import Image
from transformers import pipeline, AutoProcessor, MllamaForConditionalGeneration, BitsAndBytesConfig
from prompts import derive_normal_activity, derive_normal_objects, derive_abnormal_activities, derive_abnormal_objects

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='SHTech', choices=['SHTech', 'avenue', 'ped2', 'UBNormal'], help='Dataset name')
    parser.add_argument('--root', type=str, default='/data/tjf5667/datasets/', help='Root directory for datasets')
    parser.add_argument('--b', type=int, default=1, help='Batch number')
    parser.add_argument('--bs', type=int, default=10, help='Batch size')
    parser.add_argument('--activity_amount', type=int, default=10, help='Number of normal activities to derive')
    parser.add_argument('--object_amount', type=int, default=10, help='Number of normal objects to derive')
    return parser.parse_args()

def generate_rules(text_model, prompt):
    messages = [
        {"role": "system", "content": "You are a surveillance monitor for urban safety"},
        {"role": "user", "content": {prompt}},
    ]

    output = text_model(
        messages,
        max_new_tokens=128,
        pad_token_id=50256
    )

    decoded_output = output[0]["generated_text"][-1]
    response = decoded_output['content'].split('\n')
    rules = []
    for line in response:
        if line.strip() and line[0].isdigit():
            if '. ' in line:
                rules.append(line.split('. ', 1)[1])
            else:
                rules.append(line)
    return rules

def generate_frame_descriptions(vlm_model, processor, image_paths):
    batch_images = [Image.open(p).convert('RGB') for p in image_paths]
    description = []
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "How many people are in the image and what is each of them doing? What are in the images other than people? Think step by step."}]}
     ]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    for image in batch_images:
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to('cuda')

        with torch.no_grad():
            output = vlm_model.generate(**inputs, max_new_tokens=128)
            decoded_output = processor.decode(output[0])
            description.append(decoded_output)

    return description

if __name__ == "__main__":
    args = parse_arguments()
    batch, batch_size = args.b, args.bs
    data_name = args.data
    dataset_name = {'SHTech':'ShanghaiTech', 'avenue':'CUHK Avenue', 'ped2': 'UCSD Ped2', 'UBNormal': 'UBNormal'}[args.data]
    
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

    text_model = pipeline(
        "text-generation",
        model="meta-llama/Llama-3.2-3B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    df = pd.read_csv(f'{data_name}/train.csv')
    image_file_paths = list(df.loc[df['label'] == 0, 'image_path'].values)
    random_image_paths = np.random.choice(image_file_paths, batch_size, replace=False)
    for i in range(batch):
        selected_image_paths = [f"{args.root}{path}" for path in random_image_paths]

    processor = AutoProcessor.from_pretrained('meta-llama/Llama-3.2-11B-Vision-Instruct')
    frame_descriptions = generate_frame_descriptions(vlm_model, processor, selected_image_paths)

    prompt = derive_normal_activity(frame_descriptions, args.activity_amount)
    normal_activities = generate_rules(text_model, prompt)
    if normal_activities:
        prompt = derive_abnormal_activities(normal_activities, args.activity_amount)
        abnormal_activities = generate_rules(text_model, prompt)
    else:
        raise ValueError("No normal activities were given.")
    
    prompt = derive_normal_objects(frame_descriptions, args.object_amount)
    normal_objects = generate_rules(text_model, prompt)
    if normal_objects:
        prompt = derive_abnormal_objects(normal_objects, args.object_amount)
        abnormal_objects = generate_rules(text_model, prompt)
    else:
        raise ValueError("No normal objects were given.")

    output_data = {
        "normal_activities": normal_activities,
        "abnormal_activities": abnormal_activities,
        "normal_objects": normal_objects,
        "abnormal_objects": abnormal_objects
    }

    output_file = f"{data_name}_rules.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Rules saved to {output_file}")


    