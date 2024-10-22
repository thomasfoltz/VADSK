import argparse
import os
import torch
import pandas as pd
from PIL import Image
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from transformers import LlamaTokenizer, AutoModelForCausalLM
from utils import get_all_paths

def cogvlm(model, image_paths, mode = 'chat', root_path = None, model_path = 'lmsys/vicuna-7b-v1.5'):
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    query= 'How many people are in the image and what is each of them doing? What are in the images other than people? Think step by step'
    if root_path != None:
        image_paths = sorted(get_all_paths(root_path))

    batch_images = [Image.open(p) for p in image_paths]
    description = []

    for image in batch_images:
        if mode == 'chat':
            inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])  # chat mode
        else:
            inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image],
                                                        template_version='vqa')  # vqa mode

        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
        }

        gen_kwargs = {"max_length": 2048, "do_sample": False}

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            description.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
            print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    return description

def continue_frame(data_name):
    file_path = f'{data_name}/test.csv'
    df = pd.read_csv(file_path)
    image_path_sample = df['image_path'][0]
    last_slash_pos = image_path_sample.rfind('/')
    second_last_slash_pos = image_path_sample.rfind('/', 0, last_slash_pos)

    # Extract all unique number segments from the paths
    unique_segments = df.iloc[:, 0].apply(lambda x: x[second_last_slash_pos+1:last_slash_pos].split('/')[0]).unique()
    print(unique_segments)
    if not os.path.exists(f'{data_name}/test_frame'):
        os.makedirs(f'{data_name}/test_frame')
    for i in unique_segments:
        filtered_df = df[df.iloc[:, 0].apply(lambda x: x[second_last_slash_pos+1:last_slash_pos].split('/')[0]== i)]
        filtered_df.to_csv(f'{data_name}/test_frame/test_{i}.csv', index=False)

def get_description_frame(data_name, data_root_dir):
    with init_empty_weights():
        cog_model = AutoModelForCausalLM.from_pretrained(
            'THUDM/cogvlm-chat-hf',
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval()

    device_map = infer_auto_device_map(
        cog_model,
        max_memory={
            0: '12GiB', 1: '12GiB', 2: '12GiB', 3: '12GiB', 4: '12GiB',
            5: '12GiB', 6: '12GiB', 7: '12GiB', 8: '12GiB', 9: '12GiB', 'cpu': '32GiB'
        },
        no_split_module_classes=['CogVLMDecoderLayer', 'TransformerLayer']
    )

    cog_model = load_checkpoint_and_dispatch(
        cog_model,
        args.model_path,
        device_map=device_map,
    )

    all_video_csv_paths = get_all_paths(f'{data_name}/test_frame')
    for video_csv_path in all_video_csv_paths[31:]:   #:30 31:(test_abnormal_scene_4_scenario_7)
        name = video_csv_path.split('/')[-1].split('.')[0]
        try:
            df = pd.read_csv(video_csv_path) 
            img_paths_per_video = [f'{data_root_dir}{path}' for path in df.iloc[:, 0].tolist()]
            descriptions_per_video = cogvlm(model=cog_model, mode='chat', image_paths=img_paths_per_video)
            if not os.path.exists(f'{data_name}/test_frame_description'):
                os.makedirs(f'{data_name}/test_frame_description')
            with open(f'{data_name}/test_frame_description/{name}.txt', 'w') as file:
                for inner_list in descriptions_per_video:
                    file.write(inner_list + '\n')
        except Exception as e:
            print(f"Error processing {video_csv_path}: {e}")
        finally:
            torch.cuda.empty_cache()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # for i in range(torch.cuda.device_count()):
    #     print(f"Device {i}: {torch.cuda.get_device_name(i)}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='SHTech', choices=['SHTech', 'avenue', 'ped2', 'UBNormal'])
    parser.add_argument('--model_path', type=str, default='/data/tjf5667/.cache/models--THUDM--cogvlm-chat-hf/snapshots/e29dc3ba206d524bf8efbfc60d80fc4556ab0e3c')
    parser.add_argument('--root', type=str, default='/data/tjf5667/datasets/')
    args = parser.parse_args()

    dataset_name, root_dir = args.data, args.root
    
    continue_frame(dataset_name)
    get_description_frame(dataset_name, root_dir)



