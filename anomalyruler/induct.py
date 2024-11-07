import argparse
import torch
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

from llm import *
from image2text import cogvlm
from utils import random_select_data_without_copy

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='SHTech', choices=['SHTech', 'avenue', 'ped2', 'UBNormal'], help='Dataset name')
    parser.add_argument('--model_path', type=str, default='/data/tjf5667/.cache/models--THUDM--cogvlm-chat-hf/snapshots/e29dc3ba206d524bf8efbfc60d80fc4556ab0e3c')
    parser.add_argument('--root', type=str, default='/data/tjf5667/datasets/', help='Root directory for datasets')
    parser.add_argument('--b', type=int, default=10, help='Batch number')
    parser.add_argument('--bs', type=int, default=1, help='Batch size')
    parser.add_argument('--activity_amount', type=int, default=10, help='Number of normal activities to derive')
    parser.add_argument('--object_amount', type=int, default=10, help='Number of normal objects to derive')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    # data_name = args.data
    # dataset_name = {'SHTech':'ShanghaiTech', 'avenue':'CUHK Avenue' , 'ped2': 'UCSD Ped2', 'UBNormal': 'UBNormal'}[args.data]

    # with init_empty_weights():
    #     cog_model = AutoModelForCausalLM.from_pretrained(
    #         'THUDM/cogvlm-chat-hf',
    #         torch_dtype=torch.bfloat16,
    #         low_cpu_mem_usage=True,
    #         trust_remote_code=True
    #     ).eval()

    # device_map = infer_auto_device_map(
    #     cog_model,
    #     max_memory={
    #         0: '12GiB', 1: '12GiB', 2: '12GiB', 3: '12GiB', 4: '12GiB',
    #         5: '12GiB', 6: '12GiB', 7: '12GiB', 8: '12GiB', 9: '12GiB', 'cpu': '32GiB'
    #     },
    #     no_split_module_classes=['CogVLMDecoderLayer', 'TransformerLayer']
    # )

    # cog_model = load_checkpoint_and_dispatch(
    #     cog_model,
    #     args.model_path,
    #     device_map=device_map,
    # )

    # batch, batch_size = args.b, args.bs
    # for i in range(batch):
    #     selected_image_paths = [
    #         f"{args.root}{path}" for path in random_select_data_without_copy(
    #         path=f"{data_name}/train.csv", num=batch_size, label=0
    #         )
    #     ]

    # frame_descriptions = cogvlm(model=cog_model, mode='chat', image_paths=selected_image_paths)
    # print(frame_descriptions)

    # TODO: update method signature for cogvlm

    # Generated frame descriptions using cogvlm
    frame_descriptions = [
        'There are four people in the image. Starting from the left, the first person is walking with a backpack, the second person is walking alone, the third person is also walking alone, and the fourth person is walking with another person. Other than people, there are two manhole covers on the ground, some plants, and a trash bin.',
        'There are three people in the image. One person is walking towards the bridge, another is walking away from the bridge, and the third person is standing on the bridge. Other than people, there are trees, a pathway, a bridge, and some architectural structures in the image.',
        'There are five people in the image. Starting from the left, the first person appears to be walking with a bag. The second person is also walking, but further away. The third person is standing and seems to be looking down. The fourth person is walking with a bag, and the fifth person is walking with a child. Other than people, there are two manhole covers visible in the image.',
        'There are three people in the image. One person is walking towards the left, another is walking towards the right, and the third person is walking on the bridge. Other than people, there are trees, a pond, a bridge, and a building in the image.',
        'There are three people in the image. One person is standing near the trees, another is walking on the path, and the third is closer to the building. Other than people, there are trees, a pathway, lampposts, and a building.',
        'There are five people in the image. Starting from the left, the first person appears to be walking with a backpack. The second person is also walking, but further to the right. The third person is standing near a tree, possibly looking at it or waiting. The fourth person is walking towards the tree, and the fifth person is walking away from the tree. Other than people, there are trees, pathways, and some grassy areas in the image.',
        'There are two people in the image. One person is walking on the pathway, and the other person is standing on the edge of the pathway. Other than people, there is a water body, a fence, and a sign in the image.',
        'There are two people in the image. One person is walking with a backpack, and the other person is walking behind the first. Other than people, there are trees, a fence, a water body, and a bench visible in the image.',
        'There are four people in the image. Starting from the left, the first person is walking away from the camera, the second person is walking towards the camera, the third person is also walking towards the camera, and the fourth person is walking away from the camera. Other than people, there are posters on the wall, a door, and a sidewalk.',
        'There are two people in the image. One person is walking towards the left, and the other person is walking towards the right. Other than people, there are trees, lampposts, a fence, and a building.'
    ]

    # Load the model
    tokenizer, model = load_model(model_path="meta-llama/Llama-3.2-3B-Instruct")

    # Derive and generate normal activities
    activity_prompt = derive_normal_activity(frame_descriptions, args.activity_amount)
    activity_output = generate_output(activity_prompt, tokenizer, model)
    normal_activities = parse_output(activity_output, args.activity_amount)

    # Derive and generate normal objects
    object_prompt = derive_normal_objects(frame_descriptions, args.object_amount)
    object_output = generate_output(object_prompt, tokenizer, model)
    normal_objects = parse_output(object_output, args.object_amount)

    print(normal_activities, normal_objects)

    # Derive and generate abnormal activities if normal activities are found
    if normal_activities:
        activity_prompt = derive_abnormal_activities(normal_activities, args.activity_amount)
        activity_output = generate_output(activity_prompt, tokenizer, model)
        abnormal_activities = parse_output(activity_output, args.activity_amount)
        print(abnormal_activities)

    # Derive and generate abnormal objects if normal objects are found
    if normal_objects:
        object_prompt = derive_abnormal_objects(normal_objects, args.object_amount)
        object_output = generate_output(object_prompt, tokenizer, model)
        abnormal_objects = parse_output(object_output, args.object_amount)
        print(abnormal_objects)


    