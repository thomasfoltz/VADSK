import argparse
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from llm import *
from image2text import cogvlm

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='SHTech', choices=['SHTech', 'avenue', 'ped2', 'UBNormal'], help='Dataset name')
    parser.add_argument('--model_path', type=str, default='/data/tjf5667/.cache/models--THUDM--cogvlm-chat-hf/snapshots/e29dc3ba206d524bf8efbfc60d80fc4556ab0e3c')
    parser.add_argument('--root', type=str, default='/data/tjf5667/datasets/', help='Root directory for datasets')
    parser.add_argument('--b', type=int, default=10, help='Batch number')
    parser.add_argument('--bs', type=int, default=1, help='Batch size')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    data_name = args.data
    data_full_name = {'SHTech':'ShanghaiTech', 'avenue':'CUHK Avenue' , 'ped2': 'UCSD Ped2', 'UBNormal': 'UBNormal'}[data_name]

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

    # rule generation
    objects_list, rule_list = [], []
    batch, batch_size = args.b, args.bs

    for i in range(batch):
        selected_image_paths = [
            f"{args.root}{path}" for path in random_select_data_without_copy(
            path=f"{data_name}/train.csv", num=batch_size, label=0
            )
        ]

        objects = cogvlm(model=cog_model, mode='chat', image_paths=selected_image_paths)
        objects_list.append(objects)
        rule_list.append(gpt_induction(objects, data_full_name))

        torch.cuda.empty_cache()

    gpt_rule_correction(rule_list, batch, data_full_name)

    