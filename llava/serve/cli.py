import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image
import json
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

def save_attention_matrix(attention_matrix, file_path):
    attention_list = attention_matrix.tolist()  # 将矩阵转换为嵌套列表
    with open(file_path, 'w') as f:
        json.dump(attention_list, f)  # 把嵌套列表写入文件

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image
def print_params_info(model):
    for name, param in model.named_parameters():
        print(f"Layer: {name}")
        print(f"\tRequires Grad: {param.requires_grad}")
        print(f"\tDevice: {param.device}")



def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    prefix = model.base_model_prefix
    print(image_processor.crop_size)
    model.eval()
    print(prefix) 
    # 假设 model 是你已经有的 PyTorch 模型实例
    #print_params_info(model)    
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    image = load_image(args.image_file)
    image_size = image.size
    # Similar operation in model_worker.py
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break
        if inp == "/restart":
            print("pelease input image file")
            image_file = input(f"image file:")
            image = load_image(image_file)
            image_size = image.size
            # Similar operation in model_worker.py
            image_tensor = process_images([image], image_processor, model.config)
            #
            if type(image_tensor) is list:
                image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            conv.messages=[]
            continue
        if inp == "/clear":
            conv.messages=[]
            clear = True
            continue
        if inp == "file":
            # 打开文件并读取内容
            with open("prompt.txt", 'r', encoding='utf-8') as file:
                inp = file.read()
        if inp == "image":
            image_ids,clear_ids = model.get_closest_token_id(image_tensor.unsqueeze(0))
            #image_text = tokenizer.decode(image_ids, skip_special_tokens=True)
            #print(image_text)
            continue
        if inp == "heatmap":
            with torch.inference_mode():
                prompt = conv.get_prompt()
                heatmap_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

                model_outputs = model(input_ids=heatmap_ids, image_sizes=[image_size],images=image_tensor, return_dict=True, output_attentions = True)
                # 获取注意力权重
                all_heads_attention = model_outputs.attentions[-1].detach().cpu()
                attention_matrix_sum = all_heads_attention[0].sum(dim=0)
                num_heads = all_heads_attention.shape[1]
                average_attention_matrix = attention_matrix_sum
                if average_attention_matrix.dtype == torch.bfloat16:
                    average_attention_matrix = average_attention_matrix.to(dtype=torch.float)
                attention_matrix = average_attention_matrix.numpy()
                print(attention_matrix.shape)
                save_attention_matrix(attention_matrix, "./heatmap.json")
            continue

        print(f"{roles[1]}: ", end="")

        if image is not None or clear is True:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
            clear = False
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        print(len(input_ids))
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        #print(model.config)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True)

        outputs = tokenizer.decode(output_ids[0],skip_special_tokens=True).strip()
        
        #print(output_ids[0])
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
