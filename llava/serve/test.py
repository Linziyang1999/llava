import argparse
import pandas as pd
import torch
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images
from PIL import Image
from io import BytesIO
import json


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

def main(args):
    # Model setup
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

    # Load parquet file
    df = pd.read_parquet(args.parquet_path)

    # Store results
    results = []

    for index, row in df.iterrows():
        question = row['question']
        image_bytes = row['image']
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        image_size = image.size
        # Process image
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = image_tensor.to(args.device, dtype=torch.float16)
        

        
        question = DEFAULT_IMAGE_TOKEN + '\n' + question
        conv.append_message(conv.roles[0], question)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        # Tokenize question
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]

        # Model inference
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True
            )

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        conv.messages=[]
        # Save result
        results.append({'question': question, 'response': output_text})

    # Save all results to a JSON file
    with open('results.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--parquet-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--conv-mode", type=str, default=None)
    args = parser.parse_args()
    main(args)