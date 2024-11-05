import shutil
import os

# 源文件夹路径
source_folder = './checkpoints/llava-mistral-v1.5-7b-simple_discrib-sft'
# 目标文件夹路径
destination_folder = '../modelfile/llava-mistral-v1.5-7b-simple_discrib-sft'

# 文件名
model_file_names = [
    'config.json',
    'generation_config.json',
    'model-00001-of-00004.safetensors',
    'model-00002-of-00004.safetensors',
    'model-00003-of-00004.safetensors',
    'model-00004-of-00004.safetensors',
    'model.safetensors.index.json',
    'special_tokens_map.json',
    'tokenizer.model',
    'tokenizer_config.json',
    'tokenizer.json',
    'tokenizer_state.json',
    'trainer_state.json',
    'training_args.bin'

    ]
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

for model_file_name in model_file_names:
    # 源文件全路径
    source_file = os.path.join(source_folder, model_file_name)
    # 目标文件全路径
    destination_file = os.path.join(destination_folder, model_file_name)

    # 移动文件
    try:
        shutil.move(source_file, destination_file)
        print(f"File {model_file_name} moved successfully from {source_folder} to {destination_folder}")
    except Exception as e:
        print(f"Error occurred while moving file {model_file_name}: {e}")