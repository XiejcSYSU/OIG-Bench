import math
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
import pandas as pd
from tqdm import tqdm
import os
import re
import argparse

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="InternVL2_5-38B")
args = parser.parse_args()

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def parse_file_name(file_name):
    name_part = file_name.rsplit('.', 1)[0]
    parts = name_part.split('_')

    category = parts[0]      
    type_ = parts[1]        
    lang = parts[2]          
    number = parts[3]        
    return category, type_, lang, number

# If you set `load_in_8bit=True`, you will need two 80GB GPUs.
# If you set `load_in_8bit=False`, you will need at least three 80GB GPUs.
path = f'OpenGVLab/{args.model}'
model_name = args.model

device_map = split_model(path)
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=False,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map).eval()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

zh_prompt = "你的任务是为一张一图流攻略生成一段详细描述，将攻略内容清晰地表述出来。在撰写描述时，请遵循以下指南：\n1. 仔细研读攻略图片文本信息，确保理解其中的所有步骤和要点。\n2. 按照攻略的逻辑顺序，将各个步骤详细且有条理地描述出来\n3. 使用清晰、易懂的语言，只需描述图中的内容，不要描述图片中没有提及到的内容，不要进行介绍\n4. 确保描述内容完整，包含所有必要的信息。"

en_prompt = "Your task is to generate a detailed description for a 'One-Image Guide' (single-image itinerary). Clearly convey the content of the guide by following these guidelines:\n1. Carefully read and understand all the textual information in the guide image, ensuring you grasp every step and key point.\n2. Present each step in a detailed and well-organized manner, following the logical sequence of the guide.\n3. Use clear and easy-to-understand language. Only describe the content shown in the image — do not include any information that is not mentioned in the image, and do not provide additional introductions.\n4. Ensure the description is complete and contains all necessary information."


folder_path = '../../data/fig'


df = pd.read_excel('../../data/description.xlsx')
new_descriptions = []
file_list = df['file'].tolist()

for index, row in tqdm(df.iterrows(), total=len(df), disable=False):
    filename = row['file']
    image_path = os.path.join(folder_path, filename)
    pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()

    # set the max number of tiles in `max_num`
    # pixel_values = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=1024, do_sample=True)


    # single-image single-round conversation (单图单轮对话)
    category, type_, lang, number = parse_file_name(filename)
    if lang == 'zh':
        prompt = zh_prompt
    else:
        prompt = en_prompt
    question = '<image>\n' + prompt
    description = model.chat(tokenizer, pixel_values, question, generation_config)
    
    new_descriptions.append(description)

new_df = pd.DataFrame({'file': file_list, 'pred': new_descriptions})
new_df.to_excel(f'./{model_name}_description.xlsx', index=False)


df = pd.read_excel('../../data/vqa.xlsx')
file_list = df['file'].tolist()
question_list, choice_list, answer_list, pred_list = [], [], [], []

for index, row in tqdm(df.iterrows(), total=len(df), disable=False):
    filename = row['file']
    q = row['question']
    choice = row['choice']
    answer = row['answer']
    image_path = os.path.join(folder_path, filename)
    pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()

    # set the max number of tiles in `max_num`
    # pixel_values = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=1024, do_sample=True)

    category, type_, lang, number = parse_file_name(filename)
    if lang == 'zh':

        prompt = f"""你是一个视觉问答系统。现在给你一张图片、一个问题和若干选项，请细观察图片内容，推理出最符合事实的选项。
    注意：
    - 只根据图片和问题中的信息作答，不要引入外部知识。
    - 只有一个选项是正确的。
    - 只需要输出答案(A,B,C,D)，切勿输出其他内容，包括思考过程。
    <题目>
    {q}
    </题目>
    <选项>
    {choice}
    </选项>
    """
    else:
        prompt = f"""You are a visual question answering system. Now given an image, a question and several options, please carefully observe the image content and reason out the option that best fits the facts.
    Note:
    - Answer based only on the information in the image and question, do not introduce external knowledge.
    - Only one option is correct.
    - Only output the answer (A, B, C, D), do not output anything else.
    <Question>
    {q}
    </Question>
    <Options>
    {choice}
    </Options>
    """

    # single-image single-round conversation (单图单轮对话)
    question = '<image>\n' + prompt
    description = model.chat(tokenizer, pixel_values, question, generation_config)
    file_list.append(filename)
    question_list.append(question)
    choice_list.append(choice)
    answer_list.append(answer)
    pred_list.append(description)

new_df = pd.DataFrame({'file': file_list, 'question': question_list, 'choice': choice_list, 'answer': answer_list, 'pred': pred_list})
new_df.to_excel(f'./{model_name}_VQA.xlsx', index=False)
