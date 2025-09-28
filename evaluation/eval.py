import argparse
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
import os
import json
import numpy as np
from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--eval_file", type=str)
parser.add_argument("--processes", type=int, default=0, help="Number of processes to use for evaluation. 0 means single process.")

from venus_api_base.http_client import HttpClient
from venus_api_base.config import Config
os.environ['OPENAI_API_KEY'] = "4oS06TRCwgaSPkSNZLoTVz6t@3674"
secret_id = '4oS06TRCwgaSPkSNZLoTVz6t'
secret_key = 'GOYnZDnZJ3LCVrJful0XWmRP'
client = HttpClient(secret_id=secret_id, secret_key=secret_key, config=Config(read_timeout=300))
domain = 'http://v2.open.venus.oa.com'
header = {
    'Content-Type': 'application/json',
}
def chat_with_llm(model, prompt):
    try:
        client = OpenAI(base_url="http://v2.open.venus.oa.com/llmproxy")

        messages=[{"role": "system", "content": "You are an AI assistant."}, 
                        {"role": "user", "content": [{"type": "text", "text": prompt}]
                }]
        
        response = None
        if model != 'gpt-5':
            response = client.chat.completions.create(
                model=model,
                stream=True,
                messages=messages,
                max_tokens=2048,
                temperature=0.0,
                timeout=300,
            )
        else:
            response = client.chat.completions.create(
                model=model,
                stream=True,
                messages=messages
            )

        ret =""

        for chunk in response:
            for choice in chunk.choices:
                if choice.delta and 'reasoning_content' in choice.delta.model_extra:
                    # 思考内容
                    # print(choice.delta.model_extra['reasoning_content'], end="")
                    pass
                if choice.delta and choice.delta.content:
                    # 模型输出
                    # print(choice.delta.content, end="")
                    ret += choice.delta.content
        
        return ret
    except Exception as e:
        print(e)
        return ""

# def chat_with_llm(model, prompt):
#     client = OpenAI(api_key="your api key")  
#     response = client.chat.completions.create(
#         model=model, 
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0
#     )

#     return response.choices[0].message.content


def parse_file_name(file_name):
    name_part = file_name.rsplit('.', 1)[0]
    parts = name_part.split('_')

    category = parts[0]      # travel
    type_ = parts[1]         # logic
    lang = parts[2]          # zh
    number = parts[3]        # 10

    return category, type_, lang, number

def eval_description(input):
    file, pred, gt = input
    category, type_, lang, number = parse_file_name(file)
    if lang == 'zh':
        prompt = f"""你是一位“一图流攻略”图片描述分析专家。
    你将会得到两个输入：

    1. Ground Truth：从图片中人工提取的正确“一图流攻略”行程描述。
    2. 模型预测：模型针对同一张图片生成的描述。

    你的任务：

    1. 仔细对比 Ground Truth 和 模型预测。
    2. 找出 模型预测 中的错误，并将其归类到以下四种类型（注意：每个错误只能归入一个最合适的类型，禁止重复计入多个类型）：
    - 文字识别准确性：
        - 识别图片中文字时的错误，例如OCR错误、拼写错误等。
        - 数字错误不属于文字识别准确性，属于语义一致性。
        - 只要有拼写差异就算错误，无论是否影响整体理解。
        - 禁止将虚构信息计入此项。
    - 细节覆盖度：
        - 遗漏了 Ground Truth 中存在的重要细节，例如缺少地点、活动、属性或上下文信息等。
        - 如果细节只是被简化、同义替换、合并描述，但核心信息仍然保留，则不算遗漏。
        - 只有当细节完全缺失或被错误替换为无关信息时，才视为细节遗漏。
        - 禁止将虚构信息计入此项。
    - 虚构信息检测：
        - 生成了 Ground Truth 中不存在的内容，例如捏造的地点、事件、细节等。
        - 无论是变体、扩展还是完全捏造，均属于虚构。
    - 语义一致性：
        - 与 Ground Truth 在逻辑、顺序、地点、时间、关系等方面的不一致。
        - 轻微的描述差异（同义替换、修辞变化）不应影响评分。
        - 注意只有当差异导致事实错误、逻辑冲突、顺序颠倒、地点/时间/数量错误时，才扣分。
        - 如果细节覆盖度存在重大缺失，则语义一致性必须同时扣分，因为缺失大量关键信息会影响整体语义完整性。

    评分规则（0–1 分）：
    - 1 分：完全正确，无任何错误。
    - 0.8 分：仅有极少量轻微错误，不影响整体理解。
    - 0.6 分：存在一定数量的中等程度错误，影响部分理解。
    - 0.4 分：错误较多，严重影响理解。
    - 0.2 分：错误非常多，几乎无法正确理解。
    - 0 分：完全错误或与 Ground Truth 无关。

    评分步骤（强制执行）：
    1. 先判断错误类型：
    - 如果是拼写/字符/数字识别错误 → 文字识别准确性
    - 如果是遗漏 Ground Truth 中的重要细节 → 细节覆盖度
    - 如果是生成了 Ground Truth 中不存在的内容 → 虚构信息检测
    - 如果是逻辑/顺序/地点/时间/数量错误 → 语义一致性
    2. 禁止跨类型重复计入：
    - 虚构信息 只能计入“虚构信息检测”，不得计入其他类型
    3. 最后打分：根据错误数量和严重程度，按 0–5 分规则评分。

    输出格式要求（必须严格遵守以下 JSON 结构）：
    {{
    "文字识别准确性": {{
        "错误描述": "详细描述在文字识别方面的错误。",
        "评分": 0-1
    }},
    "细节覆盖度": {{
        "错误描述": "详细描述遗漏的细节。",
        "评分": 0-1
    }},
    "虚构信息检测": {{
        "错误描述": "详细描述虚构的内容。",
        "评分": 0-1
    }},
    "语义一致性": {{
        "错误描述": "详细描述在语义一致性方面的错误。",
        "评分": 0-1
    }},
    }}

    现在请分析以下数据：
    Ground Truth：{gt}
    模型预测：{pred}

    """
    elif lang == 'en':
        prompt = f"""You are an “One-Image Strategy” image description analysis expert.
You will be given two inputs:

1. Ground Truth: The correct “One-Image Strategy” itinerary description manually extracted from the image.
2. Model Prediction: The description generated by the model for the same image.

Your task:

1. Carefully compare Ground Truth and Model Prediction.
2. Identify errors in Model Prediction and categorize them into the following four types (note: each error can only be counted in one most appropriate type, do not count multiple types):
- Text Recognition:
    - Errors in recognizing text in the image, such as OCR errors, spelling errors, etc.
    - Numerical errors do not belong to text recognition accuracy, but to semantic consistency.
    - As long as there is a spelling difference, it counts as an error, regardless of whether it affects overall understanding.
    - Do not count hallucinated information in this item.
- Detail Coverage:
    - Important details present in Ground Truth are omitted, such as missing locations, activities, attributes, or contextual information.
    - If details are only simplified, synonymously replaced, or merged in   description, but the core information is still retained, it is not considered missing.
    - Only when details are completely missing or replaced with irrelevant information, it is considered detail omission.
    - Do not count hallucinated information in this item.
- Hallucination Control:
    - Generates content that does not exist in Ground Truth, such as fabricated locations, events, details, etc.
    - Whether it is a variant, expansion, or completely fabricated, it belongs to hallucination.
- Semantic Consistency:
    - Inconsistencies with Ground Truth in terms of logic, order, location, time, relationships, etc.
    - Minor descriptive differences (synonymous replacement, rhetorical changes) should not affect the score.
    - Note that only when the differences lead to factual errors, logical conflicts, order reversal, location/time/quantity errors, points are deducted.
    - If there are significant omissions in detail coverage, semantic consistency must also be deducted, because missing a lot of key information will affect overall semantic integrity.

Scoring Rules (0-1 points):
- 1 point: No errors
- 0.8 points: Only a very small number of minor errors that do not affect overall understanding.
- 0.6 points: A certain number of moderate errors that affect partial understanding.
- 0.4 points: Many errors that seriously affect understanding.
- 0.2 points: Very many errors that make it almost impossible to understand correctly.
- 0 points: Completely wrong or unrelated to the Ground Truth.

Scoring Steps (must be followed):
1. First determine the type of error:
- If it is a spelling/character/number recognition error → Text Recognition
- If important details in Ground Truth are omitted → Detail Coverage
- If content that does not exist in Ground Truth is generated → Hallucination Control
- If it is a logical/order/location/time/quantity error → Semantic Consistency
2. Do not count across types:
- Hallucinated information can only be counted in "Hallucination Control", and cannot be counted in other types
3. Finally score: According to the number and severity of errors, score according to the 0-1 point rule.

Output Format Requirements (must strictly follow the following JSON structure):
{{
"Text Recognition": {{
    "Error Description": "Detailed description of errors in text recognition.",
    "Score": 0-1
}},
"Detail Coverage": {{
    "Error Description": "Detailed description of omitted details.",
    "Score": 0-1
}},
"Hallucination Control": {{
    "Error Description": "Detailed description of hallucinated content.",
    "Score": 0-1
}},
"Semantic Consistency": {{
    "Error Description": "Detailed description of errors in semantic consistency.",
    "Score": 0-1
}},
}}
Now please analyze the following data:
Ground Truth: {gt}
Model Prediction: {pred}
"""

    result = chat_with_llm('gpt-4.1', prompt)
    return result

if __name__ == "__main__":

    path = '../data/'
    description = pd.read_excel(path + "description.xlsx")
    vqa = pd.read_excel(path + "VQA.xlsx")

    args = parser.parse_args()
    eval_file = args.eval_file

    parent_dir = os.path.dirname(eval_file)
    model = os.path.basename(eval_file).split('.')[0]
    model = model.replace('_description', '')

    target_path = os.path.join(parent_dir, f'{model}_description_eval.xlsx')

    eval_df = pd.read_excel(eval_file)

    """ Evaluation Description Generation """

    if not os.path.exists(target_path):

        df = pd.merge(eval_df, description, on='file')
        df = df[df['pred'].notna()]

        file = df['file'].values.tolist()
        labels = df['label'].values.tolist()
        preds = df['pred'].values.tolist()

        evals = []
        with Pool(processes=3) as pool:
                for r in tqdm(pool.imap(eval_description, zip(file, preds, labels)), total=len(preds), disable=False):
                    evals.append(r)

        eval_df = pd.DataFrame({'file': file, 'label': labels, 'pred': preds, 'eval': evals})
        eval_df.to_excel(target_path, index=False)

    else:
         eval_df = pd.read_excel(target_path)

    scores1, scores2, scores3, scores4 = [], [], [], []

    for i, row in eval_df.iterrows():
        category, type_, lang, number = parse_file_name(row['file'])
        try:
            j = json.loads(row['eval'])
            if lang == 'zh':
                scores1.append(j['语义一致性']['评分'])
                scores2.append(j['文字识别准确性']['评分'])
                scores3.append(j['细节覆盖度']['评分'])
                scores4.append(j['虚构信息检测']['评分'])
            elif lang == 'en':
                scores1.append(j['Semantic Consistency']['Score'])
                scores2.append(j['Text Recognition']['Score'])
                scores3.append(j['Detail Coverage']['Score'])
                scores4.append(j['Hallucination Control']['Score'])
        except:
            print('error json', row['file'])
            scores1.append(0)
            scores2.append(0)
            scores3.append(0)
            scores4.append(0)
            continue

    scores = np.array([scores1, scores2, scores3, scores4])
    scores = scores.mean(axis=1)

    """ Evaluation VQA """

    target_path = os.path.join(parent_dir, f'{model}_VQA.xlsx')
    if not os.path.exists(target_path):
        print(f"Model: {model}, Semantic Consistency: {(scores[0])*100:.2f}%, Text Recognition: {(scores[1])*100:.2f}%, Detail Coverage: {(scores[2])*100:.2f}%, Hallucination Control: {(scores[3])*100:.2f}%")
        print('Lack of VQA results')
        exit()
    
    eval_df = pd.read_excel(target_path)
    gt = eval_df['answer'].values.tolist()
    pred = eval_df['pred'].values.tolist()

    correct = 0
    total = 0
    for i in range(len(gt)):
        if str(gt[i]).strip().lower() == str(pred[i]).strip().lower():
            correct += 1
        total += 1
    ACC = correct / total

    print(f"Model: {model}, Semantic Consistency: {(scores[0])*100:.2f}%, Text Recognition: {(scores[1])*100:.2f}%, Detail Coverage: {(scores[2])*100:.2f}%, Hallucination Control: {(scores[3])*100:.2f}%, VQA Accuracy: {ACC*100:.2f}%, Overall: {(scores[0] + scores[1] + scores[2] + scores[3] + ACC)/5*100:.2f}%")
    
