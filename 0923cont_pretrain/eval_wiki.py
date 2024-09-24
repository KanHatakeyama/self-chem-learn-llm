"""
python eval_wiki.py --model_id "meta-llama/Meta-Llama-3.1-70B-Instruct" --tensor_parallel_size 8
python eval_wiki.py --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" --tensor_parallel_size 1

conda activate llama

python eval_wiki.py --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" --tensor_parallel_size 8

export CUDA_VISIBLE_DEVICES=0
python eval_wiki.py --model_id "/data/hatakeyama/self-loop/0923cont_pretrain/sftlab/output/sftlab-experiments/test/1-llama3_1_8b-zero1/checkpoint-130" --tensor_parallel_size 1

export CUDA_VISIBLE_DEVICES=1
python eval_wiki.py --model_id "/data/hatakeyama/self-loop/0923cont_pretrain/sftlab/output/sftlab-experiments/test/1-llama3_1_8b-zero1/checkpoint-260" --tensor_parallel_size 1

export CUDA_VISIBLE_DEVICES=2
python eval_wiki.py --model_id "/data/hatakeyama/self-loop/0923cont_pretrain/sftlab/output/sftlab-experiments/test/1-llama3_1_8b-zero1/checkpoint-390" --tensor_parallel_size 1

export CUDA_VISIBLE_DEVICES=3
python eval_wiki.py --model_id "/data/hatakeyama/self-loop/0923cont_pretrain/sftlab/output/sftlab-experiments/test/1-llama3_1_8b-zero1/checkpoint-520" --tensor_parallel_size 1

export CUDA_VISIBLE_DEVICES=4
python eval_wiki.py --model_id "/data/hatakeyama/self-loop/0923cont_pretrain/sftlab/output/sftlab-experiments/test/1-llama3_1_8b-zero1/checkpoint-650" --tensor_parallel_size 1

export CUDA_VISIBLE_DEVICES=5
python eval_wiki.py --model_id "/data/hatakeyama/self-loop/0923cont_pretrain/sftlab/output/sftlab-experiments/test/1-llama3_1_8b-zero1/checkpoint-682" --tensor_parallel_size 1

"""

# %%
import json
import os
from tqdm import tqdm
from datasets import load_dataset
import transformers
from vllm import LLM, SamplingParams
import argparse

# コマンドライン引数のパーサーを設定
parser = argparse.ArgumentParser(description='モデルIDとテンソルパラレルサイズを指定します。')
parser.add_argument('--model_id', type=str, required=True, help='使用するモデルのID')
parser.add_argument('--tensor_parallel_size', type=int,
                    default=1, help='テンソルパラレルサイズ（デフォルトは1）')

args = parser.parse_args()

model_id = args.model_id
tensor_parallel_size = args.tensor_parallel_size

# トークナイザーとモデルの準備
model = LLM(
    model=model_id,
    trust_remote_code=True,
    max_model_len=2000,
    tensor_parallel_size=tensor_parallel_size,
)

# 以下、元のコードと同じ
# %%


def llm_gen(model, prompt_list, temperature=0.0, top_k=50):
    outputs = model.generate(
        prompt_list,
        sampling_params=SamplingParams(
            temperature=temperature,
            max_tokens=1024,
            # repetition_penalty=1.2,
            top_k=top_k,
        )
    )
    return [i.outputs[0].text.strip() for i in outputs]


tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

# %%

ds = load_dataset("kanhatakeyama/material-properties", split="train")
# ds=ds.shuffle()

# %%


def gen_problem(record):
    comp_name = record["CompName"]
    smiles = record["SMILES"]
    unit = record["unit"]
    property_name = record["Property"]
    q = f"""Predict the {property_name} {unit} of the following compound. 
    #Restriction: Output can contain "#Reason" section which explains the reasoning behind the prediction. Output must contain "#Prediction" section which contains only the predicted value (number only).
    #Name: {comp_name}
    #SMILES: {smiles}"""
    actual_value = record["Value"]

    chat = [
        {"role": "user", "content": q},
    ]

    prompt = tokenizer.apply_chat_template(chat, tokenize=False,)
    assist_prefix = "assistant\n\n"
    prompt += assist_prefix

    return prompt, actual_value

# %%


def extract_answer_from_text(text):
    target_line = text.split("#Prediction")[-1].strip()
    if target_line.find("\n"):
        target_line = target_line.split("\n")[0].strip()
    noise = [" ", "\t", "#"]
    for n in noise:
        target_line = target_line.replace(n, "")
    return target_line


# %%
predictions = []
batch_size = 1000
train_ds = ds.select(range(7000))
test_ds = ds.select(range(7000, 7200))

mode = "test"
if mode == "test":
    target_ds = test_ds
else:
    target_ds = train_ds

for i in tqdm(range(0, len(target_ds), batch_size)):
    max_i = min(i+batch_size, len(target_ds))
    batch = target_ds.select(range(i, max_i))
    prompts = []
    for record in batch:
        prompt, actual_value = gen_problem(record)
        prompts.append(prompt)
    outputs = llm_gen(model, prompts)
    for record, output in zip(batch, outputs):
        predicted_value = extract_answer_from_text(output)
        record["predicted_value"] = predicted_value
        record["predicted_text"] = output
        predictions.append(record)

    # break

# %%
os.makedirs("eval_results", exist_ok=True)
save_path = f"eval_results/{model_id.replace('/','_')}_{mode}.json"
with open(save_path, "w") as f:
    json.dump(predictions, f, indent=2)
