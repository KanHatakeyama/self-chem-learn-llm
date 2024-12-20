"""
python eval_wiki.py --model_id "meta-llama/Meta-Llama-3.1-70B-Instruct" --tensor_parallel_size 8
python eval_wiki.py --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" --tensor_parallel_size 1

#python eval_wiki.py --model_id "meta-llama/Meta-Llama-3.1-70B-Instruct" --tensor_parallel_size 8 --mode train
python eval_wiki.py --model_id "meta-llama/Meta-Llama-3.1-70B-Instruct" --tensor_parallel_size 8 --mode test



conda activate llama

python eval_wiki.py --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" --tensor_parallel_size 8

conda activate llama
export CUDA_VISIBLE_DEVICES=0
python eval_wiki.py --model_id "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/test/1/output/sftlab-experiments/test/1-llama3_1_8b_2-zero1/checkpoint-20" --tensor_parallel_size 1
python eval_wiki.py --model_id "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/test/1/output/sftlab-experiments/test/1-llama3_1_8b_2-zero1/checkpoint-20" --tensor_parallel_size 1 --mode train

conda activate llama
export CUDA_VISIBLE_DEVICES=1
python eval_wiki.py --model_id "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/test/1/output/sftlab-experiments/test/1-llama3_1_8b_2-zero1/checkpoint-40" --tensor_parallel_size 1
python eval_wiki.py --model_id "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/test/1/output/sftlab-experiments/test/1-llama3_1_8b_2-zero1/checkpoint-40" --tensor_parallel_size 1 --mode train

conda activate llama
export CUDA_VISIBLE_DEVICES=2
python eval_wiki.py --model_id "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/test/1/output/sftlab-experiments/test/1-llama3_1_8b_2-zero1/checkpoint-60" --tensor_parallel_size 1
python eval_wiki.py --model_id "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/test/1/output/sftlab-experiments/test/1-llama3_1_8b_2-zero1/checkpoint-60" --tensor_parallel_size 1 --mode train

conda activate llama
export CUDA_VISIBLE_DEVICES=3
python eval_wiki.py --model_id "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/test/1/output/sftlab-experiments/test/1-llama3_1_8b_2-zero1/checkpoint-80" --tensor_parallel_size 1
python eval_wiki.py --model_id "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/test/1/output/sftlab-experiments/test/1-llama3_1_8b_2-zero1/checkpoint-80" --tensor_parallel_size 1 --mode train

conda activate llama
export CUDA_VISIBLE_DEVICES=4
python eval_wiki.py --model_id "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/test/1/output/sftlab-experiments/test/1-llama3_1_8b_2-zero1/checkpoint-97" --tensor_parallel_size 1
python eval_wiki.py --model_id "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/test/1/output/sftlab-experiments/test/1-llama3_1_8b_2-zero1/checkpoint-97" --tensor_parallel_size 1 --mode train

export CUDA_VISIBLE_DEVICES=5
python eval_wiki.py --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" --tensor_parallel_size 1
python eval_wiki.py --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" --tensor_parallel_size 1 --mode train



#70B
python eval_wiki.py --model_id "meta-llama/Meta-Llama-3.1-70B-Instruct" --tensor_parallel_size 8
python eval_wiki.py --model_id "meta-llama/Meta-Llama-3.1-70B-Instruct" --tensor_parallel_size 8 --mode train

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
parser.add_argument('--mode', type=str,
                    default="test", help='test or train')
parser.add_argument('--out_path', type=str,
                    default="eval_results_", help='output dir')


args = parser.parse_args()

mode = args.mode
model_id = args.model_id
tensor_parallel_size = args.tensor_parallel_size
out_path = args.out_path

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
ds = ds.shuffle(seed=1)

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
os.makedirs(out_path, exist_ok=True)
save_path = f"{out_path}/{model_id.replace('/','_')}_{mode}.json"
with open(save_path, "w") as f:
    json.dump(predictions, f, indent=2)
