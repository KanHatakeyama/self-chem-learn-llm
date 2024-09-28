"""

conda activate llama
export CUDA_VISIBLE_DEVICES=0


# lora 70b
python eval_wiki_auto_transformers.py --checkpoint_dir "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/70b_lora_llama_all_r128/1/output3/sftlab-experiments/70b_lora_llama_all_r128/1-llama3_1_70b_lora_full_r128-zero3" \
   --out_path "eval_results_lora_r128_70b" --mode test
python eval_wiki_auto_transformers.py --checkpoint_dir "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/70b_lora_llama_all_r128/1/output3/sftlab-experiments/70b_lora_llama_all_r128/1-llama3_1_70b_lora_full_r128-zero3" \
   --out_path "eval_results_lora_r128_70b" --mode train

python eval_wiki_auto_transformers.py --checkpoint_dir "none" --model_id "/data/hatakeyama/self-loop/925split_train_test/sftlab/experiments/70b_lora_llama_all_r128/1/output3/sftlab-experiments/70b_lora_llama_all_r128/1-llama3_1_70b_lora_full_r128-zero3/checkpoint-10" \
   --out_path "eval_results_lora_r128_70b" --mode test
python eval_wiki_auto_transformers.py --checkpoint_dir "none" --model_id "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/70b_lora_llama_all_r128/1/output3/sftlab-experiments/70b_lora_llama_all_r128/1-llama3_1_70b_lora_full_r128-zero3/checkpoint-20" \
   --out_path "eval_results_lora_r128_70b" --mode test
python eval_wiki_auto_transformers.py --checkpoint_dir "none" --model_id "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/70b_lora_llama_all_r128/1/output3/sftlab-experiments/70b_lora_llama_all_r128/1-llama3_1_70b_lora_full_r128-zero3/checkpoint-30" \
   --out_path "eval_results_lora_r128_70b" --mode test
python eval_wiki_auto_transformers.py --checkpoint_dir "none" --model_id "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/70b_lora_llama_all_r128/1/output3/sftlab-experiments/70b_lora_llama_all_r128/1-llama3_1_70b_lora_full_r128-zero3/checkpoint-40" \
   --out_path "eval_results_lora_r128_70b" --mode test
python eval_wiki_auto_transformers.py --checkpoint_dir "none" --model_id "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/70b_lora_llama_all_r128/1/output3/sftlab-experiments/70b_lora_llama_all_r128/1-llama3_1_70b_lora_full_r128-zero3/checkpoint-50" \
   --out_path "eval_results_lora_r128_70b" --mode test
python eval_wiki_auto_transformers.py --checkpoint_dir "none" --model_id "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/70b_lora_llama_all_r128/1/output3/sftlab-experiments/70b_lora_llama_all_r128/1-llama3_1_70b_lora_full_r128-zero3/checkpoint-80" \
   --out_path "eval_results_lora_r128_70b" --mode test
python eval_wiki_auto_transformers.py --checkpoint_dir "none" --model_id "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/70b_lora_llama_all_r128/1/output3/sftlab-experiments/70b_lora_llama_all_r128/1-llama3_1_70b_lora_full_r128-zero3/checkpoint-90" \
   --out_path "eval_results_lora_r128_70b" --mode test
python eval_wiki_auto_transformers.py --checkpoint_dir "none" --model_id "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/70b_lora_llama_all_r128/1/output3/sftlab-experiments/70b_lora_llama_all_r128/1-llama3_1_70b_lora_full_r128-zero3/checkpoint-97" \
   --out_path "eval_results_lora_r128_70b" --mode test









#lora with fifferent r
conda activate llama
export CUDA_VISIBLE_DEVICES=0
python eval_wiki_auto_transformers.py --checkpoint_dir "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/lora_llama_all/1/output3/sftlab-experiments/lora_llama_all/1-llama3_1_8b_4_lora_full-zero1" \
   --out_path "eval_results_lora_r128" --mode test
python eval_wiki_auto_transformers.py --checkpoint_dir "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/lora_llama_all/1/output3/sftlab-experiments/lora_llama_all/1-llama3_1_8b_4_lora_full-zero1" \
   --out_path "eval_results_lora_r128" --mode train
    
conda activate llama
export CUDA_VISIBLE_DEVICES=1
python eval_wiki_auto_transformers.py --checkpoint_dir "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/lora_llama_all_r256/1/output3/sftlab-experiments/lora_llama_all_r256/1-llama3_1_8b_5_lora_full_r256-zero1" \
   --out_path "eval_results_lora_r256" --mode test
python eval_wiki_auto_transformers.py --checkpoint_dir "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/lora_llama_all_r256/1/output3/sftlab-experiments/lora_llama_all_r256/1-llama3_1_8b_5_lora_full_r256-zero1" \
   --out_path "eval_results_lora_r256" --mode train



conda activate llama
export CUDA_VISIBLE_DEVICES=2
python eval_wiki_auto_transformers.py --checkpoint_dir "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/lora_llama_all_r512/1/output3/sftlab-experiments/lora_llama_all_r512/1-llama3_1_8b_6_lora_full_r512-zero1" \
   --out_path "eval_results_lora_r512" --mode test
python eval_wiki_auto_transformers.py --checkpoint_dir "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/lora_llama_all_r512/1/output3/sftlab-experiments/lora_llama_all_r512/1-llama3_1_8b_6_lora_full_r512-zero1" \
   --out_path "eval_results_lora_r512" --mode train


 
conda activate llama
export CUDA_VISIBLE_DEVICES=3
python eval_wiki_auto_transformers.py --checkpoint_dir "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/lora_llama_all_r1024/1/output3/sftlab-experiments/lora_llama_all_r1024/1-llama3_1_8b_7_lora_full_r1024-zero1" \
   --out_path "eval_results_lora_r1024" --mode test
python eval_wiki_auto_transformers.py --checkpoint_dir "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/lora_llama_all_r1024/1/output3/sftlab-experiments/lora_llama_all_r1024/1-llama3_1_8b_7_lora_full_r1024-zero1" \
   --out_path "eval_results_lora_r1024" --mode train




conda activate llama
export CUDA_VISIBLE_DEVICES=4
python eval_wiki_auto_transformers.py --checkpoint_dir "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/lora_llama_all_r2048/1/output3/sftlab-experiments/lora_llama_all_r2048/1-llama3_1_8b_7_lora_full_r2048-zero1" \
   --out_path "eval_results_lora_r2048" --mode test
python eval_wiki_auto_transformers.py --checkpoint_dir "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/lora_llama_all_r2048/1/output3/sftlab-experiments/lora_llama_all_r2048/1-llama3_1_8b_7_lora_full_r2048-zero1" \
   --out_path "eval_results_lora_r2048" --mode train

    


"""

# %%
import glob
import torch
import json
import os
from tqdm import tqdm
from datasets import load_dataset
import transformers
from vllm import LLM, SamplingParams
import argparse

from vllm.lora.request import LoRARequest
# コマンドライン引数のパーサーを設定
parser = argparse.ArgumentParser(description='モデルIDとテンソルパラレルサイズを指定します。')
parser.add_argument('--checkpoint_dir', type=str,
                    required=True, help='使用するモデル群のcheckpointディレクトリ')
parser.add_argument('--mode', type=str,
                    default="test", help='test or train')
parser.add_argument('--out_path', type=str,
                    default="eval_results", help='output dir')
parser.add_argument('--model_id', type=str,
                    default="", help='output dir')


args = parser.parse_args()

mode = args.mode
model_id = args.model_id
out_path = args.out_path

model = None


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

    # chat = [
    #    {"role": "user", "content": q},
    # ]

    # prompt = tokenizer.apply_chat_template(chat, tokenize=False,)
    # assist_prefix = "assistant\n\n"
    # prompt += assist_prefix

    return q, actual_value

# %%


def extract_answer_from_text(text):
    target_line = text.split("#Prediction")[-1].strip()
    if target_line.find("\n"):
        target_line = target_line.split("\n")[0].strip()
    noise = [" ", "\t", "#"]
    for n in noise:
        target_line = target_line.replace(n, "")
    return target_line


def llm_gen(pipe, q_list):
    a_list = []
    for q in tqdm(q_list):
        messages = [
            {"role": "user", "content": q},
        ]
        prompt = pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        outputs = pipe(prompt, max_new_tokens=1024,
                       temperature=0.0, do_sample=False)
        out_text = outputs[0]["generated_text"].lstrip(prompt)
        a_list.append(out_text)
    return a_list


# get model dir
model_dir_list = glob.glob(f"{args.checkpoint_dir}/checkpoint-*")

if len(model_dir_list) == 0 and model_id != "":
    model_dir_list = [model_id]

print("model_dir_list", model_dir_list)


for model_id in model_dir_list:
    del model
    torch.cuda.empty_cache()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, device_map="auto")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    pipe = transformers.pipeline(
        'text-generation', model=model, tokenizer=tokenizer)

    # %%

    ds = load_dataset("kanhatakeyama/material-properties", split="wiki")
    ds = ds.shuffle(seed=1)

    # %%

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

        outputs = llm_gen(pipe, prompts)
        for record, output in zip(batch, outputs):
            predicted_value = extract_answer_from_text(output)
            record["predicted_value"] = predicted_value
            record["predicted_text"] = output
            predictions.append(record)

    # %%
    os.makedirs(out_path, exist_ok=True)
    save_path = f"{out_path}/{model_id.replace('/','_')}_{mode}.json"
    with open(save_path, "w") as f:
        json.dump(predictions, f, indent=2)
