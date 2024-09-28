"""

conda activate llama
export CUDA_VISIBLE_DEVICES=0

#full
python eval_wiki_auto.py --checkpoint_dir "sftlab/experiments/test/1/output3/1-llama3_1_8b_3-zero1" --tensor_parallel_size 1 --mode test --out_path "eval_results_full"
python eval_wiki_auto.py --checkpoint_dir "sftlab/experiments/test/1/output3/1-llama3_1_8b_3-zero1" --tensor_parallel_size 1 --mode train --out_path "eval_results_full"

#lora
python eval_wiki_auto.py --checkpoint_dir "sftlab/experiments/lora_r_llama/1/output3/1-llama3_1_8b_3_lora-zero1" \
    --tensor_parallel_size 1 --out_path "eval_results_lora_llama" \
    --enable_lora True --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --mode test 
python eval_wiki_auto.py --checkpoint_dir "sftlab/experiments/lora_r_llama/1/output3/1-llama3_1_8b_3_lora-zero1" \
    --tensor_parallel_size 1 --out_path "eval_results_lora_llama" \
    --enable_lora True --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --mode train

#70b lora    
python eval_wiki_auto.py --checkpoint_dir "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/70b_lora_llama_all_r128/1/output3/sftlab-experiments/70b_lora_llama_all_r128/1-llama3_1_70b_lora_full_r128-zero3" \
    --tensor_parallel_size 8 --out_path "eval_results_lora_full_r128_70b" \
    --enable_lora true --model_id "meta-llama/Meta-Llama-3.1-70B-Instruct" \
    --mode test 
python eval_wiki_auto.py --checkpoint_dir "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/70b_lora_llama_all_r128/1/output3/sftlab-experiments/70b_lora_llama_all_r128/1-llama3_1_70b_lora_full_r128-zero3" \
    --tensor_parallel_size 8 --out_path "eval_results_lora_full_r128_70b" \
    --enable_lora true --model_id "meta-llama/meta-llama-3.1-70b-instruct" \
    --mode train



#lora with fifferent r
conda activate llama
export CUDA_VISIBLE_DEVICES=0
 python eval_wiki_auto.py --checkpoint_dir "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/lora_llama_all/1/output3/sftlab-experiments/lora_llama_all/1-llama3_1_8b_4_lora_full-zero1" \
    --tensor_parallel_size 1 --out_path "eval_results_lora_r128" \
    --enable_lora True --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" 
 python eval_wiki_auto.py --checkpoint_dir "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/lora_llama_all/1/output3/sftlab-experiments/lora_llama_all/1-llama3_1_8b_4_lora_full-zero1" \
    --tensor_parallel_size 1 --out_path "eval_results_lora_r128" \
    --enable_lora True --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --mode train
    
conda activate llama
export CUDA_VISIBLE_DEVICES=1
 python eval_wiki_auto.py --checkpoint_dir "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/lora_llama_all_r256/1/output3/sftlab-experiments/lora_llama_all_r256/1-llama3_1_8b_5_lora_full_r256-zero1" \
    --tensor_parallel_size 1 --out_path "eval_results_lora_r256" \
    --enable_lora True --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" 
 python eval_wiki_auto.py --checkpoint_dir "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/lora_llama_all_r256/1/output3/sftlab-experiments/lora_llama_all_r256/1-llama3_1_8b_5_lora_full_r256-zero1" \
    --tensor_parallel_size 1 --out_path "eval_results_lora_r256" \
    --enable_lora True --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --mode train

 

conda activate llama
export CUDA_VISIBLE_DEVICES=2
 python eval_wiki_auto.py --checkpoint_dir "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/lora_llama_all_r512/1/output3/sftlab-experiments/lora_llama_all_r512/1-llama3_1_8b_6_lora_full_r512-zero1" \
    --tensor_parallel_size 1 --out_path "eval_results_lora_r512" \
    --enable_lora True --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" 
    --mode train
 python eval_wiki_auto.py --checkpoint_dir "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/lora_llama_all_r512/1/output3/sftlab-experiments/lora_llama_all_r512/1-llama3_1_8b_6_lora_full_r512-zero1" \
    --tensor_parallel_size 1 --out_path "eval_results_lora_r512" \
    --enable_lora True --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
 
conda activate llama
export CUDA_VISIBLE_DEVICES=3
 python eval_wiki_auto.py --checkpoint_dir "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/lora_llama_all_r1024/1/output3/sftlab-experiments/lora_llama_all_r1024/1-llama3_1_8b_7_lora_full_r1024-zero1" \
    --tensor_parallel_size 1 --out_path "eval_results_lora_r1024" \
    --enable_lora True --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" 
    --mode train
 python eval_wiki_auto.py --checkpoint_dir "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/lora_llama_all_r1024/1/output3/sftlab-experiments/lora_llama_all_r1024/1-llama3_1_8b_7_lora_full_r1024-zero1" \
    --tensor_parallel_size 1 --out_path "eval_results_lora_r1024" \
    --enable_lora True --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
 
conda activate llama
export CUDA_VISIBLE_DEVICES=4
 python eval_wiki_auto.py --checkpoint_dir "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/lora_llama_all_r2048/1/output3/sftlab-experiments/lora_llama_all_r2048/1-llama3_1_8b_7_lora_full_r2048-zero1" \
    --tensor_parallel_size 1 --out_path "eval_results_lora_r2048" \
    --enable_lora True --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" 
    --mode train
 python eval_wiki_auto.py --checkpoint_dir "/data/hatakeyama/self-loop/0924split_train_test/sftlab/experiments/lora_llama_all_r2048/1/output3/sftlab-experiments/lora_llama_all_r2048/1-llama3_1_8b_7_lora_full_r2048-zero1" \
    --tensor_parallel_size 1 --out_path "eval_results_lora_r2048" \
    --enable_lora True --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
 


    


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
parser.add_argument('--tensor_parallel_size', type=int,
                    default=1, help='テンソルパラレルサイズ（デフォルトは1）')
parser.add_argument('--mode', type=str,
                    default="test", help='test or train')
parser.add_argument('--out_path', type=str,
                    default="eval_results", help='output dir')
parser.add_argument('--model_id', type=str,
                    default="", help='output dir')
parser.add_argument('--enable_lora', type=bool,
                    default=False, help='enable lora')

args = parser.parse_args()

mode = args.mode
model_id = args.model_id
tensor_parallel_size = args.tensor_parallel_size
out_path = args.out_path
enable_lora = args.enable_lora

model = None


def llm_gen(model, prompt_list, temperature=0.0,
            top_k=50,
            global_lora_id=0,
            lora_path="",
            enable_lora=False,
            ):
    if not enable_lora:
        outputs = model.generate(
            prompt_list,
            sampling_params=SamplingParams(
                temperature=temperature,
                max_tokens=1024,
                # repetition_penalty=1.2,
                top_k=top_k,
            )
        )
    else:
        outputs = model.generate(
            prompt_list,
            sampling_params=SamplingParams(
                temperature=temperature,
                max_tokens=1024,
                top_k=top_k,
            ),
            lora_request=LoRARequest(
                "adapter", global_lora_id, lora_path=lora_path)

        )
    return [i.outputs[0].text.strip() for i in outputs]


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


# get model dir
model_dir_list = glob.glob(f"{args.checkpoint_dir}/checkpoint-*")

if len(model_dir_list) == 0 and model_id != "":
    model_dir_list = [model_id]

print("model_dir_list", model_dir_list)

global_model_count = 0

for model_id in model_dir_list:
    global_model_count += 1
    del model
    torch.cuda.empty_cache()
    print("model_id", model_id)

    # トークナイザーとモデルの準備
    if not enable_lora:
        load_model = model_id
    else:
        print("lora base model for lora", model_id)
        load_model = args.model_id  # loraの場合はbaseモデルを指定

    model = LLM(
        model=load_model,
        trust_remote_code=True,
        max_model_len=2000,
        tensor_parallel_size=tensor_parallel_size,
        max_lora_rank=256,
        enable_lora=enable_lora,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

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

        if not enable_lora:
            outputs = llm_gen(model, prompts)
        else:
            outputs = llm_gen(model, prompts, global_lora_id=global_model_count,
                              lora_path=model_id, enable_lora=True)
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
