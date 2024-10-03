"""

conda activate llama
export CUDA_VISIBLE_DEVICES=0

#full

conda activate llama
export CUDA_VISIBLE_DEVICES=0
python eval_wiki_auto.py --checkpoint_dir "sftlab/experiments/exp/1002brad/output3/sftlab-experiments/exp/1002brad-10000_20000-zero1" --tensor_parallel_size 1 --mode both --out_path "eval_results_1002"

conda activate llama
export CUDA_VISIBLE_DEVICES=1
python eval_wiki_auto.py --checkpoint_dir "sftlab/experiments/exp/1002brad/output3/sftlab-experiments/exp/1002brad-20000_20000-zero1" --tensor_parallel_size 1 --mode both --out_path "eval_results_1002"

conda activate llama
export CUDA_VISIBLE_DEVICES=2
python eval_wiki_auto.py --checkpoint_dir "sftlab/experiments/exp/1002brad/output3/sftlab-experiments/exp/1002brad-40000_20000-zero1" --tensor_parallel_size 1 --mode both --out_path "eval_results_1002"

conda activate llama
export CUDA_VISIBLE_DEVICES=3
python eval_wiki_auto.py --checkpoint_dir "sftlab/experiments/exp/1002brad/output3/sftlab-experiments/exp/1002brad-80000_20000-zero1" --tensor_parallel_size 1 --mode both --out_path "eval_results_1002"

conda activate llama
export CUDA_VISIBLE_DEVICES=4
python eval_wiki_auto.py --checkpoint_dir "sftlab/experiments/exp/1002brad/output3/sftlab-experiments/exp/1002brad-160000_20000-zero1" --tensor_parallel_size 1 --mode both --out_path "eval_results_1002"

conda activate llama
export CUDA_VISIBLE_DEVICES=5
python eval_wiki_auto.py --checkpoint_dir "sftlab/experiments/exp/1002brad/output3/sftlab-experiments/exp/1002brad-320000_20000-zero1" --tensor_parallel_size 1 --mode both --out_path "eval_results_1002"

conda activate llama
export CUDA_VISIBLE_DEVICES=6
python eval_wiki_auto.py --checkpoint_dir "sftlab/experiments/exp/1002brad/output3/sftlab-experiments/exp/1002brad-570000_20000-zero1" --tensor_parallel_size 1 --mode both --out_path "eval_results_1002"



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
                    default="both", help='test or train')
parser.add_argument('--out_path', type=str,
                    default="eval_results", help='output dir')
parser.add_argument('--model_id', type=str,
                    default="", help='output dir')
parser.add_argument('--enable_lora', type=bool,
                    default=False, help='enable lora')
parser.add_argument('--dataset_name', type=str,
                    default="kanhatakeyama/material-properties", help='dataset name')
parser.add_argument('--dataset_split', type=str,
                    default="Bradley", help='dataset split')
parser.add_argument('--train_id_end', type=int,
                    default=24000, help='train_id_end')
parser.add_argument('--test_id_end', type=int,
                    default=24800, help='test_id_end')

args = parser.parse_args()

mode = args.mode
model_id = args.model_id
tensor_parallel_size = args.tensor_parallel_size
out_path = args.out_path
enable_lora = args.enable_lora

dataset_name = args.dataset_name
dataset_split = args.dataset_split
train_id_end = args.train_id_end
test_id_end = args.test_id_end

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

    if mode == "both":
        mode_list = ["test", "train"]
    else:
        mode_list = [mode]
    for mode in mode_list:
        print("mode", mode)

        ds = load_dataset(dataset_name, split=dataset_split)
        ds = ds.shuffle(seed=1)

        # %%

        # %%
        predictions = []
        batch_size = 1000
        train_ds = ds.select(range(train_id_end))
        test_ds = ds.select(range(train_id_end, test_id_end))

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
