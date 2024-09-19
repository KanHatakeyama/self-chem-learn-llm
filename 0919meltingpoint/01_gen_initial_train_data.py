# %%
# auto reload modules
from src.prop.utils import gen_reason, gen_masked_prediction_problem_prompts, parse_prediction_with_check
from src.llm.PredictionUtils import init_model_and_tokenizer, llm_gen
from tqdm import tqdm
from datasets import load_dataset
import json
import random
import os
from datetime import datetime


pid = os.getpid()
seed = int(pid)+int(datetime.now().timestamp())
print("seed: ", seed)
random.seed(seed)


out_dir = "data/initial_train_data"
os.system(f"mkdir -p {out_dir}")

current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")


# %%
model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
model, tokenizer = init_model_and_tokenizer(model_id)

# %%

ds = load_dataset("kanhatakeyama/material-properties", split="Bradley")
ds = ds.shuffle(seed=1)
len(ds)

# %%
train_ds = ds.select(range(24000))
test_ds = ds.select(range(24000, 24800))
train_ds[0]

# %%
n_records = 500

generated_records = []
for _ in tqdm(range(10**3)):
    problems = []
    prompt_list = [
    ]
    for i in range(n_records):
        record = random.choice(train_ds)
        # prompt,actual_value=gen_problem(record,tokenizer)
        prompt, actual_value = gen_reason(record, tokenizer)
        problems.append(
            {
                "record": record,
                "prompt": prompt,
                "actual_value": actual_value
            }

        )
        prompt_list.append(prompt)

    # interference
    predicted_text_list = llm_gen(model, prompt_list)

    # predict properties according to the generated reasonings.
    check_prompt_list, masked_reason_list = gen_masked_prediction_problem_prompts(
        predicted_text_list, problems, tokenizer)
    predicted_value_list = llm_gen(model, check_prompt_list)

    # parse result
    # prediction_records=parse_prediction(problems,predicted_text_list)
    prediction_records = parse_prediction_with_check(
        problems, masked_reason_list, predicted_value_list)

    generated_records.extend(prediction_records)
    for i in range(len(prediction_records)):
        with open(f"{out_dir}/{current_time_no_symbols}_llm_gen.jsonl", "a", encoding='utf-8') as f:
            f.write(json.dumps(prediction_records[i], ensure_ascii=False)+"\n")
