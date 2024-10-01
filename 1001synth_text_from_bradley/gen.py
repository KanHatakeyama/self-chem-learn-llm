# %%
# auto reload modules
from src.PredictionUtils import init_model_and_tokenizer, llm_gen
import copy
from datasets import load_dataset
import json
import random
import os
from datetime import datetime


pid = os.getpid()
seed = int(pid)+int(datetime.now().timestamp())
print("seed: ", seed)
random.seed(seed)

ds = load_dataset("kanhatakeyama/material-properties", split="Bradley")
ds = ds.shuffle(seed=1)
print(f"total {len(ds)} records")

out_dir = "data"
os.system(f"mkdir -p {out_dir}")

current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")


# %%
model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
model, tokenizer = init_model_and_tokenizer(model_id)

# %%


# %%
train_ds = ds.select(range(24000))
test_ds = ds.select(range(24000, 24800))
train_ds[0]

# %%

batch_size = 2000

target_ds = train_ds


def record_to_list_text(record):
    record = copy.deepcopy(record)
    record.pop("Source")
    text = json.dumps(record)
    return text


style_list = [
    "textbook-style",
    "conversation-style",
    "classroom-style",
    "Q&A-style",
    "interview-style",
    "journal-style",
]
prompt_template = """Prepare {} text according to the following data.
- Never include any other number which is not given from the data.
- Generate only the text, not e.g., system outputs.
- Add some discussion or explanation if necessary.
"""


def gen_prompt(tokenizer, target_ds, style_list):
    style = random.choice(style_list)
    q = prompt_template.format(style)

    for i in range(random.randint(1, 4)):
        record = target_ds[random.randint(0, len(target_ds)-1)]
        q += record_to_list_text(record)+"\n"

    q = q.strip()
    chat = [
        {"role": "user", "content": q},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False,)
    assist_prefix = "assistant\n\n"
    prompt += assist_prefix
    return prompt


for i in range(288):
    prompt_list = [(gen_prompt(tokenizer, target_ds, style_list))
                   for i in range(batch_size)]
    predicted_text_list = llm_gen(model, prompt_list)

    for i in range(len(predicted_text_list)):
        with open(f"data/{current_time_no_symbols}_llm_gen.jsonl", "a", encoding='utf-8') as f:
            f.write(json.dumps(
                {"text": predicted_text_list[i]}, ensure_ascii=False)+"\n")
