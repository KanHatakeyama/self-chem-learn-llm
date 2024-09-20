# %%
from trl import SFTTrainer
from transformers import TrainingArguments
from trl import DataCollatorForCompletionOnlyLM
import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from peft import get_peft_model, LoraConfig, TaskType
import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"


# %%

model = transformers.AutoModelForCausalLM.from_pretrained(model_id,
                                                          torch_dtype=torch.bfloat16,  device_map="auto")

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
pipeline = transformers.pipeline(
    "text-generation", model=model,
    tokenizer=tokenizer
)

# %%
# LoRA
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                         inference_mode=False,  # 学習時はFalse
                         r=32,
                         lora_alpha=64,
                         lora_dropout=0.05,
                         bias="none",
                         target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                                         "gate_proj", "up_proj", "down_proj",
                                         # "embed_tokens", "lm_head"
                                         ],
                         )

# モデルにLoRAアダプター適用、更新対象のパラメータ数の確認
lora_model = get_peft_model(model, peft_config)
lora_model.print_trainable_parameters()

# %%

# Load the data
# jsonl_path = "data/20240919152121_llm_gen.jsonl"
# jsonl_path = "data/initial_train_data/20240919161042_llm_gen.jsonl"

error_threshold = 0.05
max_records = 200000
jsonl_path_list = [
    "/data/hatakeyama/self-loop/0919meltingpoint/data/initial_train_data/20240919213322_llm_gen.jsonl",
    "/data/hatakeyama/self-loop/0919meltingpoint/data/initial_train_data/20240919161042_llm_gen.jsonl",
]
# Initialize an empty DataFrame
df = pd.DataFrame()

# Read each JSONL file and concatenate the results
for jsonl_path in jsonl_path_list:
    temp_df = pd.read_json(jsonl_path, lines=True)
    df = pd.concat([df, temp_df], ignore_index=True)
# df = pd.read_json(jsonl_path, lines=True)


df = df.drop(columns=["record", "prompt"])
df = df.sort_values(by="error_rate")
df["cond"] = df["CompoundName"]+" "+df["SMILES"]+" "+df["Property"]
df = df.drop_duplicates(subset=["cond"])
df = df[df["error_rate"] < error_threshold]
df = df[:max_records]
df["q"] = "Predict "+df["Property"]+" "+df["Unit"]+" for "+df["CompoundName"] + \
    " (Compound X) with SMILES " + \
    df["SMILES"]+". The prediction consists of #Reason and #Prediction. The #Reason is the quantitative explanation of the prediction. The #Prediction is the predicted value and the unit of the prediction."
df["a"] = "#Reason\n"+df["reason"]+"\n#Prediction\n" + \
    df["predicted"].astype(str)+" "+df["Unit"]
q_list = df["q"].tolist()
a_list = df["a"].tolist()
df.shape

# %%
train_text_list = []
for i in range(len(q_list)):
    q = q_list[i]
    a = a_list[i]
    messages = [
        {"role": "user", "content": q},
        {"role": "assistant", "content": a}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    train_text_list.append(prompt)


# %%
ds = datasets.Dataset.from_list([{"text": t} for t in train_text_list])

# %%

# response_templateは必須指定
response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template, tokenizer=tokenizer)


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['text'])):
        text = f"{example['text'][i]}"
        output_texts.append(text)
    return output_texts


# %%

# SFTTrainerはTrainingArgumentsを使用することができる。
# 指定しない場合、TrainingArgumentsのデフォルトが指定される。
args = TrainingArguments(
    output_dir=f'./output0920_threshold_{error_threshold}_lora_kqvo_proj',
    num_train_epochs=1,
    gradient_accumulation_steps=32,
    per_device_train_batch_size=4,
    save_strategy="steps",
    save_steps=40,
    logging_steps=1,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    warmup_ratio=0.03,
    weight_decay=0.001,
    learning_rate=5e-5,
    # save_total_limit=1,
    fp16=True,
)


# data_collatorが指定されていない場合、以下のようにDataCollatorForLanguageModelingがmlm=Falseで使われる。
# つまり通常のCausal LMを学習することになる。
# if data_collator is None:
#     data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
# packing=False（default）ではdataset_text_fieldかformatting_funcを指定する必要あり
trainer = SFTTrainer(
    lora_model,
    args=args,
    train_dataset=ds,
    formatting_func=formatting_prompts_func,
    max_seq_length=512,
    data_collator=collator,
)


# %%

tokenizer.pad_token = tokenizer.eos_token

# %%
trainer.train()

# %%

trainer.save_model()

# %%
