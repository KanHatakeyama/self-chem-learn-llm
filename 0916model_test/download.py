# %%
import transformers
import torch
import time

while True:
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    try:
        pipeline = transformers.pipeline(
            "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
        )
        pipeline("Hey how are you doing today?")
        break
    except Exception as e:
        print(e)
        time.sleep(3600)

while True:
    model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    try:
        pipeline = transformers.pipeline(
            "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
        )
        pipeline("Hey how are you doing today?")
        break
    except Exception as e:
        print(e)
        time.sleep(3600)
# %%
