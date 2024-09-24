
from vllm import LLM, SamplingParams
import transformers
from vllm.lora.request import LoRARequest


def init_model_and_tokenizer(
        model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
        enable_lora=False,
        max_model_len=2000,
        tensor_parallel_size=8,
):

    model = LLM(model=model_id,
                enable_lora=enable_lora,
                trust_remote_code=True,
                max_model_len=max_model_len,
                tensor_parallel_size=tensor_parallel_size,
                max_lora_rank=64,
                )

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer


def llm_gen(model, prompt_list,
            temperature=0.7, top_k=50,
            max_tokens=1024,
            enable_lora=False,
            lora_id=None,
            lora_path=None,
            ):
    if not enable_lora:
        outputs = model.generate(
            prompt_list,
            sampling_params=SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                top_k=top_k,
            )
        )
    else:
        outputs = model.generate(
            prompt_list,
            sampling_params=SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                top_k=top_k,
            ),
            lora_request=LoRARequest("adapter", lora_id, lora_path=lora_path)

        )

    return [i.outputs[0].text.strip() for i in outputs]
