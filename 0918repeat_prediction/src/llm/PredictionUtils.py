
from vllm import LLM, SamplingParams
import transformers


def init_model_and_tokenizer(
        model_id="meta-llama/Meta-Llama-3.1-70B-Instruct"
):

    tensor_parallel_size = 8
    # トークナイザーとモデルの準備
    model = LLM(
        model=model_id,
        trust_remote_code=True,
        max_model_len=2000,
        tensor_parallel_size=tensor_parallel_size,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer


def llm_gen(model, prompt_list,
            temperature=0.7, top_k=50,
            max_tokens=1024,
            ):
    outputs = model.generate(
        prompt_list,
        sampling_params=SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
        )
    )
    return [i.outputs[0].text.strip() for i in outputs]
