import argparse

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.llama.model import Llama
from src.nlp.chattemplates import LlamaChatTemplate
from src.nlp.generation import beam_search_generation


class HuggingFaceGenerationAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.block_size = model.config.max_position_embeddings

    def forward(self, indices, use_kv_cache=False):
        return self.model(indices, use_cache=use_kv_cache).logits

    def get_kv_cache_seqlen(self):
        return 0


@torch.no_grad()
def beam_generate(model, input_ids, max_new_tokens, beam_size, eos_token_id=None):
    generated = input_ids.clone()
    chunks = []
    for chunk in beam_search_generation(
        model,
        input_ids,
        n_tokens_to_gen=max_new_tokens,
        beam_size=beam_size,
        sample=False,
        use_kv_cache=False,
    ):
        for token in chunk:
            chunks.append(token)
            if eos_token_id is not None and token == eos_token_id:
                break
        if eos_token_id is not None and chunks and chunks[-1] == eos_token_id:
            break

    if chunks:
        generated = torch.cat(
            [generated, torch.tensor(chunks, dtype=torch.long, device=input_ids.device)[None, :]],
            dim=1,
        )
    return generated


def compare_models(model_name, prompt, max_new_tokens, beam_size, cache_dir, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    chat_template = LlamaChatTemplate()
    chat_prompt = chat_template.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
    )

    hf_model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    my_model = Llama.from_pretrained(model_name, cache_dir)

    hf_model.eval().to(device)
    my_model.eval().to(device)

    inputs = tokenizer(chat_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    hf_logits = hf_model(input_ids).logits
    my_logits = my_model(input_ids)

    logits_diff = (hf_logits - my_logits).abs()
    print(f"prompt: {prompt!r}")
    print(f"chat prompt: {chat_prompt!r}")
    print(f"logits max diff: {logits_diff.max().item():.8f}")
    print(f"logits mean diff: {logits_diff.mean().item():.8f}")

    hf_generated = beam_generate(
        HuggingFaceGenerationAdapter(hf_model),
        input_ids,
        max_new_tokens,
        beam_size,
        eos_token_id=tokenizer.eos_token_id,
    )
    my_generated = beam_generate(
        my_model,
        input_ids,
        max_new_tokens,
        beam_size,
        eos_token_id=tokenizer.eos_token_id,
    )

    same_tokens = torch.equal(hf_generated, my_generated)
    hf_new_tokens = hf_generated[0, input_ids.shape[1]:]
    my_new_tokens = my_generated[0, input_ids.shape[1]:]
    print(f"same generated token ids: {same_tokens}")
    print(f"hf token ids: {hf_generated[0].tolist()}")
    print(f"my token ids: {my_generated[0].tolist()}")
    print(f"hf new token ids: {hf_new_tokens.tolist()}")
    print(f"my new token ids: {my_new_tokens.tolist()}")
    print(f"hf text: {tokenizer.decode(hf_new_tokens, skip_special_tokens=True)!r}")
    print(f"my text: {tokenizer.decode(my_new_tokens, skip_special_tokens=True)!r}")


def main():
    parser = argparse.ArgumentParser(description="Compare Hugging Face Llama against local Llama implementation.")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--prompt", default="The quick brown fox")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    compare_models(
        model_name=args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        beam_size=args.beam_size,
        cache_dir=args.cache_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
