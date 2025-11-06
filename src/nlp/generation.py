"""Generation algorithms."""

import numpy as np
from typing import List, Callable

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from typing import List, Callable

def batch_generation(model, 
                     indices: torch.Tensor,
                     n_tokens_to_gen: int,
                     top_k: int = None,
                     top_p: float = None,
                     temperature: float = 1.0,
                     sample: bool = True,
                     use_kv_cache: bool = True):
    """Generates batch of responses with KV caching.
    
    NOTE: If KV caching, we will guarantee that:
        (kv_cache_seqlen after) == (kv_cache_seqlen before) + len(indices) + len(response)
    This will be true even if the generator terminates early, as long as generator.close() is called

    Args:
        model: Model instance. model should take in tensors of shape (batch, seqlen) and output logits of
            shape (batch, seqlen, vocab_size).
        indices (Tensor): Conditioning sequence with shape (batch, seqlen).
        n_tokens_to_gen (int): Number of tokens to generate.
        top_k (int): Filter probabilities to those in the top k.
        top_p (int): Nucleus sampling. Filter to top probs such that the sum is just less than top_p.
        temperature (float: Higher temperature raises the likelihood of lower probability sequences.
        sample (bool): True to randomly sample sequences from the distribution of probabilities
            False to take argmax.
        use_kv_cache (bool): If True, uses kv_cache to speed up inference.
            If generator terminates early, make sure to call generator.close() to properly maintain the KV cache.
            
            After generation, we will guarantee that:
                (kv_cache_seqlen after) == (kv_cache_seqlen before) + len(indices) + len(response)
            
    Returns
        generator. The generator will yield tokens as soon as they are sampled.
            
    Examples:
        (Pseudocode) In the following, response2 == kv_response2 while being faster to generate.
    
            prompt1 = 'Hi, I am John.'
            prompt2 = 'That is great.'
    
            response1 = batch_generation(prompt1, use_kv_cache=False)
            response2 = batch_generation(prompt1 + response1 + prompt2, use_kv_cache=False)
    
            response1 = batch_generation(prompt1, use_kv_cache=True)
            kv_response2 = batch_generation(prompt2, use_kv_cache=True)

    """
    model.eval()

    for token_n in range(n_tokens_to_gen):
        if indices.shape[1] >= model.block_size:
            raise RuntimeError(f'Conversation has reached the limit of {model.block_size} tokens.')

        with torch.no_grad():
            indices_to_input = indices
            if use_kv_cache:
                # After the first step, feed in one token at a time
                if token_n > 0:
                    indices_to_input = indices_to_input[:, -1:]

            next_token_logits = model(indices_to_input, use_kv_cache)[:, -1]

        probs = F.softmax(next_token_logits / (temperature + 1e-6), dim=-1).T  # shape (vocab_size, batch)
        (vocab_size, batch) = probs.shape

        if top_k is not None:
            probs = top_k_sample(probs, top_k)

        if top_p is not None:
            probs = nucleus_sample(probs, top_p)

        if sample:
            next_indices = [torch.multinomial(probs[:, i], 1).item()
                            for i in range(batch)]
        else:
            next_indices = torch.argmax(probs, dim=0).tolist()

        next_indices_tensor = torch.tensor(next_indices, dtype=torch.long, device=indices.device)[:, None]

        try:
            yield next_indices
        except GeneratorExit:
            # This means that the generator exited early. We have to feed the last
            # generated indices back in to maintain the KV cache
            if use_kv_cache:
                _ = model(next_indices_tensor, use_kv_cache=True)
            return

        indices = torch.cat([indices, next_indices_tensor], dim=1)

    # We have to feed the last generated indices back in to maintain the KV cache
    if use_kv_cache:
        _ = model(next_indices_tensor, use_kv_cache=True)
        
        
def top_k_sample(probs: torch.Tensor,
                 top_k: int):
    """Top-k sampling.
    
    Args:
        probs (torch.Tensor): Tensor of probabilities with shape (vocab_size, batch).
            Modifies probs in place.
        top_k (int): Top K words to filter to.
    
    """
    # For each row, zero out everything except for top_k probs per row
    top_k_prob = torch.sort(probs, dim=0)[0][-top_k, :]
    probs[probs < top_k_prob] = 0
    
    probs /= probs.sum(dim=0)
    
    return probs


def nucleus_sample(probs: torch.Tensor,
                   top_p: float):
    """Nucleus sampling. Filter to top probs such that the sum prob is just less than top_p.
    
    References:
        [1] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi.
            The Curious Case of Neural Text Degeneration. arXiv:1904.09751, 2019
    
    Args:
        probs (torch.Tensor): Tensor of probabilities with shape (vocab_size, batch).
            Modifies probs in place.
        top_p (float): Filter to the top `k` probs such that the sum probs is <= top_p and k is largest.
    
    """
    sorted_probs = torch.sort(probs, dim=0, descending=True)[0]
    cum_probs = torch.cumsum(sorted_probs, dim=0)
    top_k = (cum_probs <= top_p).sum(dim=0)

    ranking = probs.shape[0] - torch.argsort(torch.argsort(probs, dim=0), dim=0)
    mask = (ranking <= top_k) | (ranking == 1)  # | (ranking == 1) accounts for when the edge case if highest prob > top_p

    probs[~mask] = 0
    probs /= probs.sum(dim=0)

    return probs


# def batch_generation(model, 
#                      indices: torch.Tensor,
#                      n_tokens_to_gen: int,
#                      top_k: int = None,
#                      top_p: float = None,
#                      temperature: float = 1.0,
#                      sample: bool = True,
#                      use_kv_cache: bool = True):
#     """Generates batch of responses with KV caching.
    
#     NOTE: If KV caching, we will guarantee that:
#         (kv_cache_seqlen after) == (kv_cache_seqlen before) + len(indices) + len(response)
#     This will be true even if the generator terminates early, as long as generator.close() is called

#     Args:
#         model: Model instance. model should take in tensors of shape (batch, seqlen) and output logits of
#             shape (batch, seqlen, vocab_size).
#         indices (Tensor): Conditioning sequence with shape (batch, seqlen).
#         n_tokens_to_gen (int): Number of tokens to generate.
#         top_k (int): Filter probabilities to those in the top k.
#         top_p (int): Nucleus sampling. Filter to top probs such that the sum is just less than top_p.
#         temperature (float: Higher temperature raises the likelihood of lower probability sequences.
#         sample (bool): True to randomly sample sequences from the distribution of probabilities
#             False to take argmax.
#         use_kv_cache (bool): If True, uses kv_cache to speed up inference.
#             If generator terminates early, make sure to call generator.close() to properly maintain the KV cache.
            
#             After generation, we will guarantee that:
#                 (kv_cache_seqlen after) == (kv_cache_seqlen before) + len(indices) + len(response)
            
#     Returns
#         generator. The generator will yield tokens as soon as they are sampled.
            
#     Examples:
#         (Pseudocode) In the following, response2 == kv_response2 while being faster to generate.
    
#             prompt1 = 'Hi, I am John.'
#             prompt2 = 'That is great.'
    
#             response1 = batch_generation(prompt1, use_kv_cache=False)
#             response2 = batch_generation(prompt1 + response1 + prompt2, use_kv_cache=False)
    
#             response1 = batch_generation(prompt1, use_kv_cache=True)
#             kv_response2 = batch_generation(prompt2, use_kv_cache=True)

#     """
#     model.eval()

#     for token_n in range(n_tokens_to_gen):
#         if indices.shape[1] >= model.block_size:
#             raise RuntimeError(f'Conversation has reached the limit of {model.block_size} tokens.')

#         with torch.no_grad():
#             indices_to_input = indices
#             if use_kv_cache and token_n > 0:
#                 # After the first step, feed in one token at a time
#                 indices_to_input = indices_to_input[:, -1:]

#             next_token_logits = model(indices_to_input, use_kv_cache)[:, -1]

#         probs = F.softmax(next_token_logits / (temperature + 1e-6), dim=-1).T  # shape (vocab_size, batch)
#         (vocab_size, batch) = probs.shape

#         if top_k is not None:
#             probs = top_k_sample(probs, top_k)

#         if top_p is not None:
#             probs = nucleus_sample(probs, top_p)

#         if sample:
#             next_indices = torch.multinomial(probs.T, 1).squeeze(1)[:, None]
#         else:
#             next_indices = torch.argmax(probs, dim=0)[:, None]

#         try:
#             yield next_indices
#         except GeneratorExit:
#             # This means that the generator exited early. We have to feed the last
#             # generated indices back in to maintain the KV cache
#             if use_kv_cache:
#                 _ = model(next_indices, use_kv_cache=True)
#             return

#         indices = torch.cat([indices, next_indices], dim=1)
        
# def top_k_sample(probs: torch.Tensor, top_k: int):
#     """
#     Performs top-k sampling on the probability distribution.

#     Args:
#         probs: (vocab_size, batch) - Input probability distribution.
#         top_k: int - Number of top probabilities to keep.

#     Returns:
#         torch.Tensor: (vocab_size, batch) - Probability distribution after masking.
#     """
#     # Get the top-k largest values and their indices along the vocab dimension (dim=0)
#     # torch.topk returns values and indices; we only need the threshold values
#     top_k_vals, _ = torch.topk(probs, k=top_k, dim=0) # Shape: (top_k, batch)
#     # Get the k-th largest value for each batch item (the smallest among the top-k)
#     kth_largest = top_k_vals[-1, :] # Shape: (batch,)
#     # Create a mask: True if prob >= threshold, False otherwise
#     mask = probs >= kth_largest.unsqueeze(0) # Unsqueeze for broadcasting: (1, batch)
#     # Apply the mask: set probabilities below the threshold to 0
#     masked_probs = probs * mask.float()
#     # Renormalize the probabilities so they sum to 1 for each batch item
#     renormalized_probs = masked_probs / masked_probs.sum(dim=0, keepdim=True)
#     return renormalized_probs


# def nucleus_sample(probs: torch.Tensor, top_p: float):
#     """
#     Performs nucleus (top-p) sampling on the probability distribution.

#     Args:
#         probs: (vocab_size, batch) - Input probability distribution.
#         top_p: float - Cumulative probability threshold.

#     Returns:
#         torch.Tensor: (vocab_size, batch) - Probability distribution after masking.
#     """
#     sorted_probs, sort_indices = torch.sort(probs, descending=True, dim=0) # Shapes: (vocab_size, batch)
#     cum_probs = torch.cumsum(sorted_probs, dim=0) # Shape: (vocab_size, batch)
#     exceeding_mask = cum_probs > top_p # Shape: (vocab_size, batch)
#     first_exceeding_idx = torch.argmax(exceeding_mask.float(), dim=0) # Shape: (batch,)
#     no_exceed_mask = ~exceeding_mask[-1, :] # True if cumsum never exceeded top_p (last cumsum <= top_p)
#     mask = torch.zeros_like(probs, dtype=torch.bool) # Shape: (vocab_size, batch)
#     vocab_indices = torch.arange(probs.shape[0], device=probs.device).unsqueeze(1) # Shape: (vocab_size, 1)
#     indices_less_than_exceeding = vocab_indices < first_exceeding_idx.unsqueeze(0) # Shape: (vocab_size, batch)
#     mask = indices_less_than_exceeding | no_exceed_mask.unsqueeze(0) # Shape: (vocab_size, batch)
#     original_indices = torch.arange(probs.shape[0], device=probs.device).unsqueeze(1).expand_as(sort_indices) # Shape: (vocab_size, batch)
#     inverse_sort_indices = torch.argsort(sort_indices, dim=0) # Shape: (vocab_size, batch)
#     original_order_mask = torch.zeros_like(mask, dtype=torch.bool)
#     original_order_mask.scatter_(0, inverse_sort_indices, mask) # Apply mask to original order
    
#     masked_probs = probs * original_order_mask.float()
#     renormalized_probs = masked_probs / masked_probs.sum(dim=0, keepdim=True)
#     return renormalized_probs


# def top_k_sample(probs: torch.Tensor,
#                  top_k: int):
#     # probs : (vocab_size, batch)
#     top_k_probs = torch.sort(probs, dim=0)[0][-top_k, :]
#     probs[probs < top_k_probs] = 0
#     probs /= probs.sum(dim=0)
#     return probs

# def nucleus_sample(probs: torch.Tensor,
#                    top_p: float):
#     # probs: (vocab_size, batch)
#     sorted_probs = torch.sort(probs, descending=True)[0]
#     cum_probs = sorted_probs.cumsum(dim=0)
#     top_k = (cum_probs <= top_p).sum(dim=0)
#     ranking = probs.shape[0] - torch.argsort(torch.argsort(probs, dim=0), dim=0)
#     mask = (ranking <= top_k) | (ranking == 1)
#     probs[~mask] = 0
#     probs /= probs.sum(dim=0)
#     return probs

def beam_search_generation(model,
                           indices: torch.Tensor,
                           n_tokens_to_gen: int,
                           beam_size: int,
                           top_k: int=None,
                           top_p: float=None,
                           temperature: float=1.,
                           sample: bool=True,
                           use_kv_cache: bool=True,
                           modify_kv_cache_func: callable=None):
    model.eval()

    device = next(model.parameters()).device

    if modify_kv_cache_func is None:
        modify_kv_cache_func = default_modify_kv_cache
    if beam_size is None:
        beam_size = 1
    if use_kv_cache:
        final_kv_cache_len = model.get_kv_cache_seqlen() + len(indices)
    
    cumulative_log_prob_per_beam = torch.Tensor([0.0])
    head_index = indices.shape[1]

    for token_n in range(n_tokens_to_gen):
        if indices.shape[1] >= model.block_size:
            raise RuntimeError(f"Conversation has reached the limit of {model.block_size} tokens.")
        with torch.no_grad():
            indices_to_input = indices
            if use_kv_cache:
                if token_n > 0:
                    indices_to_input = indices_to_input[:, -1:]
            next_token_logits = model(indices_to_input, use_kv_cache)[:, -1]
        probs = F.softmax(next_token_logits / temperature, dim=0).T

        if top_k is not None:
            probs = top_k_sample(probs, top_k)
        if top_p is not None:
            probs = nucleus_sample(probs, top_p)
        
        probs += 1.0e-3 / len(probs)
        log_probs = torch.log(probs) + cumulative_log_prob_per_beam

        if sample:
            normalized_probs = F.softmax(log_probs.flatten())
            next_beam_indices = torch.multinomial(normalized_probs, beam_size, replacement=False)
        else:
            next_beam_indices = log_probs.flatten().argsort(descending=True)[-beam_size:]

        cumulative_log_prob_per_beam = log_probs.flatten()[next_beam_indices]
        new_indices = torch.zeros((beam_size, indices.shape[1] + 1), dtype=torch.long, device=device)
        reindex_kv_indices = []
        for (i, beam_index) in enumerate(next_beam_indices):
            next_index = beam_index // indices.shape[0]
            indices_i = beam_index % indices.shape[0]
        
            new_indices[i] = torch.cat([indices[indices_i], torch.tensor([next_index], dtype=torch.long, device=device)], dim=0)
            reindex_kv_indices.append(indices_i)
        indices = torch.Tensor(new_indices)

        if use_kv_cache:
            modify_kv_cache_func(model, reindex_kv_indices=reindex_kv_indices)
        indices_at_head = indices[:, head_index]
        while torch.unique(indices_at_head).numel() == 1:
            try:
                head_index += 1
                yield [int(indices_at_head[0])]
            except GeneratorExit:
                if use_kv_cache:
                    modify_kv_cache_func(model,
                                         trim_seqlen=final_kv_cache_len,
                                         reindex_kv_indices=[0])
                return
            
            if head_index == indices.shape[1]:
                break
            
            indices_at_head = indices[:, head_index]
    
    if sample:
        best_index = torch.multinomial(F.softmax(cumulative_log_prob_per_beam, dim=0), 1).item()
    else:
        best_index = cumulative_log_prob_per_beam.argmax().item()
    
    if use_kv_cache:
        modify_kv_cache_func(model, trim_seqlen=final_kv_cache_len,
                             reindex_kv_indices=[best_index])
    yield indices[best_index, head_index:].long().tolist()

def default_modify_kv_cache(model,
                            trim_seqlen: int=None,
                            reindex_batch_indices: list[int]=None):
    for decoder_block in model.decoder_blocks:
        if decoder_block.attn.kv_cache is not None:
            (key_cache, value_cache) = decoder_block.attn.kv_cache

            if trim_seqlen is not None:
                if trim_seqlen > 0:
                    key_cache = key_cache[..., :trim_seqlen]
                    value_cache = value_cache[..., :trim_seqlen]
                else:
                    key_cache = None
                    value_cache = None
            if reindex_batch_indices:
                key_cache = key_cache.data[reindex_batch_indices]
                value_cache = value_cache.data[reindex_batch_indices]
            decoder_block.attn.kv_cache = (key_cache, value_cache)