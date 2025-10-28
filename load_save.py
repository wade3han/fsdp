import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, LlamaRMSNorm
import functools
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


if __name__ == "__main__":
    torch.distributed.init_process_group(backend="nccl")
    
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    device_id = torch.distributed.get_rank()
    auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={LlamaMLP, LlamaAttention, LlamaRMSNorm, nn.Embedding})
    model = FSDP(model, device_id=device_id, auto_wrap_policy=auto_wrap_policy, sync_module_states=True)

    no_decay_name_list = [
        "embedding",
        "embed_tokens",
        "bias",
        "ln",
        "norm",
    ]

    decay_params = []
    no_decay_params = []
    frozen_params = []

    for n, p in model.named_parameters():
        if p.requires_grad:
            if any(ndnl in n for ndnl in no_decay_name_list):
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        else:
            frozen_params.append(p)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": 0.01},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if frozen_params:
        optimizer_grouped_parameters.append({
            "params": frozen_params,
            "lr": 0.0,
            "weight_decay": 0.0
        })

    opt = torch.optim.AdamW(optimizer_grouped_parameters, 
                            lr=0.000005, 
                            betas=(0.9, 0.999), 
                            weight_decay=0.01, 
                            eps=1e-8)

    state_dict = opt.state_dict()
    print(f"DEBUG: [rank {torch.distributed.get_rank()}] state_dict: {state_dict.keys()}")

    optim_state_dict = FSDP.optim_state_dict(model, opt)
    print(f"DEBUG: [rank {torch.distributed.get_rank()}] optim_state_dict: {optim_state_dict.keys()}")