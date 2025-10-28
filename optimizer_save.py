import functools
import os
import math

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, LlamaRMSNorm

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful

CHECKPOINT_DIR = "/nas/checkpoint"


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = functools.partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

class AppState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict
        }

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"]
        )


def setup():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    # initialize the process group
    dist.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank)
    return rank


def cleanup():
    dist.destroy_process_group()


def run_fsdp_checkpoint_save_example():
    rank = setup()
    print(f"Running basic FSDP checkpoint saving example on rank {rank}.")

    # create a model and move it to GPU with id rank
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={LlamaMLP, LlamaAttention, LlamaRMSNorm, nn.Embedding})
    fsdp_mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    model = FSDP(model, device_id=torch.device(f"cuda:{rank}"), auto_wrap_policy=auto_wrap_policy, sync_module_states=True, mixed_precision=fsdp_mixed_precision)

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
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, 
                                  lr=0.001, 
                                  betas=(0.9, 0.95), 
                                  weight_decay=0.01, 
                                  eps=1e-8)
    optimizer.zero_grad()
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100)

    model = torch.compile(model)
    for i in range(40):
        print(f"Training step {i}. LR: {scheduler.get_last_lr()[0]}")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            output = model(torch.randint(1, (4, 16), device="cuda")).logits
            loss = output.sum()
        loss.backward()
        optimizer.step()
        scheduler.step()

    state_dict = { "app": AppState(model, optimizer) }
    dcp.save(state_dict, checkpoint_id=CHECKPOINT_DIR)

    print("Save checkpoint successfully")

    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Running fsdp checkpoint example on {world_size} devices.")
    run_fsdp_checkpoint_save_example()