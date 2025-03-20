from functools import partial

import torch
import torch.distributed as dist
from torch import is_tensor
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


def setup_distributed_mode(rank, world_size, backend="nccl"):
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed_mode():
    dist.destroy_process_group()


def add_gumbel_noise(logits, temperature):
    '''
    As suggested by https://arxiv.org/pdf/2409.02908, we use float64 for the gumbel max method.
    '''
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


@torch.no_grad()
def ar_sample_kvcache(gpt, prompt, pc, temperature=0.5, context_length=90000, window_size=9000, device='cuda'):
    gpt.eval()
    N = prompt.shape[0]
    end_list = [0 for _ in range(N)]
    with tqdm(total=context_length - 1, desc="Processing") as pbar:
        for cur_pos in range(prompt.shape[1], context_length):
            with torch.amp.autocast(device, dtype=torch.bfloat16):
                if cur_pos >= 9001 and (cur_pos - 9001) % 4500 == 0:
                    start = 4500 + ((cur_pos - 9001) // 4500) * 4500
                else:
                    start = cur_pos - 1
                input_pos = torch.arange(cur_pos, dtype=torch.long, device=device)
                prompt_input = prompt[:, start:cur_pos]
                logits = gpt(prompt_input, pc=pc, start=start, window_size=window_size, input_pos=input_pos)[:, -1]

            logits_with_noise = add_gumbel_noise(logits, temperature)
            next_token = torch.argmax(logits_with_noise, dim=-1, keepdim=True)

            prompt = torch.cat([prompt, next_token], dim=-1)

            pbar.set_description(f"with start:{start},cur_pos:{cur_pos},length:{prompt_input.size(1)}")
            pbar.update(1)

            for u in range(N):
                if end_list[u] == 0:
                    if next_token[u] == torch.tensor([4737], device=device):
                        end_list[u] = 1
            if sum(end_list) == N:
                break
    return prompt, cur_pos


def first(it):
    return it[0]


def custom_collate(data, pad_id):
    is_dict = isinstance(first(data), dict)

    if is_dict:
        keys = first(data).keys()
        data = [d.values() for d in data]

    output = []

    for datum in zip(*data):
        if is_tensor(first(datum)):
            datum = pad_sequence(datum, batch_first=True, padding_value=pad_id)
        else:
            datum = list(datum)

        output.append(datum)

    output = tuple(output)

    if is_dict:
        output = dict(zip(keys, output))

    return output


def build_dataloader_func(bs, dataset, local_rank, world_size):
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=bs,
        num_workers=0,
        drop_last=False,
        collate_fn=partial(custom_collate, pad_id=4737)
    )
    return dataloader
