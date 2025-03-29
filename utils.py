"""
This file is modified from:
https://github.com/facebookresearch/deit/blob/main/utils.py
"""

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time,random
import numpy as np
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist
import torch.nn.functional as F

import pickle
import gzip

from itertools import groupby
import tensorflow as tf

import matplotlib.pyplot as plt  
import seaborn as sns

def count_parameters_in_MB(model):
    # sum(p.numel() for p in model.parameters() if p.requires_grad)
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters())/1e6

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def sampler_func(clip, sn, random_choice=True):
    if random_choice:
        f = lambda n: [(lambda n, arr: n if arr == [] else np.random.choice(arr))(n * i / sn,
                                                                                range(int(n * i / sn),
                                                                                        max(int(n * i / sn) + 1,
                                                                                            int(n * (
                                                                                                    i + 1) / sn))))
                        for i in range(sn)]
    else:
        f = lambda n: [(lambda n, arr: n if arr == [] else int(np.mean(arr)))(n * i / sn, range(int(n * i / sn),
                                                                                                max(int(
                                                                                                    n * i / sn) + 1,
                                                                                                    int(n * (
                                                                                                            i + 1) / sn))))
                        for i in range(sn)]
    return f(clip)

def cosine_scheduler(base_value, final_value, epochs):
    iters = np.arange(epochs)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    return schedule

def cosine_scheduler_func(base_value, final_value, iters, epochs):
    schedule = lambda x: final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * x / epochs))
    return schedule(iters)

def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object


def yield_tokens(file_path):
    with io.open(file_path, encoding = 'utf-8') as f:
        for line in f:
            yield line.strip().split()

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    
    output = torch.cat(tensors_gather,dim=0)
    return output

def gloss_tokens_to_sequences(tokens,tgt_vocab,type = 'tensor'):
    if type=='list':
        sequences = []
        for token in tokens:
            sequence = tgt_vocab.lookup_tokens(token)
            sequence = ' '.join(sequence)
            sequences.append(sequence)
        return sequences
    else:
        tokens = tokens.transpose(0,1)
        sequences = []
        for i in range(len(tokens)):
            token =  tokens[i,:].tolist()
            for j1 in range(len(token)):
                if token[j1] == PAD_IDX:
                    token = token[0:j1]
                    break
                if j1 == len(token)-1:
                    token = token[0:j1]
            sequence = tgt_vocab.lookup_tokens(token)
            sequence = ' '.join(sequence)
            sequences.append(sequence)
        return sequences

def ctc_decode(gloss_probabilities,sgn_lengths):
    gloss_probabilities = gloss_probabilities.cpu().detach().numpy()
    # tf_gloss_probabilities = np.concatenate(
    #     (gloss_probabilities[:, :, 1:], gloss_probabilities[:, :, 0, None]),
    #     axis=-1,
    # )

    # ctc_decode, _ = tf.nn.ctc_greedy_decoder(
    #     inputs=gloss_probabilities,
    #     sequence_length=np.array(sgn_lengths),
    #     blank_index=SI_IDX,
    #     merge_repeated = False
    # )
    ctc_decode, _ = tf.nn.ctc_beam_search_decoder(
                        inputs=gloss_probabilities,
                        sequence_length=np.array(sgn_lengths),
                        beam_width=5,
                        top_paths=1,
                        )
    ctc_decode = ctc_decode[0]
    # Create a decoded gloss list for each sample
    tmp_gloss_sequences = [[] for i in range(gloss_probabilities.shape[1])]
    for (value_idx, dense_idx) in enumerate(ctc_decode.indices):
        if ctc_decode.values[value_idx].numpy() != SI_IDX:
            tmp_gloss_sequences[dense_idx[0]].append(
                ctc_decode.values[value_idx].numpy()
            )
    
    decoded_gloss_sequences = []
    for seq_idx in range(0, len(tmp_gloss_sequences)):
        decoded_gloss_sequences.append(
            [x[0] for x in groupby(tmp_gloss_sequences[seq_idx])]
        )
    return decoded_gloss_sequences



def visualization(atten_maps):
    os.makedirs('./demo', exist_ok=True)
    for ii, att in enumerate(atten_maps):
        i = att.shape[0]
        idx = [max(1, int((i**0.5))), i//max(1, int((i**0.5))), 1]

        fig = plt.figure(figsize=(6*idx[1], 6*idx[0]))
        if att.squeeze().dim() == 2:
            ax = fig.add_subplot()
            att = torch.softmax(att, dim=-1)
            sns.heatmap(att.detach().cpu().numpy(), annot=False, yticklabels=False, xticklabels=False, fmt='g', ax=ax)
            
            fig.savefig(os.path.join('./demo', f'Att_score_{ii}.jpg'), dpi=fig.dpi)
            plt.close()
            continue
        
        for cmp in att:
            ax = fig.add_subplot(*idx)
            sns.heatmap(cmp.detach().cpu().numpy(), cbar=idx[-1] % idx[-2] == 0, annot=False, yticklabels=False, xticklabels=False, fmt='g', ax=ax)
            idx[-1] += 1
        fig.savefig(os.path.join('./demo', f'Att_score_{ii}.jpg'), dpi=fig.dpi)
        plt.close()

class KLLoss(torch.nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=torch.nn.KLDivLoss(reduction="batchmean")):
        super().__init__()
        print('=========using KL Loss=and has temperature and * bz==========')
        self.error_metric = error_metric

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss
        
def loss_fn_kd(outputs, teacher_outputs, T=1.0, alpha=0.5):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    KD_loss = torch.nn.KLDivLoss( reduction='sum')(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (T * T) #+ \
            #    F.cross_entropy(outputs, F.softmax(teacher_outputs, dim=1)) * (1. - alpha)

    return KD_loss

def unitwise_norm(x, norm_type=2.0):
    if x.ndim <= 1:
        return x.norm(norm_type)
    else:
      
        return x.norm(norm_type, dim=tuple(range(1, x.ndim)), keepdim=True)


class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler()

    def __call__(self, loss, optimizer, epoch,
                 clip_grad=None, 
                 clip_mode='norm', 
                 parameters=None, 
                 create_graph=False, 
                 logger=None):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if clip_grad is not None:
            assert parameters is not None
            self._scaler.unscale_(optimizer) 
            if logger is not None:
                logger.log_gradients(step=epoch)
            self.dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
        self._scaler.step(optimizer)
        self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
        
    def adaptive_clip_grad(self, parameters, 
                           clip_factor=0.01, 
                           eps=1e-3, norm_type=2.0):
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        for p in parameters:
            if p.grad is None:
                continue
            p_data = p.detach()
            g_data = p.grad.detach()
            max_norm = unitwise_norm(p_data, norm_type=norm_type).clamp_(min=eps).mul_(clip_factor)
            grad_norm = unitwise_norm(g_data, norm_type=norm_type)
            clipped_grad = g_data * (max_norm / grad_norm.clamp(min=1e-6))
            new_grads = torch.where(grad_norm < max_norm, g_data, clipped_grad)
            p.grad.detach().copy_(new_grads)

    def dispatch_clip_grad(self, parameters, value, 
                           mode= 'norm', 
                           norm_type = 2.0):
    
        if mode == 'norm':
            torch.nn.utils.clip_grad_norm_(parameters, value, norm_type=norm_type)
        elif mode == 'value':
            torch.nn.utils.clip_grad_value_(parameters, value)
        elif mode == 'agc':
            self.adaptive_clip_grad(parameters, value, norm_type=norm_type)
        else:
            assert False, f"Unknown clip mode ({mode})."

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False