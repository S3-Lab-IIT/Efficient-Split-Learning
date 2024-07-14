import torch
import copy

def merge_grads(normalized_data_sizes, params):
    # params = [params_client1,
    #           params_client2,
    #           params_client3
    #           ...
    #          ]
    num_clients = len(params)
    for j,col in enumerate(zip(*params)):
        avg = 0
        for i,param in enumerate(col):
            print('param:',param,param.grad,type(param.grad))
            avg += normalized_data_sizes[i] * param.grad
            # avg += param.grad

        # avg /= num_clients  # (since we are already doing weighted adding of gradients)
        for param in col:
            param.grad = copy.deepcopy(avg)
            # print("is para grad equal to average?", param.grad)

    return



def merge_weights_unweighted(w,lens):
    #after step op, merge weights 

    total_samples = sum(lens)

    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    
    return w_avg

def merge_weights_old(w, lens):
    total_samples = sum(lens)
    num_models = len(w)

    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] *= lens[0]
        for i in range(1, num_models):
            w_avg[k] += w[i][k] * lens[i]
        w_avg[k] /= total_samples

    return w_avg


import torch

def merge_weights(sds, lens):
    num_models = len(sds)
    assert num_models == len(lens), "Number of state dicts and lens values should match"

    merged_sd = {}

    # Iterate over each state dict
    for key in sds[0].keys():
        # Get the device and data type of the first model's weights
        device = sds[0][key].device
        dtype = sds[0][key].dtype

        # Initialize merged weights tensor with zeros on the same device and data type
        merged_weights = torch.zeros_like(sds[0][key], device=device, dtype=dtype)

        # Iterate over each model's state dict and calculate weighted sum
        for i in range(num_models):
            model_weights = sds[i][key].to(device=device, dtype=dtype)
            merged_weights += model_weights * lens[i]

        # Calculate weighted average
        total_lens = sum(lens)
        if dtype == torch.float32 or dtype == torch.float64:
            merged_weights /= total_lens
        else:
            merged_weights = (merged_weights.float() / total_lens).to(dtype)

        # Update merged state dict with weighted average weights
        merged_sd[key] = merged_weights

    return merged_sd


