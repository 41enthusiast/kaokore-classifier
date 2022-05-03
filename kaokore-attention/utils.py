import torch


#drops images in a batch from being seen by a subset of layers
def drop_connect(inputs, p, train_mode):  # bchw, 0-1, bool
    assert 0 <= p <= 1, 'p must be in range of [0,1]'

    #inference
    if not train_mode:
        return inputs

    bsz = inputs.shape[0]

    keep_prob = 1-p

    #binary mask for selection of weights
    rand_tensor = keep_prob
    rand_tensor += torch.rand([bsz, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    mask = torch.floor(rand_tensor)

    outputs = inputs / keep_prob*mask
    return outputs

