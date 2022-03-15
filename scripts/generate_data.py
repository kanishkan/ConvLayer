import random
from functools import partial

import numpy as np
import torch
from torch.nn import functional as nn_F
from torch.nn.qat.modules.conv import Conv2d as QConv2d
from torch.quantization import QConfig

# Seed
seed = 1234
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# INT8 Quantization
quant_min = -2 ** (8 - 1)
quant_max = 2 ** (8 - 1) - 1
fake_quantize = torch.quantization.fake_quantize.FakeQuantize.with_args(
    observer=partial(torch.quantization.observer.MinMaxObserver, dtype=torch.qint8,
                     quant_min=quant_min, quant_max=quant_max, qscheme=torch.per_tensor_symmetric),
    quant_min=quant_min, quant_max=quant_max
)
qconfig = QConfig(activation=fake_quantize, weight=fake_quantize)

# Helpers
def ndarray_to_file(ndarray, dtype, filename: str):
    filepath = f"../data/{filename}"
    ndarray.astype(dtype).tofile(filepath)

def tensor_to_ndarray(tensor):
    return tensor.cpu().numpy()

def tensor_to_file(tensor, dtype, filename: str):
    ndarray_to_file(tensor_to_ndarray(tensor), dtype, filename)

def tile_dim(tensor, factor, dim=0):
    new_shape = list(tensor.shape)
    new_shape.insert(dim, new_shape[dim] // factor)
    new_shape[dim+1] = factor
    tensor = tensor.reshape(new_shape)
    return tensor

def dtype_to_minmax(dtype, symmetric=True):
    num_bits = 32
    if dtype is np.int8:
        num_bits = 8
    elif dtype is np.int16:
        num_bits = 16
    if symmetric:
        quant_min = -2 ** (num_bits - 1)
        quant_max = 2 ** (num_bits - 1) - 1
    else:
        raise NotImplementedError
    return quant_min, quant_max

def to_int(x, scale, quant_min, quant_max):  # Quantization helper
    return torch.clamp(torch.round(x / scale), quant_min, quant_max)


# Data formatting
i8_weight_permutation = (0, 2, 4, 5, 1, 3) # MNRS-> MmNRS -> MmNnRS -> MNRSmn
dw_weight_permutation = (0, 2, 3, 4, 1) # MNRS-> MmNRS -> MNRSm
fm_permutation = (0, 1, 3, 4, 2) # NCHW -> NCcHW -> NCHWc

# Convolution
M, C, W, H, R, S = 128, 128, 20, 20, 3, 3
conv = QConv2d(in_channels=C, out_channels=M, kernel_size=(R, S), padding=1, bias=True, qconfig=qconfig)
# FakeQuantize input
input_fake_quant = qconfig.activation()
random_input = input_fake_quant(torch.randn((1,C,H,W)).detach())
# Execute FakeQuantized convolution
output = nn_F.relu(conv(random_input))


# Quantization scale factor as a power of two
wa_scale = input_fake_quant.scale*conv.weight_fake_quant.scale
wa_scale_pot = 2**np.round(np.log2(wa_scale))
# Real quantization
int_input = to_int(random_input, input_fake_quant.scale, *dtype_to_minmax(np.int8))
int_weight = to_int(conv.weight.detach(), conv.weight_fake_quant.scale, *dtype_to_minmax(np.int8))
int_bias = to_int(conv.bias.detach(), wa_scale, *dtype_to_minmax(np.int32))
# Execute quantized convolution
int32_output = nn_F.relu(nn_F.conv2d(input=int_input, weight=int_weight, padding=1) + int_bias.reshape([1, -1, 1, 1])).detach()


print(f'')
print(f'Mean difference between fp32 and int8 conv with int32 output: {torch.mean(int32_output*wa_scale_pot - output)}')

int32_output_max = torch.max(-torch.min(int32_output), torch.max(int32_output))
scale_pot_part1 = 2**np.round(np.log2(2*int32_output_max / 255))
scale_pot_part2 = np.log2(wa_scale_pot).abs()-np.log2(scale_pot_part1)
print(f'WA scale pot shift: {np.log2(wa_scale_pot)}. Scale pot p1: {np.log2(scale_pot_part1)}. Scale pot p2: {scale_pot_part2}')
int8_output = torch.bitwise_right_shift(int32_output, np.log2(scale_pot_part1))
print(f'Mean difference between fp32 and int8 conv with int8 output: {torch.mean(int8_output*2**-scale_pot_part2 - output)}')

# Save to file
tensor_to_file(nn_F.pad(int_input, pad=[1] * 4, value=0), dtype=np.int8, filename='input.bin')
tensor_to_file(int_weight, dtype=np.int8, filename='weight.bin')
tensor_to_file(int_bias, dtype=np.int32, filename='bias.bin')
tensor_to_file(int32_output, dtype=np.int32, filename='output_int32.bin')
tensor_to_file(int8_output, dtype=np.int8, filename='output.bin')