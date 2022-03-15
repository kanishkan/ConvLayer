import numpy as np


def tile_dim(ndarray, factor, dim=0):
    new_shape = list(ndarray.shape)
    new_shape.insert(dim, new_shape[dim] // factor)
    new_shape[dim+1] = factor
    ndarray = ndarray.reshape(new_shape)
    return ndarray


def main():
    # Settings and load
    M, C, W, H, R, S = 128, 128, 22, 22, 3, 3
    input = np.fromfile('../data/input.bin', dtype=np.int8).reshape(C,H,W)
    weight = np.fromfile('../data/weight.bin', dtype=np.int8).reshape(M,C,R,S)
    output = np.fromfile('../data/output.bin', dtype=np.int8).reshape(M,H-R+1,W-S+1)

    # Change below. Keep input/output tiled in similar fashion.
    fm_permutation = (0, 2, 3, 1) # CHW -> CcHW -> CHWc
    weight_permutation = (0, 1, 3, 4, 2) # MCRS -> MCcRS -> MCRSc
    tiled_input = tile_dim(input, factor=4, dim=0).transpose(fm_permutation)
    tiled_output = tile_dim(output, factor=4, dim=0).transpose(fm_permutation)
    # tiled_output = output
    tiled_weight = tile_dim(weight, factor=4, dim=1).transpose(weight_permutation)

    # Save to file
    tiled_input.tofile(f"../data/input_tiled.bin")
    tiled_output.tofile(f"../data/output_tiled.bin")
    tiled_weight.tofile(f"../data/weight_tiled.bin")


# Entry point
if __name__ == '__main__':
    main()
