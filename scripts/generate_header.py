#!/usr/bin/python3
import numpy as np

def write_data(f, data, var_name: str):
    dtype = 'char' if data.dtype == np.int8 else 'int32_t'
    f.write(f"static {dtype} {var_name}[{len(data)}] = {{")

    # Write as 1D array (sequentially)
    for i in range(len(data)):
        if i != 0:
            f.write(",")
        if i%32 == 0:
            f.write("\n")
        f.write("%4d" % (data[i]))

    f.write("};\n\n")

# main function
def main():
    # Create header file
    print(f"Writing input, weight and bias into header.h ...")
    input = np.fromfile('data/input.bin', dtype=np.int8)
    weight = np.fromfile('data/weight.bin', dtype=np.int8)
    bias = np.fromfile('data/bias.bin', dtype=np.int32)

    with open('header.h', 'w') as f:
        write_data(f, input, 'input')
        write_data(f, weight, 'weight')
        write_data(f, bias, 'bias')


# Entry point
if __name__ == '__main__':
    main()
