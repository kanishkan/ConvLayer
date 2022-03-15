import sys
import numpy as np

if __name__ == "__main__":
    dtype = np.int8
    ref = np.fromfile(sys.argv[1], dtype=dtype)
    out = np.fromfile(sys.argv[2], dtype=dtype)

    print(f"Comparing TTA output with reference output:")
    print(f"\tdifference (mean): {(ref-out).mean()}. Array equal: {np.array_equal(ref, out)}")
