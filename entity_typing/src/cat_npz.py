import os
import sys
import scipy
import numpy as np
from scipy.sparse import coo_matrix, vstack

print("Usage: input_path output_path")

src_path = sys.argv[1]
dst_path = sys.argv[2]

e2r = []
for i in range(18):
    e2r.append(scipy.sparse.load_npz(os.path.join(src_path, 'e2r_scores_%d.npz' % i)))
    print(e2r[-1].shape)
e2r = vstack(e2r)

scipy.sparse.save_npz(os.path.join(dst_path, "e2r_scores.npz"), e2r)
