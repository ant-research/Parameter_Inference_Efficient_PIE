import numpy as np
import sys

input_prefix=sys.argv[1]
output=sys.argv[2]

e2r=[]
for i in range(15):
  e2r.append(np.load(input_prefix + '_%d.npy' % i)[:, 3:])
  print(e2r[-1].shape)
e2r = np.concatenate(e2r, axis=0)
print(e2r.shape)

np.save(output, e2r)
