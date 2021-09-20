import h5py as h5
import numpy as np
from numpy.lib.function_base import percentile, piecewise
import torch
import time
start = time.time()

momentum_info=np.array([])
for file_ord in range(10,11):
    with h5.File("./final-{}.h5".format(file_ord), "r") as ipt:
        # 读取particletruth作为标签
        b = ipt['ParticleTruth']

        momentum_info = np.concatenate((momentum_info,np.array([truth[4] for truth in b ])))
    print("file:",file_ord)
torch.save(torch.tensor(momentum_info),"./event/real_momentum")
end=time.time()
print("Running time %s seconds"%(end-start))
