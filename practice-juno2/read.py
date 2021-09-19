import h5py as h5
import numpy as np

with h5.File("final-2.h5", "r") as ipt:
    a = ipt['Waveform']
    f = np.array([[np.max(wave[2]),np.min(wave[2]),np.mean(wave[2]),np.std(wave[2])] for wave in a])
    np.savetxt('data2.txt',f)