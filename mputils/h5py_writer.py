import numpy as np
import h5py

def write_file(output_file, data, storage='h5', left_handed=None):
    with h5py.File(output_file, 'w') as h5_file:
        dset = h5_file.create_dataset('data', 
            data=data,
            dtype='f'
        )
        
        dset.attrs['left_handed'] = left_handed
