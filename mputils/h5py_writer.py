import numpy as np
import h5py

def write_file(output_file, data, storage='h5'):
    with h5py.File(output_file, 'w') as h5_file:
        h5_file.create_dataset('data', 
            data=data,
            dtype='f'
        )

