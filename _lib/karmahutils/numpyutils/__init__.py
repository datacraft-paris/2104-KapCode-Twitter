import numpy as np
import pandas as pd

version = '1.3.2'


def nd_sample(nd_array, n=1, ratio=0):
    """A simple copy of the pandas sample() meant for 2D arrays.
    n is the number of rows returned , default is 1.
    ratio is a % of the original data
    providing a ratio != 0 will supersede any n value."""
    mask = np.random.choice([True, False], len(nd_array), p=[ratio, 1 - ratio]) if ratio \
        else np.random.choice(len(nd_array), n, replace=False)
    return nd_array[mask]


class KarmahNdArray:
    def __init__(self, nd_array):
        self.nd_array = nd_array
        self.dataframe = None

    def __repr__(self):
        return str(self.nd_array)

    def sample(self, n=1, ratio=0):
        """A simple copy of the pandas sample() meant for 2D arrays.
            n is the number of rows returned , default is 1.
            ratio is a % of the original data
            providing a ratio != 0 will supersede any n value."""
        return nd_sample(self.nd_array, n=n, ratio=ratio)

    def get_dataframe(self, columns=None, store=False):
        df = pd.DataFrame(self.nd_array, columns=columns)
        if store:
            self.dataframe = df
        return df


print('loaded karmah numpy-utils version', version)
