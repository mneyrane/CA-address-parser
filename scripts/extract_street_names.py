"""
Extract street names from Ottawa's civic addresses.

The addresses are open data that can be exported to CSV from here:

https://open.ottawa.ca/datasets/municipal-address-points
"""

from itertools import product

import pandas as pd

from address.utils import normalize_text

csvfile = 'on-ottawa.csv'
stname = 'ROAD_NAME'

df = pd.read_csv(csvfile, usecols=[stname], dtype='string')

# extract unique non-numeric street names
street_names = pd.Series(df[stname].unique())
mask = street_names.str.contains('\d', regex=True)
street_names = street_names[~mask]

street_names = street_names.apply(normalize_text)

# add numeric names
numerics = [str(i) for i in range(1, 300)]
suffixes = ['', 'A', 'B', 'C']

num_names = [''.join(s) for s in product(numerics, suffixes)]
num_names = pd.Series(num_names)

# concatenate arrays, process and save to text file
data = pd.concat([num_names, street_names])
data = data.sort_values()
data = data.reset_index(drop=True)

data.to_csv('street_names.txt', index=False)
