import csv
import gzip
import json
import time

import numpy as np
import pandas as pd

from address.proc_gen import TokenCategory as TC
import address.proc_gen as pg


# constants and functions
SEED = 128
NUM_TRAIN_PTS = 192000
NUM_TEST_PTS = 32000

def create_input_output_pair(genmap, tmpls, rng):
    idx = rng.choice(len(tmpls))
    tokens = pg.sample_address_tokens(genmap, tmpls[idx])
    address, clf = pg.join_address_tokens(tokens, tmpls[idx])
    clf_chars = ['{}'.format(e.value) for e in clf]
    
    input_seq = ''.join(address)
    output_seq = ' '.join(clf_chars)
    
    return input_seq, output_seq

# define random number generator
rng = np.random.default_rng(SEED)

#
# define generators
#

# unit numbers
df_unit = pd.read_csv(
    'datasets/unit_designation.txt', names=['values'], dtype='string', na_filter=False)

g_unit = pg.AlphanumGenerator(
    nmin=1, nmax=10000, 
    name='unit', 
    rng=rng,
    desig=df_unit['values'],
)

# house numbers
g_house_num = pg.AlphanumGenerator(
    nmin=1, nmax=50000,
    name='house_num',
    rng=rng,
)

# street names
df_st_name = pd.read_csv(
    'datasets/street_names.txt', names=['values'], dtype='string')

g_st_name = pg.UniformListSampler(
    name='st_name',
    rng=rng,
    values=df_st_name['values'],
)

# street type
df_st_type = pd.read_csv(
    'datasets/street_types.txt', names=['values'], dtype='string')

g_st_type = pg.UniformListSampler(
    name='st_type',
    rng=rng,
    values=df_st_type['values'],
)

# direction
df_dir = pd.read_csv(
    'datasets/directions.txt', names=['values'], dtype='string')
    
g_dir = pg.UniformListSampler(
    name='dir',
    rng=rng,
    values=df_dir['values'],
)

# city
df_city = pd.read_csv(
    'datasets/cities.txt', names=['values'], dtype='string')
    
g_city = pg.UniformListSampler(
    name='city',
    rng=rng,
    values=df_city['values'],
)

# province or territory
df_prov = pd.read_csv(
    'datasets/provinces.txt', names=['values'], dtype='string')
    
g_prov = pg.UniformListSampler(
    name='prov',
    rng=rng,
    values=df_prov['values'],
)

# postal code
g_postcode = pg.PostalCodeGenerator(
    name='postcode',
    sep_prob=.5,
    rng=rng,
)

#
# prepare procedural generation
#

# load address templates
with open('datasets/templates.json', 'r') as jsonfile:
    templates = json.load(jsonfile)

# define template-string-to-generator mapping
generator_map = {
    'un' : g_unit,
    'hn' : g_house_num,
    'sn' : g_st_name,
    'st' : g_st_type,
    'dp' : g_dir,
    'ds' : g_dir,
    'ci' : g_city,
    'pr' : g_prov,
    'po' : g_postcode,
}

#
# generate and save data
#

train_data_path = 'datasets/train_sequences_pg.csv.gz' 
test_data_path = 'datasets/test_sequences_pg.csv.gz'

# training data
with gzip.open(train_data_path, 'wt', newline='') as trainfile:
    for _ in range(NUM_TRAIN_PTS):
        writer = csv.writer(trainfile, delimiter='|')
        pair = create_input_output_pair(generator_map, templates, rng)
        writer.writerow(pair)

print(f"Saved {NUM_TRAIN_PTS} training points to '{train_data_path}'.")
        
# synthetic test data
with gzip.open(test_data_path, 'wt', newline='') as testfile:
    for _ in range(NUM_TEST_PTS):
        writer = csv.writer(testfile, delimiter='|')
        pair = create_input_output_pair(generator_map, templates, rng)
        writer.writerow(pair)

print(f"Saved {NUM_TEST_PTS} test points to '{test_data_path}'.")
