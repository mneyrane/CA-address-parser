"""
Synthesize real world test data .

The CSV file loaded by Pandas in this script can be exported from

https://data.winnipeg.ca/City-Planning/Addresses/cam2-ii3u

This is the city of Winnipeg's civic address, published on their
open data portal.
"""

import pandas as pd
from address.utils import normalize_text
from address.proc_gen import join_address_tokens

token_cols = [
    'Street Number', 
    'Street Number Suffix', 
    'Street Name', 
    'Street Type',
    'Street Direction', 
    'Unit Type', 
    'Unit Number',
]

extra_cols = [
    'Full Address',
]

cols = token_cols + extra_cols

# load data
df = pd.read_csv('mb-winnipeg.csv', usecols=cols, dtype='string')

norm_text_wrapper = lambda s : s if pd.isna(s) else normalize_text(s)
df = df.applymap(norm_text_wrapper)

# test that combination is valid
df['Combined'] = df[token_cols].apply(
    lambda x : x.str.cat(sep=' '), axis='columns')

assert (df['Combined'] == df['Full Address']).all()

# define data retrieval
def define_data(x):
    data = {
        'hn' : x[['Street Number', 'Street Number Suffix']].str.cat(sep=' '),
        'sn' : x['Street Name'],
        'st' : x['Street Type'],
        'ds' : x['Street Direction'],
        'un' : x[['Unit Type', 'Unit Number']].str.cat(sep=' '),
    }
    
    template = ['hn','sn','st','ds','un']

    tokens = {k : data[k] for k in data if pd.notna(data[k])}
    
    address, clf = join_address_tokens(tokens, template, strict=False)

    if address[-1] == ' ':
        address.pop()
        clf.pop()

    clf_chars = ['{}'.format(e.value) for e in clf]

    input_seq = ''.join(address)
    output_seq = ' '.join(clf_chars)

    return input_seq, output_seq

# create test data
n_samples = 56320 # divisible by 128 and 256; about 25% of the original data
seed = 26

df_sample = df[token_cols].sample(n=n_samples, random_state=seed)

test_data = df_sample.apply(
    define_data, axis='columns', result_type='expand')

test_data.to_csv(
    'datasets/test_sequences_real.csv.gz', header=False, index=False, sep='|')
