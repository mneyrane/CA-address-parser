# Canadian address parser

Mockup of a character-based Canadian address parser, based on training a neural network on data synthesized by a context-free grammar.
A short technical report of the implementation is detailed on my [website](https://mneyrane.com/projects/addressparser), which I encourage to read.

*NOTE: this work is experimental, and the model provided in question is not suitable for production!*

## Motivation

The key motivation of this work is purely hobbyist.
It is a voluntary exercise, on my part, to assemble data and a machine learning model to tackle a nontrivial natural language task.
It also extends and improves upon my past class (statistical learning) project, where back then, my machine learning background was in its infancy.

The choice of address parsing is inspired by my past work at Statistics Canada.
At the time, I was responsible to help assemble public Canadian infrastructure datasets from open sources.
It was desired to tokenize addresses into a house number, street name, street type, etc., and many datasets we encountered had unsplit address strings.

## Requirements

All the scripts provided are written in Python and have been ran using **Python 3.11.3**.
The key packages used are:

| Package | Version |
| ------- | ------- |
| `numpy` | 1.24.3 |
| `pandas` | 2.0.2 | 
| `unidecode` | 1.3.6 |
| `torch` | 2.0.1 |
| `matplotlib` | 3.7.1 |
| `seaborn` | 0.12.2 |
| `scikit-learn` | 1.2.2 |

For convenience, I have included `requirements.txt` to install the above packages via `pip`.

It may be possible to run the scripts with older versions of these packages or Python, but I have not attempted to test this.

## Code organization and running scripts

*NOTE that to run any of the main scripts, you must do so from the repository root!*
Any results produced by scripts, such as figures or model checkpoints, are self-contained and will appear in folders created in the repository root.

### Folders

- `address/` : Tools to randomly generate addresses, augment text data with typos, and other text processing utilities.
- `datasets/` : Used to randomly generate and save address data. For more information on the data appearing therein, read [here](datasets/README.md).
- `scripts/` : Helper scripts to extract outside data. These are not part of the workflow and can be ignored.

### Main scripts

These are presented in order of execution if starting from scratch.

- `generate_data.py` : Randomly generate addresses.[^1]
- `model.py` : Neural network model definition, termed CCAPNet (**C**anadian **C**ivic **A**ddress **P**arser neural **net**work).
- `train.py` : Train the model. Has command line arguments that can be viewed with `-h` or `--help`.
- `inference.py` : Use the model for inference on input text. Has command line arguments that can be viewed with `-h` or `--help`.
- `plot_*.py` : Create figures for metrics and performance. Accepts a specific input from the model folder produced by `train.py`.

[^1]: Takes about 65 seconds to run on a Intel i7-7700K CPU to generate 128000 training points 16000 test points.

## Attributions and related work

I attribute using

- a *character-based* address parser
- *typos* to regularize training

to Jason Rigby's [AddressNet](https://towardsdatascience.com/addressnet-how-to-build-a-robust-street-address-parser-using-a-recurrent-neural-network-518d97b9aebd) project.
Thank you Jason!

If you are interested in address parsing or working with address data, I strongly recommend the following resources:

- [libpostal](https://github.com/openvenues/libpostal): state-of-the-art international address normalizer and parser
- [OpenAddresses](https://github.com/openaddresses/openaddresses): open global address data
