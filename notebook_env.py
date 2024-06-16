# Built-in
import collections
import functools
import itertools
import os
import re
import time

# 3rd-party
import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm.auto import tqdm

# matplotlib setup
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath,amssymb}',
})
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('retina')

# Paper-specific
import statcontracts
datasets_directory = os.path.expanduser('~/Documents/data/llm_contracts')
