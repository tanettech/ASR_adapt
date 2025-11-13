#data/dataset.py

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from datasets import load_dataset, Dataset, Audio, DatasetDict
import json
import logging
from typing import Dict, Tuple, List, Any, Optional, Union
from datasets import load_dataset, Dataset, Audio, DatasetDict
from tqdm import tqdm

print(f'This is the initial test')


if __name__ == "__main__":
    print("Dataset module is running directly")

