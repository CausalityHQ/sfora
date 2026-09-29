import argparse
from pathlib import Path
from unittest.mock import patch
import torch
import train_pe_large_coverage as driver
parser=argparse.ArgumentParser()
parser.add_argument('execution_sha')
args=parser.parse_args()
assert not torch.cuda.is_available()
root=Path(driver.__file__).resolve().parent
driver.startup(root,args.execution_sha)
original=driver.pair.sha
for target in (root/'train_pe_large_coverage.py',driver.CPU/'receipt.json'):
    with patch.object(driver.pair,'sha',lambda path:'altered' if Path(path)==target else original(path)):
        try:
            driver.startup(root,args.execution_sha)
        except AssertionError:
            pass
        else:
            raise AssertionError('altered training code/CPU authority accepted')
print('PASS actual87-source training startup and altered driver/CPU-proof rejection; no model/GPU/optimizer')
