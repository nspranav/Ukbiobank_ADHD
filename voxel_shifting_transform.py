import nibabel as nib
import pandas as pd
import numpy as np
import torch

'''

'''

class VoxelShift(object):

    def __init__(self,img : torch.Tensor) -> None:
        self.img = img

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)