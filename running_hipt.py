"""
Here, we simply test the HIPT_4K model on the demo image.
"""

import sys
from PIL import Image

sys.path.extend([
    './1-Hierarchical-Pretraining/', 'HIPT_4K'
])

from HIPT_4K.hipt_4k import HIPT_4K
from HIPT_4K.hipt_model_utils import eval_transforms

model = HIPT_4K()
model.eval()

region = Image.open('HIPT_4K/image_demo/image_4k.png')
x = eval_transforms()(region).unsqueeze(dim=0)
out = model.forward(x)

