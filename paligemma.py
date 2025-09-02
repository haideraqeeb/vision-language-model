from typing import Dict, List, Tuple, Optional, Union, Iterable
import numpy as np
from PIL import Image
import torch

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

class PaliGemmaProcessor:

    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_images_tokens: int, image_size: int):
        super().__init__()

        self.image_seq_length = num_images_tokens
        self.image_size = image_size

        # Tokenizer described here: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)

        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ] # these tokens are used for object detection (bounding box)

        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ] # these tokens are used for object segmentation

        tokenizer.add_tokens(EXTRA_TOKENS)
        
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        # we add BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer