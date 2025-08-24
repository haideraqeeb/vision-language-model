import torch
import torch.nn as nn

class SiglipVisionConfig:

# config class made since PaliGemma comes in different sizes
    def __init__(
        self,
        hidden_size=768, # size of embedding vector
        intermediate_size=3072, # size of linear layer
        num_hidden_layers=12, # num of layers in vision transformer model
        num_attention_heads=12, # num of attention heads
        num_channels=3, # num channels, RGB
        image_size=224, # size of image
        patch_size=16, # size of patches, hence num patches is 14
        layer_norm_eps=1e-6, # layer normalization hyper param
        attention_dropout=0.0, # attention dropout value
        num_image_tokens: int = None, # num of output embeddings the vision transformer will output
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens