from turtle import forward
import torch
import torch.nn as nn
from typing import Optional, Tuple

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

class SigLipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernels=self.patch_size,
            stride=self.patch_size,
            padding="valid" # no padding
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arrange(self.num_positions).expand(-1, 1),
            persistent=False,
        )
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape

        patch_embeds = self.patch_embedding(pixel_values)

        embeddings = patch_embeds.flatten(2)

        embeddings = embeddings.transpose(1, 2)

        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings

class SigLipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)

        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")

        hidden_states = self.fc2(hidden_states)

        return hidden_states

class SigLipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** (-0.5)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # hidden_states: [Batch_size, Num_patches, Embed_dim]    
        batch_size, seq_len, _ = hidden_states.size()

        # query_states: [Batch_size, Num_patches, Embed_dim]
        query_states = self.q_proj(hidden_states)

        # key_states: [Batch_size, Num_patches, Embed_dim]
        key_states = self.k_proj(hidden_states)

        # value_states: [Batch_size, Num_patches, Embed_dim]
        value_states = self.v_proj(hidden_states)

        # query_states: [Batch_size, Num_patches, Num_heads, Head_dim]
        # then we convert it to 
        # query_states: [Batch_size, Num_heads, Num_patches, Head_dim]
        # so, first we split Embed_dim into Num_heads by Head_dim
        # then we convert it into the above config via transpose
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # we do the same for k and v states as well
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # attn_weights: [Batch_size, Num_heads, Num_patches, Num_patches]
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is of"
                f"{attn_weights.size()}"
            )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is of"
                f"{attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

class SigLipVisionEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SigLipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)

        hidden_states, _ = self.self_attn(hidden_states=hidden_states)

        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.layer_norm2(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states

class SigLipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SigLipVisionEmbeddings(config)
        self.enoder = SigLipVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.embeddings(pixel_values)

        last_hidden_state = self.encoder(input_embeds=hidden_states)

        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state

class SigLipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SigLipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values)