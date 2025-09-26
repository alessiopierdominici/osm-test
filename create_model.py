import clip
import torch.nn as nn
from utils.general_utils import get_preprocess, get_remoteclip_transform
import torch
from clip.model import Transformer, convert_weights
from typing import Union, Tuple
import numpy as np
from clip.clip import _MODELS, _download, available_models
import os

IMAGE_SIZE = 224

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class ModifiedVisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, return_cls=True):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        self.return_cls = return_cls
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.return_cls:
            x = self.ln_post(x[:, 0, :])
        else:
            x = self.ln_post(x[:, 1:, :])
        
        if self.proj is not None:
            x = x @ self.proj

        return x
class OSM_CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 return_cls:bool = True,
                 device = "cuda:0"
                 ):
        super().__init__()

        self.context_length = context_length


        vision_heads = vision_width // 64
        self.visual = ModifiedVisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            return_cls=return_cls
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()
        
        self.preprocess = get_preprocess(IMAGE_SIZE)
        self.device = device

    def set_return_cls(self, return_cls:bool):
        '''
        Set the return_cls parameter of the visual transformer. This is used to return the CLS token or all the patches.
        '''
        self.visual.return_cls = return_cls

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, images:torch.Tensor, unique_tags_tensor:torch.Tensor):
        '''
        Encode the images and return the last layer embeddings.
        Encode the unique_tags and return the last layer embeddings.
        '''
        # Forward to the model and have the activations
        image_features = self.encode_image(images)
        text_features = self.encode_text(unique_tags_tensor)
        
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        return image_features, text_features
    
    def _freeze_text(self):
        for name, param in self.clip_model.named_parameters():
            if "transformer" in name and not "visual" in name:
                param.requires_grad = False
            if "positional_embedding" in name and not "visual" in name: # Freeze only the text positional embeddings
                param.requires_grad = False
            if "text_projection" in name:
                param.requires_grad = False
            if "token_embedding" in name:
                param.requires_grad = False
            if "ln_final" in name:
                param.requires_grad = False

def _load_osm_clip(USE_AUGMENTATION: bool, name, device, return_cls=True, download_root: str = None):
    '''
    Load the slightly modified structure of OSM CLIP. The return_cls parameter change the behavior in the forward method. If return_cls=False, it returns the embeddings of every single patch.
    '''
    # if model_name in AVAILABLE_MODELS:
    #     # Load state dict
    #     state_dict = torch.load(f"original_clip_models/"+model_name+".pth", map_location="cpu")
    # else:
    #     raise RuntimeError(f"Model {model_name} not found; available models = {AVAILABLE_MODELS}")
    original = False
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
        original = True
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}, nor found in the os path.")

    state_dict = None
    if original:
        tmp_model = clip.load(name, USE_AUGMENTATION=USE_AUGMENTATION)
        state_dict = tmp_model[0].state_dict()
    else:
        state_dict = torch.load(model_path, map_location="cpu")

    # Infer the values (copied from CLIP source code)
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    # Load the model
    model = OSM_CLIP(embed_dim=embed_dim,
                 # vision
                 image_resolution = image_resolution,
                 vision_layers = vision_layers,
                 vision_width = vision_width,
                 vision_patch_size = vision_patch_size,
                 # text
                 context_length = context_length,
                 vocab_size = vocab_size,
                 transformer_width = transformer_width,
                 transformer_heads = transformer_heads,
                 transformer_layers = transformer_layers,
                 return_cls = return_cls,
                 device = device)

    # Load state dict
    convert_weights(model)
    # Load checkpoint here
    # state_dict = torch.load(f"{CHECKPOINT_PATH}")
    model.load_state_dict(state_dict)
    model = model.to(device)

    if USE_AUGMENTATION:
        return model, get_remoteclip_transform(IMAGE_SIZE)
    else:
        return model, get_preprocess(IMAGE_SIZE)
