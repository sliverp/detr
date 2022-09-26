import math
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.nn import MultiheadAttention

from vit_pytorch import ViT

from models.backbone import Backbone, FrozenBatchNorm2d, build_backbone
from util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class BaseSelfAttention(nn.Module):
    def __init__(self, num_heads: int = 8, embed_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attention = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos


class BoxSelfAttention(BaseSelfAttention):
    def forward(self, box, queries):
        """

        Args:
            box (_type_): box 的特征
            queries (_type_): Object queries 用来残差连接
        """
        q = k = self.with_pos_embed(box, queries)
        attened = self.self_attention(
            query=q,
            key=k,
            value=box
        )

        box = box + self.dropout(attened)
        box = self.norm(box)
        return box
        

class FeatureSelfAttention(BaseSelfAttention):

    def __init__(
        self, 
        num_heads: int = 8, 
        embed_dim: int = 256, 
        dropout: float = 0.1,
        dim_feedforward:int = 2048
    ) -> None:
        super().__init__(num_heads, embed_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.activation = F.relu

    def forward(self, feature, pos):
        q = k = self.with_pos_embed(feature, pos)
        attened = self.self_attention(
            query=q,
            key=k,
            value=feature
        )
        feature = feature + self.dropout1(attened)
        feature = self.norm(feature)
        feature2 = self.linear2(self.dropout(self.activation(self.linear1(feature))))
        feature = feature + self.dropout2(feature2)
        return self.norm1(feature)

class EncoderLayer(nn.Module):
    def __init__(self, num_heads: int = 8, embed_dim: int = 256, dropout:float = 0.1) -> None:
        super().__init__()
        self.box_self_attention = BoxSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.feature_self_attention = FeatureSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.cross_attention = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)


    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, box, queries, feature, pos):
        q = queries + self.box_self_attention(
            box=box,
            queries=queries
        )
        v = self.feature_self_attention(
            feature=feature,
            pos=pos
        )
        k = self.with_pos_embed(v, pos)
        atten = self.cross_attention(
            query=q,
            key=k,
            value=v
        )
        atten = self.norm(atten)
        print("###################################")
        print(f"atten shape:{atten.shape}")
        print("###################################")
        return atten

    




class TransformerEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder_layer = EncoderLayer()

    def forward(self, feature, box:torch.Tensor, pos_embed):
        queries = box.clone()
        return self.encoder_layer(box, queries, feature, pos_embed)

class Transformer(nn.Module):
    def __init__(self, num_queries:int=50) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.transformer_encoder = TransformerEncoder()

    def forward(self, feature, box, pos_embed):
        return self.transformer_encoder(feature, box, pos_embed)

class FastDETR(nn.Module):
    
    def __init__(self, args) -> None:
        """
        Args:
            args:
                num_queries : box数量
                hidden_dim : transformer隐藏维度
        """
        super().__init__()
        hidden_dim = args.hidden_dim
        num_queries = args.num_queries
        self.backbone = build_backbone(args)
        self.transformer = Transformer(num_queries)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.pos_embed = self._build_position_encoding(args)
        self.class_embed = nn.Linear(hidden_dim, num_queries + 1)

    def _build_position_encoding(self, args):
        N_steps = args.hidden_dim // 2
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
        return position_embedding

    def forward(self, samples: NestedTensor):
        features, pos = self.backbone(samples)
        pred = self.transformer(features, self.query_embed, self.pos_embed)[:self.num_queries]
        outputs_class = self.class_embed(pred)
        outputs_coord = self.bbox_embed(pred).sigmoid()
