"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

# drop_path方法： 随机深度方法
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# patch embedding
class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        # 224/16 = 14 -> [14*14]
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # 计算 patches的数目
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # conv [3,768,16,16]
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 如果未传入nom_layer：不做任何操作
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    # 正向传播过程
    def forward(self, x):
        B, C, H, W = x.shape
        # 判断传入图片的尺寸是否符合要求  [vit模型中输入图片的大小必须是固定的]
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

# transformer：multi-head self-attention模块
# 用于计算 Attention(Q,K,V) = softmax(QK^T/dk^0.5)V
class Attention(nn.Module):
    def __init__(self,
                 dim,                   # 输入token的dim
                 num_heads=8,           # head的个数
                 qkv_bias=False,        # 生成 QKV 的时候，是否使用偏置
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        # 每一个head的 QKV 对应的dim
        head_dim = dim // num_heads
        # 未传入qk_scale -> 1/DK^0.5
        self.scale = qk_scale or head_dim ** -0.5
        # 通过一个全连接层 -> 生成 Q，K，V
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 定义一个dropout层
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        # 通过一个全连接层 -> 生成 WO
        self.proj = nn.Linear(dim, dim)
        # 定义一个dropout层
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    # 定义正向传播过程
    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        # [图片数目, 14*14+1(class_token), 768]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head] (调整数据顺序 )
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # 通过切片的方式得到q,k,v
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # 接下来的操作：针对每个head的Q K V进行操作
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # -1 代表对每一行 数据进行 softmax处理
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# MLP Block 层
class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self,
                 in_features,              # 输入结点个数
                 hidden_features=None,     # 第一个全连接层节点个数（in_features的4倍）
                 out_features=None,        # 默认= in_features
                 act_layer=nn.GELU,        # GELU激活函数
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# Encoder Block -> 重复堆叠L次
class Block(nn.Module):
    def __init__(self,
                 dim,                   # 每个token的dim
                 num_heads,             # heads个数
                 mlp_ratio=4.,          # 第一个全连接层节点个数 = 4 * 输入结点个数
                 qkv_bias=False,        # 是否使用 bias
                 qk_scale=None,         # 是否使用 qk_scale (q * k转置)
                 drop_ratio=0.,         # 最后全连接层后使用的drop_ratio
                 attn_drop_ratio=0.,    # (q * k转置)/(根号dk) 通过 softmax后的dropout层
                 drop_path_ratio=0.,    # drop path ratio
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        # Layer Norm 1
        self.norm1 = norm_layer(dim)
        # Multi-Head Attention
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # 如果传入的drop_path_ratio > 0: 创建一个dropout层
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        # Layer Norm 2
        self.norm2 = norm_layer(dim)
        # [dim * 4]
        mlp_hidden_dim = int(dim * mlp_ratio)
        # MLP结构
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    # 正向传播过程
    def forward(self, x):
        # x -> norm1 -> Attention -> drop_path + x
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # x -> norm2 -> MLP -> drop_path + x
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# 定义 Vision Transformer类
class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size [图片尺寸：224*224]
            patch_size (int, tuple): patch size     [patch斑块尺寸：16*16]
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer 【transformer encoder中重复堆叠encoder block的次数】
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set 【MLP head最后的pre-logits中全连接层的节点个数】
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer 【embedding层】
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        # num_tokens = 1
        self.num_tokens = 2 if distilled else 1
        # 默认用 partial 方法传入默认参数： eps
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        # 默认使用GELU
        act_layer = act_layer or nn.GELU

        # 构建 patch embedding
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        # 得到 patches 的个数
        num_patches = self.patch_embed.num_patches

        # 构建一个可训练的参数（0矩阵初始化）[1,1,768] -> Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 兼容别的模型（使用不到）
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # 构建一个可训练的参数（0矩阵初始化）[1,197,768] -> Position Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        # 构建 dropout 层
        self.pos_drop = nn.Dropout(p=drop_ratio)

        # [0,drop_path_ratio],共有depth个 -> 等差序列 -> 每个encoder block中的 drop path值：递增
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        # 重复堆叠 L 次 -> Encoder Block -> 唯一变化的参数是drop_path_ratio
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        # 构建 norm_layer
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            # 构建 pre_logits
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        # 最终的全连接层！
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    # 正向传播方法
    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768] 将我们的class token在 batch维度 复制 batchsize份
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            # 将class token与 x 在 196 的维度进行拼接 -> 197
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        # 将 x 与 position embedding 相加 并通过一个dropout层
        x = self.pos_drop(x + self.pos_embed)
        # 通过堆叠 L 次的 Encoder Block层
        x = self.blocks(x)
        # 通过Layer Nrom层
        x = self.norm(x)
        if self.dist_token is None:
            # 将x 第二个维度 索引为 0 的数据 [class token] -> pre_logits
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            # 通过我们的全连接层 liner 得到输出
            x = self.head(x)
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

# 根据不同的参数传入不同的类 [base large huge]
# patch16 : patch_size 16*16 [16 比 32 计算量更大]
# in21k : imagenet21k，对应类别个数 21843
# 如果不去使用预训练模型的话，效果特别差 [只有在非常大的数据集上才会有好的效果]
def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model

# 太大了，不建议使用
def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model
