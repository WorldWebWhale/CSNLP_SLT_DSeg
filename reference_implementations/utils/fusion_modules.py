import torch
import torch.nn as nn
import torch.nn.functional as F

# Temporal Convolution Block as defined in the SpaMo paper
class TemporalConvBlock(nn.Module):
    """
    1-D temporal convolution with residual connection
    Arg
    in_channels   : Input feature dim
    out_channels  : Output feature dim
    kernel_size   : Size of temporal kernel (e.g. 3)
    dilation      : Dilation factor
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=5, dilation=1, dropout=0.1):
        super().__init__()
        # We add a padding to keep length of input unchanged
        padding = (kernel_size - 1) // 2 * dilation  
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.rescale = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, 1)
        )

    def forward(self, x):       
        y = self.conv(x)
        y = self.norm(F.gelu(y))
        y = self.dropout(y)
        return y + self.rescale(x)   # Note that we have a residual connection (B, out_channels, T)


# Sign Adapter (SpaMo fusion module)
class SignAdapter(nn.Module):
    """
    Fuse spatial & motion streams.

    Args
    spatial : Tensor (B, T, D_s)
    motion  : Tensor (B, N, D_m)

    Returns
    fused   : Tensor (B, M, D_out)
    """
    def __init__(self,
                 d_spatial: int,
                 d_motion: int,
                 d_lmk:     int | None = None,
                 d_model: int = 768,
                 d_hidden: int = 1024,
                 tcn_kernel: int = 5,
                 tcn_dropout: float = 0.1):
        super().__init__()

        # project each modality to common dim d_model
        self.spatial_proj = nn.Linear(d_spatial, d_model)
        self.motion_proj  = nn.Linear(d_motion,  d_model)

        if d_lmk is not None:
            self.use_lmk = True
            self.lmk_proj = nn.Linear(d_lmk, d_model)
        else:
            self.use_lmk = False
            self.lmk_proj = None

        # TCN expects (B, C, T)
        self.tcn1 = TemporalConvBlock(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=tcn_kernel,
            dropout=tcn_dropout
        )

        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # TCN expects (B, C, T)
        self.tcn2 = TemporalConvBlock(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=tcn_kernel,
            dropout=tcn_dropout
        )

        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Cross-modal MLP (Linear -> GELU -> Linear)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model)
        )

    def forward(self, spatial, motion, lmk: torch.Tensor | None = None):
        """
        spatial: (B,T,Ds)   motion: (B,N,Dm)
        returns fused: (B,T,D_model)
        """
        # Projection to the same space using our defined linear layers
        s = self.spatial_proj(spatial)   # (B,T,d_model)
        m = self.motion_proj(motion)     # (B,N,d_model)

        # Optional landmarks
        if self.use_lmk and lmk is not None:
            l = self.lmk_proj(lmk)              # (B, L, d_model)
            Z_cat = torch.cat([s, m, l], dim=1) # (B, T+N+L, d_model)
        else:
            Z_cat = torch.cat([s, m], dim=1)    # (B, T+N, d_model)

        # 1-D TCN over temporal axis
        x = Z_cat.transpose(1, 2)            # (B, d_model, T+N)
        x = self.tcn1(x)       # (B, d_model, Seq)
        x = self.pool1(x)      # (B, d_model, Seq/2)
        x = self.tcn2(x)       # (B, d_model, Seq/2)
        x = self.pool2(x)      # (B, d_model, Seq/4)

        x = x.transpose(1, 2)            # (B,M,d_model)

        # Cross-modal MLP
        out = self.mlp(x)                # (B,M,d_model)
        return out


import torch, torch.nn as nn, math

# Reusing the same Positional encoding from day 1
class SinusoidalPE(nn.Module):
    def __init__(self, dim, max_len=1024):
        super().__init__()
        self.dim = dim
        self.register_buffer("pe", self._build(max_len, dim), persistent=False)

    @staticmethod
    def _build(L: int, D: int) -> torch.Tensor:
        pos = torch.arange(L, dtype=torch.float32).unsqueeze(1)          # (L,1)
        div = torch.exp(torch.arange(0, D, 2) * (-math.log(10000.0) / D))
    
        pe = torch.zeros(L, D)
        pe[:, 0::2] = torch.sin(pos * div)              # Even positions
    
        if D % 2 == 1:                                  # Odd dimension
            pe[:, 1::2] = torch.cos(pos * div[:-1])     # One div value fewer
        else:
            pe[:, 1::2] = torch.cos(pos * div)          # Even dimension
    
        return pe

    def forward(self, x):                         # x (B,T,D)
        T = x.size(1)
        if T > self.pe.size(0):                  
            self.pe = self._build(T, self.dim).to(x.device)
        return x + self.pe[:T]

# Not used there, this is just an idea we give you to have a different way to encode motion positions
class ChunkSpanEmbedding(nn.Module):
    """
    start / end frame IDs  → concat(start_emb ⊕ end_emb)
    spans : LongTensor (B, N, 2)  [start, end]
    """
    def __init__(self, dim, max_frames=4000):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        half = dim // 2
        self.start_emb = nn.Embedding(max_frames + 1, half)
        self.end_emb   = nn.Embedding(max_frames + 1, half)

    def forward(self, spans):                     # (B,N,2)
        s, e = spans[..., 0], spans[..., 1]
        return torch.cat([self.start_emb(s), self.end_emb(e)], dim=-1)

# ---------- (1) fusion block --------------------------------------------
class FusionBlock(nn.Module):
    def __init__(self, spa_dim, flow_dim=None, lmk_dim=None,
                 nhead=8, mlp_ratio=4, dropout=0.1):
        super().__init__()

        self.self_attn  = nn.TransformerEncoderLayer(
            spa_dim, nhead, batch_first=True, dropout=dropout)

        # optional cross-attentions ............................................
        self.cross_flow = (nn.MultiheadAttention(spa_dim, nhead,
                           kdim=flow_dim, vdim=flow_dim, batch_first=True)
                           if flow_dim is not None else None)

        self.cross_lmk  = (nn.MultiheadAttention(spa_dim, nhead,
                           kdim=lmk_dim,  vdim=lmk_dim,  batch_first=True)
                           if lmk_dim  is not None else None)

        # residual-MLP + BN ....................................................
        hidden = int(spa_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(spa_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, spa_dim),
        )
        # **real** BatchNorm1d keeps running_mean / running_var
        self.bn  = nn.BatchNorm1d(spa_dim)

    # --------------------------------------------------------------------- #
    def forward(self, spa, flow=None, lmk=None):           # spa (B,T,D)
        # 1) self-attention (with residual)
        y = self.self_attn(spa)
        spa = spa + y

        # 2) cross-attn(s)
        if self.cross_flow is not None and flow is not None:
            y, _ = self.cross_flow(spa, flow, flow)
            spa = spa + y
        if self.cross_lmk  is not None and lmk  is not None:
            y, _ = self.cross_lmk(spa, lmk, lmk)
            spa = spa + y

        # 3) token-wise MLP + BN (Transformer style)
        y   = self.mlp(spa)
        spa = spa + y                       # residual
        spa = self.bn(spa.transpose(1, 2))  # BN expects (B,C,T)
        spa = spa.transpose(1, 2)
        return spa


# ---------- (2) full fusion module --------------------------------------
class CrossAttentionFusion(nn.Module):
    """
    spatial_feats  : (B,T,D_spa)
    flow_feats     : (B,N,D_flow)  (optional)
    landmark_feats : (B,L,D_lmk)   (optional)
    returns        : (B,T,d_model)  ready for the LLM
    """
    def __init__(self,
                 spatial_dim, flow_dim=None, landmark_dim=None,
                 d_model=2048, use_span=False,
                 nhead=8, num_blocks=2,
                 max_len=1024, max_frames=4000):

        super().__init__()
        self.use_flow = flow_dim is not None
        self.use_lmk  = landmark_dim is not None
        self.use_span = use_span

        # positional encodings
        self.pos_spa  = SinusoidalPE(spatial_dim,  max_len)
        self.pos_flow = SinusoidalPE(flow_dim,     max_len)  if self.use_flow else None
        self.pos_lmk  = SinusoidalPE(landmark_dim, max_len)  if self.use_lmk  else None

        if self.use_span and self.use_flow:
            self.span_emb = nn.Embedding(max_frames + 1, flow_dim)

        # stacked fusion blocks
        self.blocks = nn.ModuleList([
            FusionBlock(spatial_dim,
                        flow_dim if self.use_flow else None,
                        landmark_dim if self.use_lmk else None,
                        nhead=nhead)
            for _ in range(num_blocks)
        ])

        # final projection
        self.to_llm = nn.Linear(spatial_dim, d_model)

    # ------------------------------------------------------------------
    def forward(self, spatial_feats, flow_feats=None,
                spans=None, landmark_feats=None):

        # add positional information
        s = self.pos_spa(spatial_feats)

        f = None
        if self.use_flow and flow_feats is not None:
            f = self.pos_flow(flow_feats)
            if self.use_span and spans is not None:
                f = f + self.span_emb(spans)          # simple additive span-aware bias

        l = self.pos_lmk(landmark_feats) if (self.use_lmk and landmark_feats is not None) else None

        # iterative refinement
        for blk in self.blocks:
            s = blk(s, flow=f, lmk=l)

        return self.to_llm(s)                         # (B,T,d_model)