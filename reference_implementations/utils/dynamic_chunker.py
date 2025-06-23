import torch
import torch.nn as nn
from typing import List
from torch.cuda.amp import autocast

class DynamicChunker(nn.Module):
    def __init__(self, dim: int, max_chunk_len: int = 32, nhead: int = 4):
        super().__init__()
        self.max_chunk_len = max_chunk_len

        self.enc = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, batch_first=True,
            norm_first=True, dropout=0.1
        ).float()                               # keep in fp32

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.eoc_head  = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, 1),  nn.Sigmoid()
        ).float()

    # ----------------------------------------------------------
    @staticmethod
    def _pad_mask(seq: torch.Tensor) -> torch.Tensor:
        return (seq.abs().sum(-1) == 0)         # (L,)

    # ----------------------------------------------------------
    def forward(self, frames: torch.Tensor, threshold: float = .5):
        """
        frames : (B,T,D)
        returns list[Tensor] – len == B, each (N_chunks_i,D)
        """
        B, T, D = frames.shape
        device  = frames.device
        out     = []

        for b in range(B):
            i = 0
            while i < T and torch.any(frames[b, i:] != 0):
                j = i + 1                       # start with one frame
                is_end = False

                while j <= min(i + self.max_chunk_len, T) and not is_end:
                    win = frames[b, i:j]        # (L,D)  growing window
                    pad = self._pad_mask(win)

                    # ------------- contextualise (fp32) -------------
                    with torch.cuda.amp.autocast(enabled=False):
                        seq = torch.cat(
                            [self.cls_token.to(device), win.unsqueeze(0)], 1)
                        kp  = torch.cat(
                            [torch.zeros(1, dtype=torch.bool, device=device),
                             pad], 0).unsqueeze(0)

                        enc = self.enc(seq.float(),
                                       src_key_padding_mask=kp)[:, 0]  # (1,D)
                        p_eoc = self.eoc_head(enc)                      # (1,1)
                        is_end = p_eoc.item() > threshold

                    if not is_end and j - i < self.max_chunk_len:
                        j += 1          # enlarge the window
                    else:
                        break           # stop growing

                out.append(enc.squeeze(0).half())   # save chunk rep.
                i = j                                # next chunk starts here

        # group chunks by sample
        if not out:
            return [torch.zeros(1, D, device=device, dtype=frames.dtype)]
        chunks_per_sample = []
        cursor = 0
        for b in range(B):
            # count how many chunk reps belong to sample b
            # (they were produced in temporal order)
            if cursor >= len(out):                   # finished all
                chunks_per_sample.append(
                    torch.zeros(1, D, device=device, dtype=frames.dtype))
                continue
            cnt = 0
            while (cursor + cnt) < len(out) and out[cursor + cnt].dim() == 1:
                cnt += 1
            if cnt == 0:
                chunks_per_sample.append(
                    torch.zeros(1, D, device=device, dtype=frames.dtype))
            else:
                chunks_per_sample.append(torch.stack(out[cursor:cursor+cnt]))
            cursor += cnt

        return chunks_per_sample

import torch
import torch.nn.functional as F

def pad_chunk_batch(chunk_list):
    """
    chunk_list : list[Tensor] – each (N_i, D)
    returns    : padded (B, N_max, D) and boolean mask (B, N_max)
    """
    lengths  = [c.size(0) for c in chunk_list]
    max_len  = max(lengths)
    D        = chunk_list[0].size(1)
    device   = chunk_list[0].device

    padded = torch.zeros(len(chunk_list), max_len, D, device=device)
    mask   = torch.zeros(len(chunk_list), max_len,   device=device, dtype=torch.bool)

    for i, (tensor, L) in enumerate(zip(chunk_list, lengths)):
        padded[i, :L] = tensor
        mask[i, :L]   = True

    return padded, mask
from torch.cuda.amp import autocast

from torch.cuda.amp import autocast
import torch, torch.nn as nn, torch.nn.functional as F

# ---------------- utilities ----------------
class SafeEncLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            norm_first=True
        ).float()

    def forward(
        self,
        x: torch.Tensor,
        *,                              # keyword-only for clarity
        key_padding_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ):
        # if the caller used the “old” name, fall back to it
        if key_padding_mask is None:
            key_padding_mask = src_key_padding_mask

        with torch.amp.autocast("cuda", enabled=False):
            x = x + 1e-5 * torch.randn_like(x)         # anti-collapse noise
            y = self.layer(x.float(),
                           src_key_padding_mask=key_padding_mask)
            if not torch.isfinite(y).all():
                y = torch.nan_to_num(y)
        return y


def valid_mask(win: torch.Tensor) -> torch.Tensor:
    return win.abs().sum(-1) == 0          # True ↔ row is padding
# utils -----------------------------------------------------------------
def _pad_mask(win: torch.Tensor) -> torch.Tensor:          # (L,D)
    """Bool mask True = padding row (all-zeros)."""
    return (win.abs().sum(-1) == 0)

def _enc_layer(d_model: int, n_heads: int = 4) -> nn.Module:
    layer = nn.TransformerEncoderLayer(
        d_model, n_heads,
        batch_first=True, norm_first=True, dropout=0.1
    )
    return layer.float()           # keep weights in fp32

# model -----------------------------------------------------------------
class FixedStrideMeanChunker(nn.Module):
    """
    • Split every sample into consecutive, non-overlapping windows
      of exactly `stride` frames (except the tail).
    • Summarise each window by masked mean  → proj MLP.
    • Return per-sample list of chunk vectors.

    Args
    ----
    dim      : feature dimension D
    stride   : window length (default 8)
    """
    def __init__(self, dim: int, stride: int = 8):
        super().__init__()
        self.stride = stride

        # small projection just like before
        self.proj = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim)
        ).float()

    # ------------------------------------------------------------------
    @staticmethod
    def _pad_mask(seq: torch.Tensor) -> torch.Tensor:
        """True where the row is all-zero (padding)."""
        return (seq.abs().sum(-1) == 0)          # (L,)

    # ------------------------------------------------------------------
    def forward(self,
                frames: torch.Tensor            # (B,T,D) fp16/bf16/fp32
               ) -> List[torch.Tensor]:
        B, T, D = frames.shape
        device  = frames.device
        out_all = []

        #print("\n--- [FixedStrideMeanChunker Trace] ---")
        for b in range(B):
            sample_chunks = []
            #print(f"[Sample {b}]")
            for start in range(0, T, self.stride):
                end = min(start + self.stride, T)
                win = frames[b, start:end]                    # (L,D)
                if torch.all(win == 0):
                    break                                     # rest is padding

                pad = self._pad_mask(win)                     # (L,)
                valid = (~pad).float().unsqueeze(-1)          # (L,1)

                # masked mean in fp32 for numeric stability
                with torch.cuda.amp.autocast(enabled=False):
                    denom = valid.sum().clamp(min=1.)
                    mean  = (win.float() * valid).sum(0) / denom   # (D,)
                    rep   = self.proj(mean.unsqueeze(0))           # (1,D)

                #print(f"  ✓ chunk committed: [{start}:{end}]"
                #      f" (len={end-start})")
                sample_chunks.append(rep.squeeze(0).half())  # store fp16

            # if a sample ended up empty (all-pad) we give a single 0-vec
            if not sample_chunks:
                sample_chunks.append(
                    torch.zeros(D, device=device, dtype=frames.dtype))

            out_all.append(torch.stack(sample_chunks, 0))    # (N_i,D)

        #print("--- End Trace ---\n")
        return out_all

# ---------------- mean-pooled dynamic ----------------
import torch, torch.nn as nn
from torch.cuda.amp import autocast
from typing import List


class MeanPoolDynamicChunker(nn.Module):
    """
    Dynamic segmentation identical to your previous ‘CLS’ version,
    **except** that the window is summarised by a masked mean instead
    of a CLS-token Transformer pass.
    """
    def __init__(self, dim: int, max_chunk_len: int = 32):
        super().__init__()
        self.max_chunk_len = max_chunk_len

        # light projection for the averaged window
        self.proj = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim)
        ).float()

        # end-of-chunk classifier
        self.eoc_head = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, 1), nn.Sigmoid()
        ).float()

    # ----------------------------------------------------------
    @staticmethod
    def _pad_mask(seq: torch.Tensor) -> torch.Tensor:
        """Bool mask – True where the row is all-zero (padding)."""
        return (seq.abs().sum(-1) == 0)        # (L,)

    # ----------------------------------------------------------
    def forward(self,
                frames: torch.Tensor,          # (B,T,D)
                threshold: float = .5) -> List[torch.Tensor]:
        """
        returns list[Tensor] – len == B; each of shape (N_chunks_i, D)
        """
        B, T, D = frames.shape
        device  = frames.device
        out     = []

        #print("\n--- [MeanPoolDynamicChunker Trace] ---")
        for b in range(B):
            i = 0
            #print(f"[Sample {b}]")
            while i < T and torch.any(frames[b, i:] != 0):
                j, is_end = i + 1, False     # start with 1-frame window

                while j <= min(i + self.max_chunk_len, T) and not is_end:
                    win = frames[b, i:j]                     # (L,D)
                    pad = self._pad_mask(win)                # (L,)
                    valid = (~pad).float().unsqueeze(-1)     # (L,1)

                    # --------------------------------------------------
                    # masked mean in fp32 for numeric stability
                    # --------------------------------------------------
                    with autocast(enabled=False):
                        denom = valid.sum()
                        if denom == 0:
                            rep = torch.zeros(1, D, device=device)
                            is_end = True
                        else:
                            
                            mean = (win.float() * valid).sum(0) / denom
                            rep  = self.proj(mean.unsqueeze(0)) 
                            is_end = self.eoc_head(rep).item() > threshold

                    if not is_end and j - i < self.max_chunk_len:
                        j += 1
                    else:
                        break

                #print(f"  ✓ chunk committed: [{i}:{j}] (len={j-i})")
                out.append(rep.squeeze(0).half())           # store fp16 rep
                i = j                                       # slide window

        # --------------------------------------------------------------
        # regroup reps per sample (exactly as in your original code)
        # --------------------------------------------------------------
        if not out:
            return [torch.zeros(1, D, device=device, dtype=frames.dtype)]

        chunks_per_sample, cursor = [], 0
        for b in range(B):
            if cursor >= len(out):
                chunks_per_sample.append(
                    torch.zeros(1, D, device=device, dtype=frames.dtype))
                continue
            cnt = 0
            while (cursor + cnt) < len(out) and out[cursor + cnt].dim() == 1:
                cnt += 1
            if cnt == 0:
                chunks_per_sample.append(
                    torch.zeros(1, D, device=device, dtype=frames.dtype))
            else:
                chunks_per_sample.append(torch.stack(out[cursor:cursor+cnt]))
            cursor += cnt

        #print("--- End Trace ---\n")
        return chunks_per_sample
