import torch, torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    VideoMAEModel,
    VideoMAEImageProcessor,
)
import os, glob, torch, tempfile, shutil
import numpy as np
import pandas as pd
from pathlib import Path
from torchvision.io import read_image
from torch.utils.data import Dataset
from typing import Callable, List, Dict, Any
from torch.nn.utils.rnn import pad_sequence

# You might not need to use this atomic saving function but this is of great help
# when first saving the feature files because otherwise they can get corrupted if a crash happen
# while the save is taking place
def atomic_save(obj: Any, path: Path):
    tmp = path.with_suffix(".tmp")
    # Might be obvious but torch save is not atomic
    torch.save(obj, tmp)
    # Os replace is atomic !
    os.replace(tmp, path)

class PhoenixSpaMoLazy(Dataset):
    """
    Returns a dict with keys:
        • motion    : (N, Dm)
        • spatial   : (T, Ds)
        • landmarks : (T, D_lmk)  or  None if *_lmk.pt is missing
        • sentence  : str
    """
    def __init__(self, root: str, sampler: Callable, split="train",
                 cache_dir="cache_spamo",
                 motion_enc=None, spatial_enc=None):
        self.root = Path(root)
        self.split = split
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)

        csv_path = self.root / "annotations" / "manual" / f"PHOENIX-2014-T.{split}.corpus.csv"
        self.df = pd.read_csv(csv_path, delimiter="|")

        # path to the PNG files
        self.png_root = self.root / "features" / "fullFrame-210x260px" / split

        self.motion_enc = motion_enc
        self.spatial_enc = spatial_enc
        self.sampler = sampler

    def _paths(self, name: str):
        mot = self.cache / f"{name}_mot.pt"
        spa = self.cache / f"{name}_se.pt"

        # This is not mandatory
        lmk = self.cache / f"{name}_lmk.pt"     
        return mot, spa, lmk

    # The idea is that we won't precompute everything in the init (which is what you would usually do in a real setting but 
    # this doesn't fit our time constraint), we thus compute live and then cache
    def _compute_and_cache(self, name: str, frame_paths: List[Path]):

        full_imgs  = [np.array(Image.open(p)) for p in frame_paths]
        all_frames = torch.from_numpy(np.stack(full_imgs, axis=0))  # (T,H,W,C)

        # Compute the spatial features
        if self.spatial_enc is not None:
            with torch.no_grad():
                spatial = self.spatial_enc(all_frames).cpu()
        else:
            spatial = torch.empty(0)

        # Compute the motion features
        # Note that this is where we are definining our sampling (i.e. with window stride etc)
        mot_clips = self.sampler(frame_paths)
        mot_list  = []
        for clip in mot_clips:
            vid = clip.permute(3,0,1,2).unsqueeze(0).float() / 255
            with torch.no_grad():
                mot = self.motion_enc(vid)
            mot_list.append(mot)
        motion = torch.stack(mot_list, dim=0)

        # Get the paths where we will cache each feature vector
        mot_p, spa_p, _ = self._paths(name)
        atomic_save(motion,  mot_p)
        atomic_save(spatial, spa_p)
        return motion, spatial

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row  = self.df.iloc[idx]
        vid  = row["name"]
        sent = row["translation"]

        mot_p, spa_p, lmk_p = self._paths(vid)

        # Load from cache or compute motion and spatial features if they don't exist
        if mot_p.exists() and spa_p.exists():
            motion  = torch.load(mot_p, map_location="cpu")
            spatial = torch.load(spa_p, map_location="cpu")
        else:
            folder = self.png_root / vid
            paths  = sorted(folder.glob("*.png"))
            if not paths:
                raise FileNotFoundError(f"No PNGs for {vid}")
            motion, spatial = self._compute_and_cache(vid, paths)

        # Landmarks : load if cached, else None, note that they were all precached so this 
        # is why we don't compute them
        landmarks = torch.load(lmk_p, map_location="cpu") if lmk_p.exists() else None

        return {
            "video_id":  vid,
            "motion":    motion,
            "spatial":   spatial,
            "landmarks": landmarks,   # None if unavailable
            "sentence":  sent,
        }

    def __len__(self):
        return len(self.df)
    
def collate_fn(batch):
    motions   = [b["motion"]    for b in batch]        # (Ni, Dm)
    spatials  = [b["spatial"]   for b in batch]        # (Ti, Ds)
    landmarks = [b["landmarks"] for b in batch]        # (Ti, 129) or None

    # Only the first axis (without counting batch) is used for padding
    # Padding happens with zeros
    mot_p = pad_sequence(motions,   batch_first=True)  # (B, N_max, Dm)
    spa_p = pad_sequence(spatials,  batch_first=True)  # (B, T_max, Ds)

    # Convert None to empty tensor with 0-len time-axis if no landmarks
    lmk_fixed = [l if l is not None else torch.empty(0, 129) for l in landmarks]
    lmk_p = pad_sequence(lmk_fixed, batch_first=True)          # (B, L_max, 129)

    return {
        "video_id": [b["video_id"] for b in batch],
        "motion":   mot_p,
        "spatial":  spa_p,
        "landmarks": lmk_p,
        "tgt":      [b["sentence"] for b in batch],
    }

def spatial_encoder(frames: torch.Tensor) -> torch.Tensor:
    """
    Arg:
    frames : Tensor (L,H,W,C) uint8  [0,255]
    Return: 
    se_feats : Tensor, shape (L, 2*D_s)
        [256 patch-tokens per frame (16x16), each of 2xhidden_size channels]
    """
    L, H, W, C = frames.shape

    feats = []
    # We get the hidden size of our spatial encoder (i.e. the size of the 'embedding space')
    hidden_size = se_model.config.hidden_size  # D_s
    
    for f in frames:                      
        img = Image.fromarray(f.numpy())

        # We start by getting the features for the global image 224x224
        inputs = se_proc(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            # This has shape = (1, 1 + 16*16, hidden_size)
            # +1 coming from the CLS token
            g_feat = se_model(**inputs).last_hidden_state  
        # We rescale it as a patch
        patch_small = g_feat[:, 1:, :].view(1, 16, 16, hidden_size)

        # We upsample the image to 448x448 using BICUBIC interpolation
        img_448 = img.resize((448, 448), Image.BICUBIC)
        tile_feats = []

        # We create the four non overlapping tiles: TL (Top left), TR, BL, BR
        coords = [(0, 0), (224, 0), (0, 224), (224, 224)]
        for x, y in coords:
            crop = img_448.crop((x, y, x + 224, y + 224))
            inputs = se_proc(images=crop, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                q_feat = se_model(**inputs).last_hidden_state 
            tiles = q_feat[:, 1:, :].view(1, 16, 16, hidden_size)
            tile_feats.append(tiles)

        # First we combine top left and top right together (i.e. column combination)
        top = torch.cat(tile_feats[0:2], dim=2)  # (1,16,32,hidden_size)
        # Then we combine bottom left and bottom right the same way
        bottom = torch.cat(tile_feats[2:4], dim=2)  # (1,16,32,hidden_size)
        # Finally we combine top and bottom
        grid32 = torch.cat([top, bottom],   dim=1)  # (1,32,32,hidden_size)

        # We do the average pooling from 32x32 to 16x16
        # Bring channels forward to use avg_pool2d
        grid32 = grid32.permute(0, 3, 1, 2)         # (1, hidden_size,32,32)
        # Note that (16,16) is the resulting size, it will thus select a 2x2 kernel
        pooled = F.adaptive_avg_pool2d(grid32, (16, 16))
        # Permute back to (1,16,16,hidden_size)
        patches_large = pooled.permute(0, 2, 3, 1)

        # Concatenate small and pooled large -> (1,16,16,2*hidden_size)
        multi = torch.cat([patch_small, patches_large], dim=-1)

        # We flatten all patches -> (256, 2*hidden_size) because we will want to average from all patches
        frame_feats = multi.view(16 * 16, 2 * hidden_size)
        feats.append(frame_feats.cpu())

    stacked = torch.stack(feats, dim=0)

    # We take the said average 
    Zs = stacked.mean(dim=1)

    return Zs   # (L, 2*D_s)

def motion_encoder(clip: torch.Tensor) -> torch.Tensor:
    """
    Arg
    clip : Tensor (1,C,T,H,W) float in [0,1]
           (C=3, T=16)
    Returns
    mot_feats : Tensor (T, Dm)
    """
    # VideoMAE expects list of PIL / numpy
    frames = [Image.fromarray(
        (clip[0, :, t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    ) for t in range(clip.shape[2])]

    inputs = me_proc(frames, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = me_model(**inputs).last_hidden_state  # (1, N+1, D_m)
    
        # Take CLS token (index 0, from the documentation) 
        cls = outputs[:, 0, :]                 # (1,D_m)
    return cls.squeeze(0).cpu()  # shape (1, D_m)

# Sampler with stride 8
def sample_with_stride(frame_paths, chunk_len=16, stride=8, max_chunks=None):
    """Return list of chunks (L=16,H,W,C) with stride 8"""
    frames = [np.array(Image.open(p)) for p in sorted(frame_paths)]
    chunks = []
    for start in range(0, len(frames) - chunk_len + 1, stride):
        chunk = np.stack(frames[start : start + chunk_len], axis=0)
        chunks.append(torch.from_numpy(chunk))  # (16,H,W,C)
        if max_chunks and len(chunks) >= max_chunks:
            break
    if not chunks:  # Fallback
        chunks.append(torch.from_numpy(np.stack(frames, axis=0)))
    return chunks