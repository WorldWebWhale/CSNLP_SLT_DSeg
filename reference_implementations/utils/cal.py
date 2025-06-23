import torch.nn as nn
import torch, torch.nn.functional as F


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

def vt_align_step(sign_batch, text_batch, tokenizer, llm, temperature):

    # Unwrap if wrapped (useful for sbatch).
    base_llm = llm.module if hasattr(llm, "module") else llm

    # Pool and normalize sign features
    z_s = sign_batch.mean(dim=1)           # -> (B, d')
    z_s = F.normalize(z_s, dim=-1)         # -> (B, d')

    # Tokenize references exactly like SpaMo
    tok = tokenizer(
        text_batch,
        padding="longest",
        truncation=True,
        return_tensors="pt"
    ).to(base_llm.device)

    # Embed, pool and normalize text
    text_embs = base_llm.encoder.embed_tokens(tok.input_ids)  #  (B, U, d')
    z_t = text_embs.mean(dim=1)                                #  (B, d')
    z_t = F.normalize(z_t, dim=-1)                             #  (B, d')

    # Cosine‐scaled logits with learned temperature
    logit_scale = temperature.exp()
    logits = z_t @ z_s.t() * logit_scale  # -> (B, B)

    # Contrastive loss
    loss = clip_loss(logits)
    return loss