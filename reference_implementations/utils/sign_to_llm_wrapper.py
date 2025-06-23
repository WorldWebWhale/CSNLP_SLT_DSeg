import random, torch, math
from typing import List, Tuple

# Prompt builder 
def build_prompt(
    icl_pairs: List[Tuple[str, str]],
    shuffle: bool = True,
) -> str:
    """
    Return an instruction prompt that contains up to icl_pairs
    (Foreign -> German) examples
    """
    if shuffle:
        random.shuffle(icl_pairs)

    lines = ["Translate the sign-language input above into German. Your task is a translation task just as:\n"]

    # Adding the ICL examples
    for src, trg in icl_pairs:
        lines.append(f"{src} -> {trg}")          

    # The model will continue from there
    lines.append("\nAnswer:")                   
    return "\n".join(lines)

# Wrapper module that takes in our fused sign features and prepare them to be fed into the LLM space (here the LLM chosen is T5)
class Sign2T5(torch.nn.Module):
    """
    Minimal wrapper that:
        - Concatenates sign features (already in d_model) before token embeddings
        - Feeds fused sequence into LoRA-augmented Flan-T5 to generate translation
    """
    def __init__(self, llm, tok, translation_pool):
        super().__init__()
        self.llm = llm
        self.tok = tok
        self.translation_pool = translation_pool

    def forward(
        self,
        sign_feats: torch.Tensor,   # (B, M, d_model) from Sign-Adapter
        sign_mask : torch.Tensor | None = None,
        max_new_tokens: int = 50,
        labels: List[str]| None = None,
    ):
        B, M, _ = sign_feats.shape
        device = next(self.parameters()).device

        vis_mask = (sign_feats.abs().sum(-1) > 0).long()

        if sign_mask is None:                            
            vis_mask = (sign_feats.abs().sum(-1) > 0).long()
        else:
            vis_mask = sign_mask.long()                   # provided mask

        # Build prompts        
        prompts = []
        for i in range(B):
            examples = random.sample(self.translation_pool, k=3)
            prompt = build_prompt(examples)
            prompts.append(prompt)

        # Tokenize prompts
        enc = self.tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        text_mask = enc.attention_mask

        # Prepare embeddings
        text_emb = self.llm.encoder.embed_tokens(enc.input_ids)  # (B, U, d)
        sign_emb = sign_feats.to(device)                         # (B, M, d)

        # Concat time-wise (B, U+M, d)
        # i.e. now we have our sign embeddings coming just before our prompt
        fused_emb = torch.cat([sign_emb, text_emb], dim=1)

        # New attention mask: 1 for all tokens from either the signs or the text
        fused_mask = torch.cat([vis_mask, text_mask], dim=1)

        # Encode everything
        encoder_outputs = self.llm.encoder(
            inputs_embeds=fused_emb,
            attention_mask=fused_mask,
            return_dict=True,
        )

        # If we HAVE labels, we are training => we want to return a Loss
        if labels is not None:
            tgt = self.tok(
                labels,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).input_ids.to(device)

            out = self.llm(
                encoder_outputs=encoder_outputs,
                attention_mask=fused_mask,
                labels=tgt,
                return_dict=True,
            )
            # Note that as we gave labels, the out is already a loss and not new text
            return out

        # If we DON'T HAVE labels, we are on inference path -> we simply generate next tokens
        gen_ids = self.llm.generate(
            encoder_outputs = encoder_outputs,
            attention_mask  = fused_mask,
            num_beams       = 5,     
            length_penalty  = 0.6,    
            max_new_tokens  = max_new_tokens,
            pad_token_id    = self.tok.pad_token_id,
        )

        return self.tok.batch_decode(gen_ids, skip_special_tokens=True)