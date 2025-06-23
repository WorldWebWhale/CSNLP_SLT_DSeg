# Setup cell, don't forget to run it

DATA_PATH_LORIC = "/capstor/scratch/cscs/lherbe/heilbronnn-ai-summer-school-2025/notebooks/day-4-blockEngineering/phoenix_dir"
DATA_PATH_CARLOS = "/iopsstor/scratch/cscs/ccotrini"

import os

os.environ["HF_TOKEN"] = ""
os.environ["HF_HOME"] = os.path.join(os.environ["SCRATCH"], "huggingface")
print(os.environ["HF_HOME"])

from huggingface_hub import login

login(token=os.environ["HF_TOKEN"])

# Setup: Importing essential libraries
from accelerate import Accelerator
import torch

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.fusion_modules import TemporalConvBlock, SignAdapter
from utils.sign_to_llm_wrapper import build_prompt, Sign2T5

import random, torch, math
from typing import List, Tuple

import torch, math, random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from accelerate import Accelerator
from pathlib import Path
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import ConstantLR      
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
from torch.utils.data import DataLoader

from utils.helper import spatial_encoder, motion_encoder, sample_with_stride, collate_fn
from utils.helper import PhoenixSpaMoLazy

# IN CASE YOU ALREADY STARTED TRAINING, AND WANT TO START BACK FROM WHERE YOU STOPPED, FILL THIS VARIABLE
# Example : ckpt_dir   = "checkpoints/spamo_sadfusion_epoch_39_20250613_213359" 
ckpt_dir   = ""   

# Hyper-params 
BATCH_SIZE    = 4
GRAD_ACCUM    = 4
EPOCHS        = 40
LR            = 1e-4
MAX_NEW_TOK   = 50
MAX_SRC_LEN   = 128
MAX_TGT_LEN   = 128
ICL_K         = 2                      # ICL = In-Context Learning, i.e. this defines how many examples of translation we put in the prompt

D_SPATIAL = 2048        # CLIP pooled dim
D_MOTION  = 1024        # VideoMAE CLS dim
D_LMK = 129

from utils.scheduler import LambdaWarmUpCosineScheduler 
from torch.optim.lr_scheduler import LambdaLR

def build_components():
        
    # Load the Spatial encoder
    # Instantiating the pretrained spatial encoder
    se_model_name = "openai/clip-vit-large-patch14"
    # Note that we put it in eval mode as this is frozen, if you choose to train it you should
    # modify that part
    se_model  = CLIPVisionModel.from_pretrained(se_model_name).eval()
    se_proc   = CLIPImageProcessor.from_pretrained(se_model_name)
    
    # Load the Motion encoder
    # We instantiate the pretrained motion encoder
    me_model_name = "MCG-NJU/videomae-large-finetuned-kinetics"
    me_model = VideoMAEModel.from_pretrained(me_model_name).eval()
    me_proc  = VideoMAEImageProcessor.from_pretrained(me_model_name)
    
    # Load your LLM and add LoRA to it
    llm_name = "google/flan-t5-xl"
    tokenizer = AutoTokenizer.from_pretrained(llm_name, model_max_length=1024)
    
    base_model = AutoModelForSeq2SeqLM.from_pretrained(llm_name)
    # Freeze base weights (remember the Mellama Block !)
    for p in base_model.parameters():
        p.requires_grad_(False)
    
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q", "v", "k", "o", "wi", "wo"],  # T5 linear sub-modules
        lora_dropout=0.05,
        bias="none", task_type="SEQ_2_SEQ_LM",
    )
    llm = get_peft_model(base_model, lora_cfg)
    
    D_MODEL   = llm.config.d_model
    
    # Define your translation pool (yes, it's a bit biased toward weather forecasting)
    translation_pool: List[Tuple[str, str]] = [
        
        # General stuff
        ("Quel temps fait-il ?", "Wie ist das Wetter?"),
        ("Il fait beau aujourd'hui.", "Heute ist schönes Wetter."),
        ("Il pleut.", "Es regnet."),
        ("Il neige.", "Es schneit."),
        ("Il fait froid.", "Es ist kalt."),
        ("Il fait chaud.", "Es ist heiß."),
        ("Il y a du vent.", "Es ist windig."),
        ("Le soleil brille.", "Die Sonne scheint."),
        ("Le ciel est couvert.", "Der Himmel ist bedeckt."),
        ("Il fait mauvais.", "Es ist schlechtes Wetter."),
        
        # Temperature
        ("Quelle est la température ?", "Wie hoch ist die Temperatur?"),
        ("Il fait dix degrés.", "Es sind zehn Grad."),
        ("Les températures vont baisser.", "Die Temperaturen werden sinken."),
        ("Il va faire plus chaud demain.", "Es wird morgen wärmer."),
        
        # Forecasting
        ("La météo annonce de la pluie.", "Der Wetterbericht sagt Regen voraus."),
        ("Il y aura des orages ce soir.", "Heute Abend wird es Gewitter geben."),
        ("Le temps va s'améliorer.", "Das Wetter wird besser."),
        ("Le temps va empirer.", "Das Wetter wird schlechter."),
        ("Il y aura du brouillard demain matin.", "Morgen früh wird es Nebel geben."),
        
        # Others
        ("J’ai entendu qu’il allait neiger.", "Ich habe gehört, dass es schneien wird."),
        ("Le vent souffle fort.", "Der Wind weht stark."),
        ("Les routes sont glissantes à cause du verglas.", "Die Straßen sind wegen Glatteis rutschig."),
        ("Un orage approche.", "Ein Gewitter zieht auf."),
        ("Le temps est instable.", "Das Wetter ist unbeständig."),
    ]
    
    # Instantiate our fusion module
    adapter = SignAdapter(D_SPATIAL, D_MOTION, d_model=D_MODEL)  
    
    # Instantiate our wrapper from out of adapter -> in of LLM
    sign2t5 = Sign2T5(llm, tokenizer, translation_pool=translation_pool)

    # Define the Dataset and DataLoader
    train_csv = DATA_PATH_LORIC+"/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv"
    png_root  = DATA_PATH_LORIC+"/PHOENIX-2014-T/features/fullFrame-210x260px/train"
    
    # Creates a Phoenix Dataset based on what was done previously
    dataset = PhoenixSpaMoLazy(
        root=DATA_PATH_LORIC+"/PHOENIX-2014-T",
        split="train",
        sampler=sample_with_stride, 
        cache_dir=DATA_PATH_CARLOS+"/cache_spamo/cache_spamo",
        motion_enc=motion_encoder,
        spatial_enc=spatial_encoder  
    )
    
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Build optimizer (all trainable params)
    lora_params   = [p for p in llm.parameters() if p.requires_grad]
    adapter_params = list(adapter.parameters())
    
    trainable = lora_params + adapter_params
    
    optimizer = torch.optim.AdamW(
        trainable,
        lr          = 1e-4,          
        betas       = (0.9, 0.98),
        weight_decay= 0.01
    )
    
    total_updates = (len(loader) // GRAD_ACCUM) * EPOCHS   # one call per real optimiser step
    warmup_steps  = 10_000 

    # Constant-LR scheduler (factor = 1.0 means that it doesn't change), we define one here in case you wan't
    # to try out a specific scheduler

    sched_lambda = LambdaWarmUpCosineScheduler(
        warm_up_steps   = warmup_steps,
        lr_min          = 0.5,       # = 5e-5 / 1e-4
        lr_max          = 1.0,
        lr_start        = 0.0,
        max_decay_steps = total_updates
    )

    scheduler = LambdaLR(optimizer, lr_lambda=sched_lambda)

    return sign2t5, llm, adapter, optimizer, scheduler, dataset, loader,trainable

from datetime import datetime
import os
import torch.multiprocessing as mp

accelerator = Accelerator(mixed_precision="fp8")
device = accelerator.device

sign2t5, llm, adapter, optimizer, scheduler, dataset, loader, trainable = build_components()

sign2t5, llm, adapter, optimizer, loader, scheduler = accelerator.prepare(
    sign2t5, llm, adapter, optimizer, loader, scheduler
)

import numpy as np
import torch.serialization as ts

ts.add_safe_globals([
    np.core.multiarray.scalar,  # already done
    np.dtype,
    np.float64,
])

# If you did not let this path empty then it will reload your model
if ckpt_dir:
    accelerator.load_state(ckpt_dir)

# Create logging directory (This setup will help you report your experiments)
loss_log_dir = "loss_logs"
os.makedirs(loss_log_dir, exist_ok=True)

# Unique file for this run
run_name = f"spamo_sadfusionloss2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
loss_log_path = os.path.join(loss_log_dir, run_name)

# Open a list to collect loss logs
loss_log = []

for epoch in range(EPOCHS):
    print("Epoch {}".format(epoch))
    for step, batch in enumerate(loader):
        with accelerator.accumulate(sign2t5):
            motion  = batch["motion"].to(device)    # (B, N_max, Dm)
            spatial = batch["spatial"].to(device)   # (B, T_max, Ds)
            
            fused   = adapter(spatial, motion)               # (B, M, d_sign)

            # This is the SIGN2T5
            output = sign2t5(
                sign_feats = fused,
                max_new_tokens = MAX_TGT_LEN,
                labels = batch["tgt"],      # Feed in labels to compute the loss
            )
            loss = output.loss / GRAD_ACCUM

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        if accelerator.is_main_process and step % 50 == 0:
            loss_val = loss.item() * GRAD_ACCUM
            print(f"Epoch {epoch} | Step {step} | Loss {loss_val:.4f}")
            
            # Log to memory
            loss_log.append(f"{epoch},{step},{loss_val:.6f}")

    accelerator.wait_for_everyone()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #ckpt_dir = f"checkpoints/spamo_sadfusion_epoch_{epoch}_{timestamp}"
    #accelerator.save_state(ckpt_dir, safe_serialization=True)

print("Training finished")

if accelerator.is_main_process and ckpt_dir:
    with open(loss_log_path, "w") as f:
        f.write("\n".join(loss_log))
    print(f"Loss log saved to {loss_log_path}")


# Put everything in eval mode and disable grads
sign2t5.eval()
adapter.eval()
llm.eval()
torch.set_grad_enabled(False)

sign2t5.to(device)
adapter.to(device)
llm.to(device)

# Build a small test‐loader 
test_dataset = PhoenixSpaMoLazy(
    root=DATA_PATH_LORIC+"/PHOENIX-2014-T",
    split="test",            
    cache_dir=DATA_PATH_CARLOS+"/cache_spamo_test/cache_spamo_test",
    motion_enc=motion_encoder,
    spatial_enc=spatial_encoder,
    sampler=sample_with_stride,
)
# Limit to first 10 examples
from torch.utils.data import Subset
test_dataset = Subset(test_dataset, list(range(0, 10)))

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=1,   
    collate_fn=collate_fn
)

# Loop and generate
for batch in test_loader:
    # Load features and ground truth
    motion  = batch["motion"].to(device)    # (1, N, Dm)
    spatial = batch["spatial"].to(device)   # (1, T, Ds)
    gt = batch["tgt"][0]               # the gold German

    # Fuse and generate
    fused    = adapter(spatial, motion)     # (1, M, d_sign)
    pred = sign2t5(
        sign_feats   = fused,
        max_new_tokens = MAX_NEW_TOK
    )[0]  # Extract string

    print(f"\n -> Video: {batch['video_id'][0]}")
    print(f"    Gold: {gt}")
    print(f"    Pred: {pred}")


import torch
from torch.utils.data import DataLoader, Subset
from utils.bleu import compute_bleu  
import re
from tqdm import tqdm

# Ensuring eval mode and disabled gradients
sign2t5.eval(); adapter.eval(); llm.eval()
torch.set_grad_enabled(False)

test_dataset = PhoenixSpaMoLazy(
    root=DATA_PATH_LORIC+"/PHOENIX-2014-T",
    split="test",                 
    cache_dir=DATA_PATH_CARLOS+"/cache_spamo_test/cache_spamo_test",
    motion_enc=motion_encoder,
    spatial_enc=spatial_encoder,
    sampler=sample_with_stride,
)

test_loader = DataLoader(
    test_dataset, batch_size=4, shuffle=False,
    collate_fn=collate_fn, num_workers=200
)
test_loader = accelerator.prepare(test_loader) 

# Collect reference and predictions
reference_corpus   = []   # list[list[tokens]]
translation_corpus = []   # list[tokens]

llm.to(device)
adapter.to(device)
sign2t5.to(device)

from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU, CHRF, TER

def evaluate_results(predictions, references, split="train", device='cpu', tokenizer='13a'):
    """
    Evaluate prediction results using BLEU and ROUGE metrics.

    Args:
        predictions (list): List of predicted sequences.
        references (list): List of reference sequences.
        tokenizer (object, optional): Tokenizer if needed for evaluation.
        split (str): The data split being evaluated.

    Returns:
        dict: A dictionary of evaluation scores.
    """
    log_dicts = {}

    bleu4 = BLEU(max_ngram_order=4, tokenize=tokenizer).corpus_score(predictions, [references]).score
    log_dicts[f"{split}/bleu4"] = bleu4

    if split == 'test':
        for i in range(1, 4):
            score = BLEU(max_ngram_order=i, tokenize=tokenizer).corpus_score(predictions, [references]).score
            log_dicts[f"{split}/bleu" + str(i)] = score

        # Calculate ROUGE-L score
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = [scorer.score(ref, pred)['rougeL'] for ref, pred in zip(references, predictions)]
        
        # Aggregate ROUGE-L scores (average precision, recall, and F1)
        avg_precision = sum(score.precision for score in rouge_scores) / len(rouge_scores)
        avg_recall = sum(score.recall for score in rouge_scores) / len(rouge_scores)
        avg_f1 = sum(score.fmeasure for score in rouge_scores) / len(rouge_scores)

        log_dicts[f"{split}/rougeL_precision"] = avg_precision
        log_dicts[f"{split}/rougeL_recall"] = avg_recall
        log_dicts[f"{split}/rougeL_f1"] = avg_f1

    return log_dicts

predictions, references = [], []

for batch in tqdm(test_loader, desc="Inference", unit="sample"):
    refs = batch["tgt"]            # gold sentence (str)

    motion   = batch["motion"].to(device)
    spatial  = batch["spatial"].to(device)

    fused    = adapter(spatial, motion)

    preds = sign2t5(
        sign_feats     = fused,
        max_new_tokens = MAX_TGT_LEN,
        labels         = None,           # inference
    )

    predictions.extend(preds)
    references.extend(refs)

# If you run on multiple GPUs, gather across processes
predictions = accelerator.gather_for_metrics(predictions)
references  = accelerator.gather_for_metrics(references)

if accelerator.is_main_process:
    metrics = evaluate_results(predictions, references, split="test", tokenizer="13a")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")

accelerator.wait_for_everyone()