# --------------------------------------------------------
# BEATs Knowledge Distillation
# Distill a smaller student model from a pre-trained BEATs teacher.
# Usage:
#   python distill.py \
#       --teacher_ckpt BEATs_iter3_plus_AS2M_finetuned.pt \
#       --data_dir /path/to/audio \
#       --csv_path /path/to/manifest.csv \
#       --student_size medium \
#       --epochs 50
# --------------------------------------------------------

import argparse
import os
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchaudio.transforms as T

from BEATs import BEATs, BEATsConfig
from distill_dataset import AudioDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# 1. Student Config Factory
# ============================================================

def get_student_config(size: str = "medium", num_classes: int = 527) -> BEATsConfig:
    """Return a BEATsConfig for a smaller student model.

    Args:
        size: One of 'small', 'medium', 'large'.
        num_classes: Number of output classes (default 527 for AudioSet).

    Returns:
        A BEATsConfig instance.
    """
    presets = {
        "small": {
            "input_patch_size": 16,
            "embed_dim": 512,
            "conv_bias": False,
            "encoder_layers": 3,
            "encoder_embed_dim": 256,
            "encoder_ffn_embed_dim": 1024,
            "encoder_attention_heads": 4,
            "activation_fn": "gelu",
            "layer_norm_first": False,
            "deep_norm": False,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "activation_dropout": 0.0,
            "encoder_layerdrop": 0.0,
            "dropout_input": 0.0,
            "conv_pos": 128,
            "conv_pos_groups": 16,
            "relative_position_embedding": True,
            "num_buckets": 320,
            "max_distance": 1280,
            "gru_rel_pos": True,
            "finetuned_model": True,
            "predictor_dropout": 0.1,
            "predictor_class": num_classes,
            "layer_wise_gradient_decay_ratio": 1.0,
        },
        "medium": {
            "input_patch_size": 16,
            "embed_dim": 512,
            "conv_bias": False,
            "encoder_layers": 4,
            "encoder_embed_dim": 384,
            "encoder_ffn_embed_dim": 1536,
            "encoder_attention_heads": 6,
            "activation_fn": "gelu",
            "layer_norm_first": False,
            "deep_norm": False,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "activation_dropout": 0.0,
            "encoder_layerdrop": 0.0,
            "dropout_input": 0.0,
            "conv_pos": 128,
            "conv_pos_groups": 16,
            "relative_position_embedding": True,
            "num_buckets": 320,
            "max_distance": 1280,
            "gru_rel_pos": True,
            "finetuned_model": True,
            "predictor_dropout": 0.1,
            "predictor_class": num_classes,
            "layer_wise_gradient_decay_ratio": 1.0,
        },
        "large": {
            "input_patch_size": 16,
            "embed_dim": 512,
            "conv_bias": False,
            "encoder_layers": 6,
            "encoder_embed_dim": 512,
            "encoder_ffn_embed_dim": 2048,
            "encoder_attention_heads": 8,
            "activation_fn": "gelu",
            "layer_norm_first": False,
            "deep_norm": False,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "activation_dropout": 0.0,
            "encoder_layerdrop": 0.0,
            "dropout_input": 0.0,
            "conv_pos": 128,
            "conv_pos_groups": 16,
            "relative_position_embedding": True,
            "num_buckets": 320,
            "max_distance": 1280,
            "gru_rel_pos": True,
            "finetuned_model": True,
            "predictor_dropout": 0.1,
            "predictor_class": num_classes,
            "layer_wise_gradient_decay_ratio": 1.0,
        },
    }

    if size not in presets:
        raise ValueError(f"Unknown student size '{size}'. Choose from: {list(presets.keys())}")

    return BEATsConfig(presets[size])


# ============================================================
# 2. Distillation Loss
# ============================================================

class DistillationLoss(nn.Module):
    """Multi-component knowledge distillation loss for BEATs.

    Components:
        1. Logits MSE loss — MSE between teacher/student raw logits (unactivated).
        2. Hard loss — BCE with ground-truth labels.
        3. Hidden states MSE loss — MSE between projected student/teacher hidden states at multiple layers.

    Args:
        alpha: Weight for logits MSE loss.
        beta: Weight for hard (BCE) loss.
        gamma: Weight for feature hint loss.
        teacher_dim: Teacher hidden dimension (for hint projection).
        student_dim: Student hidden dimension (for hint projection).
        num_student_layers: Number of student layers to align.
        layer_mapping: Dict mapping student_layer_idx -> teacher_layer_idx.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.2,
        gamma: float = 0.3,
        temperature: float = 1.0,
        teacher_dim: int = 768,
        student_dim: int = 384,
        num_student_layers: int = 4,
        layer_mapping: dict = None,
        use_cosine_for_last_layer: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.use_cosine_for_last_layer = use_cosine_for_last_layer

        # Default layer mapping: student[i] -> teacher[(i+1)*3]
        # student[0]->teacher[3], student[1]->teacher[6], student[2]->teacher[9], student[3]->teacher[12]
        if layer_mapping is None:
            self.layer_mapping = {0: 2, 1: 5, 2: 8, 3: 11}
        else:
            self.layer_mapping = layer_mapping

        # Projection layers for each student hidden state to match teacher dimension
        # Use MLP (multi-layer perceptron) instead of single Linear layer for better capacity
        self.hidden_projs = nn.ModuleDict()
        for student_idx in range(num_student_layers):
            proj_name = f"proj_{student_idx}"
            
            # Create a 2-layer MLP with GELU activation
            # student_dim -> hidden_dim -> teacher_dim
            hidden_dim = (student_dim + teacher_dim) // 2  # Middle dimension
            
            mlp = nn.Sequential(
                nn.Linear(student_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, teacher_dim)
            )
            
            # Initialize weights
            # Use Kaiming initialization for better gradient flow with GELU activation
            for module in mlp:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            
            self.hidden_projs[proj_name] = mlp

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_hidden_states: list = None,
        teacher_hidden_states: list = None,
        labels: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            student_logits: (B, C) raw logits from student (before sigmoid)
            teacher_logits: (B, C) raw logits from teacher (before sigmoid)
            student_hidden_states: List of (B, T, D_s) hidden states from selected student layers
            teacher_hidden_states: List of (B, T, D_t) hidden states from corresponding teacher layers
            labels: (B, C) multi-hot ground-truth labels (optional)

        Returns:
            Scalar loss tensor.
        """
        loss = 0.0

        # --- 1. Logits MSE Loss with Temperature Scaling ---
        if self.alpha > 0 and self.temperature > 0:
            # Apply temperature scaling to smooth the distributions
            student_logits_scaled = student_logits / self.temperature
            teacher_logits_scaled = teacher_logits.detach() / self.temperature
            
            # Compute MSE loss on scaled logits
            logits_mse = F.mse_loss(student_logits_scaled, teacher_logits_scaled)
            
            # Scale the loss by T^2 to maintain gradient magnitude (as in Hinton's distillation paper)
            loss = loss + self.alpha * logits_mse * (self.temperature ** 2)
        elif self.alpha > 0:
            # Fallback to regular MSE if temperature is 0 or negative
            logits_mse = F.mse_loss(student_logits, teacher_logits.detach())
            loss = loss + self.alpha * logits_mse

        # --- 2. Hard Label Loss ---
        hard_loss = 0
        loss = loss + self.beta * hard_loss
        # if labels is not None and self.beta > 0:
        #     hard_loss = F.binary_cross_entropy_with_logits(student_logits, labels)
        #     loss = loss + self.beta * hard_loss

        # --- 3. Hidden States MSE Loss (multi-layer) ---
        if (
            student_hidden_states is not None
            and teacher_hidden_states is not None
            and self.gamma > 0
        ):
            hidden_loss = 0.0
            num_aligned = 0

            for student_idx, teacher_idx in self.layer_mapping.items():
                if student_idx < len(student_hidden_states) and teacher_idx < len(teacher_hidden_states):
                    s_h = student_hidden_states[student_idx]
                    t_h = teacher_hidden_states[teacher_idx].detach()

                    # Ensure shapes are (B, T, C)
                    if s_h.dim() == 3:
                        B, T, C = s_h.shape
                    else:
                        raise ValueError(f"Expected student hidden state to be 3D (B, T, C), got {s_h.shape}")

                    # Project student hidden state to teacher dimension
                    proj_name = f"proj_{student_idx}"
                    # Linear layer applies to the last dimension
                    s_h_proj = self.hidden_projs[proj_name](s_h) # Input: (B, T, C_s) -> Output: (B, T, C_t)

                    # Align sequence lengths (take minimum)
                    min_len = min(s_h_proj.size(1), t_h.size(1))
                    s_h_proj = s_h_proj[:, :min_len]
                    t_h = t_h[:, :min_len]

                    # Check if this is the last layer mapping
                    is_last_layer = (student_idx == max(self.layer_mapping.keys()))
                    
                    # Use cosine similarity for last layer, MSE for others
                    if is_last_layer and self.use_cosine_for_last_layer:
                        # Cosine similarity loss: 1 - cosine_similarity
                        # Reshape to (B*T, C) for cosine similarity computation
                        s_flat = s_h_proj.reshape(-1, s_h_proj.size(-1))
                        t_flat = t_h.reshape(-1, t_h.size(-1))
                        
                        # Compute cosine similarity
                        cos_sim = F.cosine_similarity(s_flat, t_flat, dim=-1)  # Shape: (B*T,)
                        
                        # Cosine loss: 1 - mean(cosine_similarity)
                        cos_loss = 1.0 - cos_sim.mean()
                        
                        # Add std constraint to prevent scale explosion
                        # This ensures student and teacher have similar feature scales
                        std_loss = (s_h_proj.std() - t_h.std()).pow(2)
                        
                        # Combined loss: cosine + small weight * std constraint
                        layer_loss = cos_loss + 0.1 * std_loss
                    else:
                        # MSE loss for other layers
                        layer_loss = F.mse_loss(s_h_proj, t_h)
                    
                    hidden_loss += layer_loss
                    num_aligned += 1

            if num_aligned > 0:
                hidden_loss = hidden_loss / num_aligned
                loss = loss + self.gamma * hidden_loss

        return loss


# ============================================================
# 3. Cosine LR Scheduler with Warmup
# ============================================================

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup followed by cosine annealing."""

    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch)
        if step < self.warmup_steps:
            scale = step / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        return [base_lr * scale for base_lr in self.base_lrs]


# ============================================================
# 4. Helper: load teacher model
# ============================================================

def load_teacher(ckpt_path: str, device: torch.device) -> BEATs:
    """Load a fine-tuned BEATs teacher model from a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")

    cfg = BEATsConfig(ckpt["cfg"])
    cfg.finetuned_model = True
    if "predictor_class" not in ckpt["cfg"]:
        cfg.predictor_class = 527

    teacher = BEATs(cfg)
    teacher.load_state_dict(ckpt["model"], strict=False)
    teacher = teacher.to(device)
    teacher.eval()

    # Freeze all parameters
    for p in teacher.parameters():
        p.requires_grad = False

    num_params = sum(p.numel() for p in teacher.parameters()) / 1e6
    logger.info(f"Teacher loaded: {num_params:.1f}M parameters (frozen)")
    return teacher


# ============================================================
# 5. Training Loop
# ============================================================

def train_one_epoch(
    teacher: BEATs,
    student: BEATs,
    loss_fn: DistillationLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    use_specaugment: bool = False,
    specaugment_transform = None,
):
    student.train()
    total_loss = 0.0
    num_batches = 0
    
    logger.info(f"Starting epoch {epoch}, total batches: {len(dataloader)}, dataset size: {len(dataloader.dataset)}")

    for batch_idx, (waveforms, labels) in enumerate(dataloader):
        # Ensure waveforms has a batch dimension
        if waveforms.dim() == 1:
            waveforms = waveforms.unsqueeze(0)
        
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        # --- Teacher forward (no grad) ---
        with torch.no_grad():
            t_fbank = teacher.preprocess(waveforms)
            t_fbank = t_fbank.unsqueeze(1)
            
            # Apply SpecAugment to teacher fbank if enabled
            if use_specaugment and specaugment_transform is not None:
                # SpecAugment expects (batch, channel, freq, time)
                t_fbank_augmented = specaugment_transform(t_fbank)
            else:
                t_fbank_augmented = t_fbank
            
            t_features = teacher.patch_embedding(t_fbank_augmented)
            t_features = t_features.reshape(t_features.shape[0], t_features.shape[1], -1)
            t_features = t_features.transpose(1, 2)
            t_features = teacher.layer_norm(t_features)
            if teacher.post_extract_proj is not None:
                t_features = teacher.post_extract_proj(t_features)
            t_features = teacher.dropout_input(t_features)
            
            # Extract all layer outputs from teacher
            # extract_features returns (final_output, layer_results)
            # We want layer_results which is a list of (T, B, C) tensors
            _, t_hidden_all = teacher.encoder.extract_features(t_features, padding_mask=None)
            
            # print(f"DEBUG: Number of teacher hidden states: {len(t_hidden_all)}")
            
            # Convert hidden states from (T, B, C) to (B, T, C) for loss computation
            t_hidden_states = []
            for h in t_hidden_all:
                h_transposed = h.transpose(0, 1).contiguous()  # (T, B, C) -> (B, T, C)
                # Ensure batch dimension is preserved
                if h_transposed.dim() == 2:
                    # If we lost the batch dim, reshape to (1, T, C)
                    h_transposed = h_transposed.unsqueeze(0)
                t_hidden_states.append(h_transposed)
            
            # Teacher logits (from last layer)
            # t_hidden_all[-1] is (T, B, C). We need to pool over T.
            t_last_hidden = t_hidden_all[-1] # Shape: (T, B, C)
            t_last_hidden_pooled = t_last_hidden.mean(dim=0) # Shape: (B, C)
            
            # Ensure batch dimension is preserved
            if t_last_hidden_pooled.dim() == 1:
                t_last_hidden_pooled = t_last_hidden_pooled.unsqueeze(0)
            
            t_logits = teacher.predictor(t_last_hidden_pooled) # Shape: (B, num_classes)

        # --- Student forward ---
        s_fbank = student.preprocess(waveforms)
        s_fbank = s_fbank.unsqueeze(1)
        
        # Apply SpecAugment to student fbank if enabled
        if use_specaugment and specaugment_transform is not None:
            # SpecAugment expects (batch, channel, freq, time)
            s_fbank_augmented = specaugment_transform(s_fbank)
        else:
            s_fbank_augmented = s_fbank
        
        s_features = student.patch_embedding(s_fbank_augmented)
        # print(f"DEBUG: After patch_embedding: {s_features.shape}")
        s_features = s_features.reshape(s_features.shape[0], s_features.shape[1], -1)
        # print(f"DEBUG: After reshape: {s_features.shape}")
        s_features = s_features.transpose(1, 2) # Now (B, T, C)
        # print(f"DEBUG: After transpose: {s_features.shape}")
        s_features = student.layer_norm(s_features)
        if student.post_extract_proj is not None:
            s_features = student.post_extract_proj(s_features)
            # print(f"DEBUG: After post_extract_proj: {s_features.shape}")
        s_features = student.dropout_input(s_features)
        
        # Extract all layer outputs from student
        # extract_features returns (final_output, layer_results)
        # We want layer_results which is a list of (T, B, C) tensors
        _, s_hidden_all = student.encoder.extract_features(s_features, padding_mask=None)
        
        # print(f"DEBUG: Number of student hidden states: {len(s_hidden_all)}")
        # print(f"DEBUG: s_hidden_all[0] raw shape (T,B,C): {s_hidden_all[0].shape}")
        
        # Convert hidden states from (T, B, C) to (B, T, C) for loss computation
        s_hidden_states = []
        for h in s_hidden_all:
            h_transposed = h.transpose(0, 1).contiguous()  # (T, B, C) -> (B, T, C)
            # print(f"DEBUG: After transpose shape: {h_transposed.shape}")
            # Ensure batch dimension is preserved
            if h_transposed.dim() == 2:
                # If we lost the batch dim, reshape to (1, T, C)
                h_transposed = h_transposed.unsqueeze(0)
            s_hidden_states.append(h_transposed)
        
        # Student logits (from last layer)
        # s_hidden_all[-1] is (T, B, C). We need to pool over T.
        s_last_hidden = s_hidden_all[-1] # Shape: (T, B, C)
        s_last_hidden_pooled = s_last_hidden.mean(dim=0) # Shape: (B, C)
        
        # Ensure batch dimension is preserved
        if s_last_hidden_pooled.dim() == 1:
            s_last_hidden_pooled = s_last_hidden_pooled.unsqueeze(0)
        
        s_logits = student.predictor_dropout(s_last_hidden_pooled)
        s_logits = student.predictor(s_logits) # Shape: (B, num_classes)
        
        # print(f"DEBUG: s_logits shape after predictor: {s_logits.shape}, batch_size: {s_logits.shape[0]}")

        # --- Compute loss ---
        # print(f"DEBUG LOSS INPUT: s_logits={s_logits.shape}, t_logits={t_logits.shape}")
        # print(f"DEBUG LOSS INPUT: s_hidden_states[0]={s_hidden_states[0].shape if s_hidden_states else None}")
        # print(f"DEBUG LOSS INPUT: t_hidden_states[0]={t_hidden_states[0].shape if t_hidden_states else None}")
        # print(f"DEBUG: Number of s_hidden_states={len(s_hidden_states)}, t_hidden_states={len(t_hidden_states)}")
        
        loss = loss_fn(
            student_logits=s_logits,
            teacher_logits=t_logits,
            student_hidden_states=s_hidden_states,
            teacher_hidden_states=t_hidden_states,
            labels=labels,
        )
        
        # Debug: print individual loss components for first batch
        if batch_idx == 0:
            with torch.no_grad():
                logits_mse = F.mse_loss(s_logits, t_logits.detach())
                logger.info(f"DEBUG LOSS COMPONENTS: logits_mse={logits_mse.item():.4f}, alpha={loss_fn.alpha}, temperature={loss_fn.temperature}")
                
                # Print logits statistics
                logger.info(f"DEBUG LOGITS: s_logits mean={s_logits.mean().item():.4f}, std={s_logits.std().item():.4f}, min={s_logits.min().item():.4f}, max={s_logits.max().item():.4f}")
                logger.info(f"DEBUG LOGITS: t_logits mean={t_logits.mean().item():.4f}, std={t_logits.std().item():.4f}, min={t_logits.min().item():.4f}, max={t_logits.max().item():.4f}")
                
                if s_hidden_states and t_hidden_states:
                    # Check all layers hidden state mse
                    total_hidden_loss = 0.0
                    num_layers = 0
                    is_last_layer_idx = max(loss_fn.layer_mapping.keys())
                    
                    for student_idx, teacher_idx in loss_fn.layer_mapping.items():
                        if student_idx < len(s_hidden_states) and teacher_idx < len(t_hidden_states):
                            s_h_proj = loss_fn.hidden_projs[f"proj_{student_idx}"](s_hidden_states[student_idx])
                            t_h = t_hidden_states[teacher_idx].detach()
                            min_len = min(s_h_proj.size(1), t_h.size(1))
                            s_h_aligned = s_h_proj[:, :min_len]
                            t_h_aligned = t_h[:, :min_len]
                            
                            # Use cosine for last layer if enabled
                            if student_idx == is_last_layer_idx and loss_fn.use_cosine_for_last_layer:
                                s_flat = s_h_aligned.reshape(-1, s_h_aligned.size(-1))
                                t_flat = t_h_aligned.reshape(-1, t_h_aligned.size(-1))
                                cos_sim = F.cosine_similarity(s_flat, t_flat, dim=-1)
                                cos_loss = 1.0 - cos_sim.mean()
                                
                                # Std constraint
                                std_loss = (s_h_aligned.std() - t_h_aligned.std()).pow(2)
                                layer_loss = cos_loss + 0.1 * std_loss
                                
                                logger.info(f"DEBUG HIDDEN LAYER {student_idx}->t{teacher_idx}: CosineLoss={cos_loss.item():.4f}, StdLoss={std_loss.item():.4f}, Total={layer_loss.item():.4f}")
                                logger.info(f"  Student std: {s_h_aligned.std().item():.4f}, Teacher std: {t_h_aligned.std().item():.4f}, Diff: {abs(s_h_aligned.std().item() - t_h_aligned.std().item()):.4f}")
                            else:
                                layer_loss = F.mse_loss(s_h_aligned, t_h_aligned)
                                logger.info(f"DEBUG HIDDEN LAYER {student_idx}->t{teacher_idx}: MSE={layer_loss.item():.4f}")
                            
                            # Print per-layer stats
                            logger.info(f"  Student proj: mean={s_h_aligned.mean().item():.4f}, std={s_h_aligned.std().item():.4f}, min={s_h_aligned.min().item():.4f}, max={s_h_aligned.max().item():.4f}")
                            logger.info(f"  Teacher:      mean={t_h_aligned.mean().item():.4f}, std={t_h_aligned.std().item():.4f}, min={t_h_aligned.min().item():.4f}, max={t_h_aligned.max().item():.4f}")
                            
                            total_hidden_loss += layer_loss.item()
                            num_layers += 1
                    
                    if num_layers > 0:
                        avg_hidden_loss = total_hidden_loss / num_layers
                        logger.info(f"DEBUG LOSS COMPONENTS: avg_hidden_loss={avg_hidden_loss:.4f}, gamma={loss_fn.gamma}")
                logger.info(f"DEBUG LOSS COMPONENTS: total_loss={loss.item():.4f}")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % 50 == 0:
            avg = total_loss / num_batches
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} | "
                f"Loss: {avg:.4f} | LR: {lr:.6f}"
            )
    
    # Log epoch summary
    avg_loss = total_loss / max(1, num_batches)
    logger.info(f"Epoch {epoch} completed - Average Loss: {avg_loss:.4f}")

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate(
    teacher: BEATs,
    student: BEATs,
    dataloader: DataLoader,
    device: torch.device,
):
    """Compute mean Average Precision (mAP) on validation set."""
    student.eval()
    all_preds = []
    all_labels = []

    for waveforms, labels in dataloader:
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        lprobs, _ = student.extract_features(waveforms)
        all_preds.append(lprobs.cpu())
        all_labels.append(labels.cpu())

    preds = torch.cat(all_preds, dim=0)  # (N, C)
    labels = torch.cat(all_labels, dim=0)

    # Compute per-class AP
    aps = []
    for c in range(labels.shape[1]):
        if labels[:, c].sum() == 0:
            continue
        # Sort by predicted probability descending
        sorted_indices = preds[:, c].argsort(descending=True)
        sorted_labels = labels[sorted_indices, c]

        tp = sorted_labels.cumsum(0)
        precision = tp / torch.arange(1, len(sorted_labels) + 1, dtype=torch.float)
        ap = (precision * sorted_labels).sum() / sorted_labels.sum()
        aps.append(ap.item())

    mAP = sum(aps) / max(1, len(aps))
    return mAP


# ============================================================
# 6. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="BEATs Knowledge Distillation")
    parser.add_argument(
        "--teacher_ckpt", type=str, required=True,
        help="Path to the fine-tuned BEATs teacher checkpoint (.pt)",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Root directory of audio files",
    )
    parser.add_argument(
        "--csv_path", type=str, default=None,
        help="Path to CSV manifest (audio_path, label_indices). "
             "If not provided, loads all audio from data_dir without labels.",
    )
    parser.add_argument(
        "--student_size", type=str, default="medium",
        choices=["small", "medium", "large"],
        help="Student model size preset (default: medium)",
    )
    parser.add_argument("--num_classes", type=int, default=527)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.5, help="Logits MSE loss weight")
    parser.add_argument("--beta", type=float, default=0.2, help="Hard loss weight")
    parser.add_argument("--gamma", type=float, default=0.3, help="Hidden states MSE loss weight")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for logits distillation (default: 1.0)")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_duration", type=float, default=10.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--save_dir", type=str, default="./distill_checkpoints",
        help="Directory to save student checkpoints",
    )
    parser.add_argument(
        "--log_file", type=str, default="./log.log",
        help="Path to log file (default: ./log.log)",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None,
        help="Path to pretrained student checkpoint (for multi-stage training)",
    )
    parser.add_argument("--device", type=str, default=None)
    
    # SpecAugment parameters
    parser.add_argument(
        "--use_specaugment", action="store_true",
        help="Enable online SpecAugment during training",
    )
    parser.add_argument(
        "--specaugment_freq_mask_param", type=int, default=27,
        help="Frequency mask parameter for SpecAugment (default: 27)",
    )
    parser.add_argument(
        "--specaugment_time_mask_param", type=int, default=100,
        help="Time mask parameter for SpecAugment (default: 100)",
    )
    parser.add_argument(
        "--num_freq_masks", type=int, default=2,
        help="Number of frequency masks (default: 2)",
    )
    parser.add_argument(
        "--num_time_masks", type=int, default=2,
        help="Number of time masks (default: 2)",
    )

    args = parser.parse_args()

    # Setup file logging
    log_file = args.log_file
    
    # Add file handler to logger
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)
    logger.info("BEATs Knowledge Distillation Training")
    logger.info("=" * 80)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # --- Load teacher ---
    teacher = load_teacher(args.teacher_ckpt, device)
    teacher_dim = teacher.cfg.encoder_embed_dim

    # --- Create student ---
    student_cfg = get_student_config(args.student_size, args.num_classes)
    logger.info(f"Student config: embed_dim={student_cfg.embed_dim}, encoder_embed_dim={student_cfg.encoder_embed_dim}")
    student = BEATs(student_cfg).to(device)
    student_dim = student_cfg.encoder_embed_dim
    num_params = sum(p.numel() for p in student.parameters()) / 1e6
    logger.info(
        f"Student ({args.student_size}): {num_params:.1f}M parameters, "
        f"dim={student_dim}, layers={student_cfg.encoder_layers}"
    )
    
    # --- Load student checkpoint if specified (for multi-stage training) ---
    start_epoch = 1
    if args.ckpt_path:
        if os.path.exists(args.ckpt_path):
            logger.info(f"Loading student checkpoint from: {args.ckpt_path}")
            logger.info(f"Checkpoint file size: {os.path.getsize(args.ckpt_path) / (1024**2):.2f} MB")
            
            import time
            load_start = time.time()
            ckpt = torch.load(args.ckpt_path, map_location=device)
            load_time = time.time() - load_start
            logger.info(f"Checkpoint loaded in {load_time:.2f} seconds")
            
            logger.info("Applying state dict to student model...")
            student.load_state_dict(ckpt["model"])
            logger.info("Student model loaded successfully")
            
            # Load projection layers if available
            if "hidden_projs" in ckpt:
                logger.info("Loading projection layers...")
                loss_fn.hidden_projs.load_state_dict(ckpt["hidden_projs"])
                logger.info("Projection layers loaded successfully")
            else:
                logger.warning("No projection layers found in checkpoint, using random initialization")
            
            start_epoch = ckpt.get("epoch", 0) + 1
            logger.info(f"Loaded student from epoch {ckpt.get('epoch', 'N/A')}, mAP: {ckpt.get('mAP', 'N/A'):.4f}")
        else:
            logger.warning(f"Checkpoint not found: {args.ckpt_path}, starting from scratch")

    # --- Dataset & DataLoader ---
    dataset = AudioDataset(
        audio_dir=args.data_dir,
        csv_path=args.csv_path,
        num_classes=args.num_classes,
        sample_rate=16000,
        max_duration=args.max_duration,
        use_specaugment=args.use_specaugment,
        specaugment_freq_mask_param=args.specaugment_freq_mask_param,
        specaugment_time_mask_param=args.specaugment_time_mask_param,
        num_freq_masks=args.num_freq_masks,
        num_time_masks=args.num_time_masks,
    )
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logger.info(f"Dataset: {len(dataset)} total, {train_size} train, {val_size} val")
    logger.info(f"Train loader batches: {len(train_loader)}, Val loader batches: {len(val_loader)}")

    # --- Loss ---
    loss_fn = DistillationLoss(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        temperature=args.temperature,
        teacher_dim=teacher_dim,
        student_dim=student_dim,
        num_student_layers=student_cfg.encoder_layers,
    ).to(device)

    # --- Optimizer & Scheduler ---
    # Collect all trainable params (student + hidden state projection layers)
    params = list(student.parameters())
    params += list(loss_fn.hidden_projs.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineWarmupScheduler(optimizer, args.warmup_steps, total_steps)

    # --- Save dir ---
    os.makedirs(args.save_dir, exist_ok=True)

    # --- Training ---
    # Initialize SpecAugment transform if enabled
    specaugment_transform = None
    if args.use_specaugment:
        specaugment_transform = T.SpecAugment(
            n_freq_masks=args.num_freq_masks,
            freq_mask_param=args.specaugment_freq_mask_param,
            n_time_masks=args.num_time_masks,
            time_mask_param=args.specaugment_time_mask_param,
        ).to(device)
        logger.info(
            f"SpecAugment enabled: freq_mask={args.specaugment_freq_mask_param}, "
            f"time_mask={args.specaugment_time_mask_param}, "
            f"num_freq_masks={args.num_freq_masks}, num_time_masks={args.num_time_masks}"
        )
    
    best_mAP = 0.0
    
    # If resuming, we might want to load the best mAP from checkpoint
    # For now, start from 0 and let it find new best
    
    for epoch in range(start_epoch, args.epochs + 1):
        avg_loss = train_one_epoch(
            teacher, student, loss_fn, train_loader, optimizer, scheduler, device, epoch,
            use_specaugment=args.use_specaugment,
            specaugment_transform=specaugment_transform,
        )
        
        # Only evaluate mAP if alpha > 0 (predictor is being trained)
        if args.alpha > 0:
            mAP = evaluate(teacher, student, val_loader, device)
            logger.info(f"Epoch {epoch}/{args.epochs} — Loss: {avg_loss:.4f}, Val mAP: {mAP:.4f}")
        else:
            mAP = 0.0
            logger.info(f"Epoch {epoch}/{args.epochs} — Loss: {avg_loss:.4f} (mAP skipped, alpha=0)")

        # Save checkpoint for this epoch
        ckpt = {
            "epoch": epoch,
            "model": student.state_dict(),
            "hidden_projs": loss_fn.hidden_projs.state_dict(),  # Save projection layers
            "cfg": student_cfg.__dict__,
            "mAP": mAP,
        }
        
        # Save last checkpoint (overwritten each epoch)
        torch.save(ckpt, os.path.join(args.save_dir, "last.pt"))
        
        # Save epoch-specific checkpoint
        epoch_ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch}.pt")
        torch.save(ckpt, epoch_ckpt_path)
        logger.info(f"  Saved checkpoint: {epoch_ckpt_path}")

        # Save best checkpoint based on mAP
        if mAP > best_mAP:
            best_mAP = mAP
            torch.save(ckpt, os.path.join(args.save_dir, "best.pt"))
            logger.info(f"  ★ New best mAP: {best_mAP:.4f} (saved to best.pt)")

    logger.info(f"Training complete. Best mAP: {best_mAP:.4f}")
    logger.info(f"Checkpoints saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
