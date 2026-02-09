import torch
from typing import Tuple, Optional
import torch.nn.functional as F


def train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Runs one training epoch.

    Returns:
        avg_loss, accuracy
    """
    model.train()
    total_loss = 0.0
    correct = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)

    return avg_loss, accuracy


@torch.no_grad()
def eval_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Runs one evaluation epoch.

    Returns:
        avg_loss, accuracy
    """
    model.eval()
    total_loss = 0.0
    correct = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)

    return avg_loss, accuracy

# --- Knowledge Distillation Functions ---

def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    T: float = 4.0,
    alpha: float = 0.3
) -> torch.Tensor:
    """
    Calculates the KD loss: a weighted average of Soft Target Loss (KL Div)
    and Hard Target Loss (Cross Entropy).
    """
    # Hard loss (Standard Cross Entropy)
    ce = F.cross_entropy(student_logits, targets)

    # Soft loss (KL Divergence)
    # T^2 scaling is used to align gradient magnitudes
    p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits / T, dim=1)
    
    kl = F.kl_div(p_s, p_t, reduction="batchmean")

    return alpha * ce + (1 - alpha) * (T ** 2) * kl


def train_kd_epoch(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    T: float = 4.0,
    alpha: float = 0.3,
) -> Tuple[float, float]:
    """
    Runs one Knowledge Distillation training epoch.
    
    Returns:
        avg_loss, accuracy
    """
    student.train()
    # Teacher must be in eval mode to ensure deterministic output (e.g. no dropout)
    teacher.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        # Get teacher logits (no grad needed)
        with torch.no_grad():
            teacher_logits = teacher(x)

        # Get student logits
        student_logits = student(x)
        
        # Calculate KD loss
        loss = distillation_loss(
            student_logits,
            teacher_logits,
            y,
            T=T,
            alpha=alpha
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        
        # Accuracy based on student predictions
        preds = student_logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / total
    acc = correct / total

    return avg_loss, acc


@torch.no_grad()
def eval_kd_epoch(
    student: torch.nn.Module,
    teacher: Optional[torch.nn.Module],
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    T: float = 4.0,
    alpha: float = 0.3,
) -> Tuple[float, float]:
    """
    Evaluate the student on a dataset.
    If teacher is provided, computes distillation loss.
    If teacher is None, computes standard Cross Entropy loss.
    
    Returns:
        avg_loss, accuracy
    """
    student.eval()
    if teacher is not None:
        teacher.eval()
        
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        student_logits = student(x)

        if teacher is not None:
            teacher_logits = teacher(x)
            loss = distillation_loss(student_logits, teacher_logits, y, T=T, alpha=alpha)
        else:
            loss = F.cross_entropy(student_logits, y)

        total_loss += loss.item() * x.size(0)
        
        preds = student_logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    
    return avg_loss, acc