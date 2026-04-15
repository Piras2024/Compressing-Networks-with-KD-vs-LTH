"""
Run the LLFC_CKA experiment and save plots to disk.
Usage: python run_llfc_cka.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy

from models import BigCNN, StudentCNN
from lth_utils import get_prunable_layers
from training import eval_epoch
from mnist1d_dataset import get_mnist1d_loaders

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

train_loader, test_loader = get_mnist1d_loaders()
criterion = nn.CrossEntropyLoss()

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def extract_features(model, layer_path: str, dataloader, device) -> torch.Tensor:
    model.eval()
    model.to(device)
    collected = []

    def hook(_, __, output):
        collected.append(output.detach().cpu())

    modules = dict(model.named_modules())
    handle = modules[layer_path].register_forward_hook(hook)

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            model(x.to(device))

    handle.remove()
    feats = torch.cat(collected, dim=0)
    return feats.view(feats.size(0), -1)


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    hsic   = torch.norm(X.T @ Y, p='fro') ** 2
    norm_x = torch.norm(X.T @ X, p='fro')
    norm_y = torch.norm(Y.T @ Y, p='fro')
    if norm_x == 0 or norm_y == 0:
        return float('nan')
    return (hsic / (norm_x * norm_y)).item()


def recompute_bn_stats(model, data_loader, device, num_batches: int = 50):
    model.train()
    with torch.no_grad():
        for i, (x, _) in enumerate(data_loader):
            if i >= num_batches:
                break
            model(x.to(device))
    model.eval()


def extract_features_and_labels(model, layer_path: str, dataloader, device):
    """Like extract_features but also returns all ground-truth labels."""
    model.eval()
    model.to(device)
    collected, labels_list = [], []

    def hook(_, __, output):
        collected.append(output.detach().cpu())

    modules = dict(model.named_modules())
    handle = modules[layer_path].register_forward_hook(hook)

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch[0], batch[1]
            model(x.to(device))
            labels_list.append(y)

    handle.remove()
    feats = torch.cat(collected, dim=0).view(len(torch.cat(labels_list)), -1)
    labels = torch.cat(labels_list, dim=0)
    return feats, labels


def llfc_error_from_features(theta_alpha, h_lin: torch.Tensor,
                              labels: torch.Tensor, device: torch.device,
                              penultimate_idx: int = 1) -> float:
    """
    Pass linearly-interpolated features h_lin through the classifier tail of
    theta_alpha (everything after classifier[penultimate_idx]) and return error rate.

    For both BigCNN and StudentCNN the penultimate layer is classifier.1 (idx=1),
    so the tail is classifier[2:].
    """
    tail = nn.Sequential(*list(theta_alpha.classifier.children())[penultimate_idx + 1:])
    tail.eval().to(device)

    with torch.no_grad():
        logits = tail(h_lin.to(device))
        preds  = logits.argmax(dim=1).cpu()

    error = (preds != labels).float().mean().item()
    return error


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def load_pruned_bigcnn(ckpt_path: str, device) -> BigCNN:
    model = BigCNN().to(device)
    for m, p in get_prunable_layers(model):
        prune.identity(m, p)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def load_student(ckpt_path: str, device) -> StudentCNN:
    model = StudentCNN().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------
def interpolate_pruned_bigcnn(model_a, model_b, alpha: float, device) -> BigCNN:
    interp = BigCNN().to(device)
    for m, p in get_prunable_layers(interp):
        prune.identity(m, p)
    sd = {}
    for (k, va), (_, vb) in zip(model_a.state_dict().items(), model_b.state_dict().items()):
        sd[k] = (1 - alpha) * va + alpha * vb
    interp.load_state_dict(sd)
    interp.eval()
    return interp


def interpolate_student(model_a, model_b, alpha: float, device) -> StudentCNN:
    interp = StudentCNN().to(device)
    sd = {}
    for (k, va), (_, vb) in zip(model_a.state_dict().items(), model_b.state_dict().items()):
        sd[k] = (1 - alpha) * va + alpha * vb
    interp.load_state_dict(sd)
    interp.eval()
    return interp


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------
def run_llfc_cka_experiment(model_a, model_b, interpolate_fn, layer_path,
                             alphas, dataloader, train_loader, device, label="",
                             penultimate_idx: int = 1):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Layer: {layer_path}  |  #alphas: {len(alphas)}")
    print(f"{'='*60}")

    print("Extracting endpoint features h_A, h_B and labels ...")
    h_A, labels = extract_features_and_labels(model_a, layer_path, dataloader, device)
    h_B, _      = extract_features_and_labels(model_b, layer_path, dataloader, device)
    print(f"  h_A: {h_A.shape}  |  h_B: {h_B.shape}  |  labels: {labels.shape}")

    results = dict(alphas=list(alphas), llfc_cka=[], cka_vs_A=[],
                   lmc_errors=[], llfc_errors=[])

    for alpha in alphas:
        theta_alpha = interpolate_fn(model_a, model_b, alpha, device)
        recompute_bn_stats(theta_alpha, train_loader, device)

        h_alpha     = extract_features(theta_alpha, layer_path, dataloader, device)
        h_lin_alpha = (1 - alpha) * h_A + alpha * h_B

        cka_llfc = linear_cka(h_alpha, h_lin_alpha)
        cka_a    = linear_cka(h_alpha, h_A)

        # LMC error: full forward pass through weight-interpolated model
        _, acc_lmc = eval_epoch(theta_alpha, dataloader, criterion, device=device)
        lmc_error  = 1.0 - acc_lmc

        # LLFC error: linearly-interpolated features → θ(α)'s classifier tail
        llfc_error = llfc_error_from_features(
            theta_alpha, h_lin_alpha, labels, device, penultimate_idx
        )

        results['llfc_cka'].append(cka_llfc)
        results['cka_vs_A'].append(cka_a)
        results['lmc_errors'].append(lmc_error)
        results['llfc_errors'].append(llfc_error)

        print(f"  α={alpha:.2f}  LLFC_CKA={cka_llfc:.4f}  "
              f"CKA(h_α,h_A)={cka_a:.4f}  "
              f"LMC_err={lmc_error:.4f}  LLFC_err={llfc_error:.4f}")

    lmc_endpoint = max(results['lmc_errors'][0], results['lmc_errors'][-1])
    lmc_barrier  = max(results['lmc_errors']) - lmc_endpoint
    llfc_endpoint = max(results['llfc_errors'][0], results['llfc_errors'][-1])
    llfc_barrier  = max(results['llfc_errors']) - llfc_endpoint

    print(f"\n  LMC error barrier:   {lmc_barrier:.4f}")
    print(f"  LLFC error barrier:  {llfc_barrier:.4f}")
    print(f"  Mean LLFC_CKA:       {np.mean(results['llfc_cka']):.4f}")
    print(f"  LLFC_CKA @ α=0.5:   {results['llfc_cka'][len(alphas)//2]:.4f}")
    results['lmc_barrier']  = lmc_barrier
    results['llfc_barrier'] = llfc_barrier
    # keep backward-compat key
    results['errors']        = results['lmc_errors']
    results['error_barrier'] = lmc_barrier
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def save_individual_plot(results, label: str, filename: str):
    alphas      = results['alphas']
    llfc_cka    = results['llfc_cka']
    cka_a       = results['cka_vs_A']
    lmc_errors  = results['lmc_errors']
    llfc_errors = results['llfc_errors']
    lmc_bar     = results['lmc_barrier']
    llfc_bar    = results['llfc_barrier']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"LLFC_CKA — {label}", fontsize=13, fontweight='bold')

    # --- Left: CKA curves ---
    ax1.plot(alphas, llfc_cka, marker='o', lw=2, color='tab:green',
             label='CKA(h_α , h_lin(α))  [LLFC_CKA]')
    ax1.plot(alphas, cka_a,    marker='s', lw=2, linestyle='--', color='tab:orange',
             label='CKA(h_α , h_A)')
    ax1.axhline(1.0, color='gray', lw=0.8, linestyle=':')
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.05)
    ax1.set_xlabel('α', fontsize=12); ax1.set_ylabel('CKA', fontsize=12)
    ax1.set_title('LLFC_CKA along interpolation path', fontsize=11)
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
    for x, y in zip(alphas, llfc_cka):
        ax1.annotate(f'{y:.2f}', xy=(x, y), xytext=(0, 7),
                     textcoords='offset points', ha='center', fontsize=8, color='tab:green')

    # --- Right: LMC error vs LLFC error ---
    y_top = max(max(lmc_errors), max(llfc_errors)) * 1.4 + 0.01
    ax2.plot(alphas, lmc_errors,  marker='o', lw=2, color='tab:blue',
             label=f'LMC error  (barrier={lmc_bar:.3f})')
    ax2.plot(alphas, llfc_errors, marker='s', lw=2, linestyle='--', color='tab:red',
             label=f'LLFC error (barrier={llfc_bar:.3f})')
    ax2.set_xlim(0, 1); ax2.set_ylim(0, y_top)
    ax2.set_xlabel('α', fontsize=12); ax2.set_ylabel('Test error rate', fontsize=12)
    ax2.set_title('LMC vs LLFC error along interpolation path', fontsize=11)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    for x, y in zip(alphas, lmc_errors):
        ax2.annotate(f'{y:.3f}', xy=(x, y), xytext=(0, 7),
                     textcoords='offset points', ha='center', fontsize=8, color='tab:blue')
    for x, y in zip(alphas, llfc_errors):
        ax2.annotate(f'{y:.3f}', xy=(x, y), xytext=(0, -13),
                     textcoords='offset points', ha='center', fontsize=8, color='tab:red')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def save_summary_plot(all_results, filename: str):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle("LLFC_CKA experiment — summary", fontsize=14, fontweight='bold')

    titles = [
        "Matching Tickets\n(LMC ✓ expected)",
        "LTH Tickets (indep.)\n(LMC ✗ expected)",
        "KD Students (indep.)\n(LMC ✗ expected)",
    ]

    for col, ((res, lbl), title) in enumerate(zip(all_results, titles)):
        # --- top row: LLFC_CKA ---
        ax_top = axes[0, col]
        ax_top.plot(res['alphas'], res['llfc_cka'], marker='o', lw=2,
                    color='tab:green', label='LLFC_CKA')
        ax_top.plot(res['alphas'], res['cka_vs_A'], marker='s', lw=2,
                    linestyle='--', color='tab:orange', label='CKA(h_α, h_A)')
        ax_top.set_ylim(0, 1.05); ax_top.set_xlim(0, 1)
        ax_top.set_title(title, fontsize=10)
        if col == 0:
            ax_top.set_ylabel('CKA', fontsize=11)
        ax_top.grid(True, alpha=0.3); ax_top.legend(fontsize=8)
        for x, y in zip(res['alphas'], res['llfc_cka']):
            ax_top.annotate(f'{y:.2f}', xy=(x, y), xytext=(0, 6),
                            textcoords='offset points', ha='center',
                            fontsize=7, color='tab:green')

        # --- bottom row: LMC error vs LLFC error ---
        ax_bot = axes[1, col]
        y_top = max(max(res['lmc_errors']), max(res['llfc_errors'])) * 1.4 + 0.01
        ax_bot.plot(res['alphas'], res['lmc_errors'],  marker='o', lw=2,
                    color='tab:blue',
                    label=f"LMC  (bar={res['lmc_barrier']:.3f})")
        ax_bot.plot(res['alphas'], res['llfc_errors'], marker='s', lw=2,
                    linestyle='--', color='tab:red',
                    label=f"LLFC (bar={res['llfc_barrier']:.3f})")
        ax_bot.set_ylim(0, y_top)
        ax_bot.set_xlim(0, 1); ax_bot.set_xlabel('α')
        if col == 0:
            ax_bot.set_ylabel('Test error rate', fontsize=11)
        ax_bot.grid(True, alpha=0.3); ax_bot.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
OUT = os.path.dirname(__file__)
alphas = np.linspace(0.0, 1.0, 11)
BIGCNN_LAYER  = "classifier.1"
STUDENT_LAYER = "classifier.1"

# --- Experiment 1: Matching tickets ---
print("\n>>> Loading matching tickets ...")
matching_A = load_pruned_bigcnn("models/matching/matching_model_run0.pth", device)
matching_B = load_pruned_bigcnn("models/matching/matching_model_run1.pth", device)
_, acc = eval_epoch(matching_A, test_loader, criterion, device=device)
print(f"  Matching A acc: {acc:.4f}")
_, acc = eval_epoch(matching_B, test_loader, criterion, device=device)
print(f"  Matching B acc: {acc:.4f}")

results_matching = run_llfc_cka_experiment(
    matching_A, matching_B, interpolate_pruned_bigcnn,
    BIGCNN_LAYER, alphas, test_loader, train_loader, device,
    label="Matching Tickets",
)
save_individual_plot(results_matching, "Matching Tickets",
                     os.path.join(OUT, "llfc_cka_matching.png"))

# --- Experiment 2: Regular LTH tickets ---
print("\n>>> Loading LTH tickets ...")
ticket_A = load_pruned_bigcnn("models/lth/ticket_model_run0.pth", device)
ticket_B = load_pruned_bigcnn("models/lth/ticket_model_run1.pth", device)
_, acc = eval_epoch(ticket_A, test_loader, criterion, device=device)
print(f"  Ticket A acc: {acc:.4f}")
_, acc = eval_epoch(ticket_B, test_loader, criterion, device=device)
print(f"  Ticket B acc: {acc:.4f}")

results_lth = run_llfc_cka_experiment(
    ticket_A, ticket_B, interpolate_pruned_bigcnn,
    BIGCNN_LAYER, alphas, test_loader, train_loader, device,
    label="LTH Tickets (independent)",
)
save_individual_plot(results_lth, "LTH Tickets (independent)",
                     os.path.join(OUT, "llfc_cka_lth.png"))

# --- Experiment 3: KD students ---
print("\n>>> Loading KD students ...")
student_A = load_student("models/best_kd_student_0.pth", device)
student_B = load_student("models/best_kd_student_1.pth", device)
_, acc = eval_epoch(student_A, test_loader, criterion, device=device)
print(f"  Student A acc: {acc:.4f}")
_, acc = eval_epoch(student_B, test_loader, criterion, device=device)
print(f"  Student B acc: {acc:.4f}")

results_students = run_llfc_cka_experiment(
    student_A, student_B, interpolate_student,
    STUDENT_LAYER, alphas, test_loader, train_loader, device,
    label="KD Students (independent)",
)
save_individual_plot(results_students, "KD Students (independent)",
                     os.path.join(OUT, "llfc_cka_students.png"))

# --- Summary plot ---
print("\n>>> Saving summary plot ...")
all_results = [
    (results_matching, "Matching Tickets"),
    (results_lth,      "LTH Tickets (indep.)"),
    (results_students, "KD Students (indep.)"),
]
save_summary_plot(all_results, os.path.join(OUT, "llfc_cka_summary.png"))

# --- Numerical table ---
print(f"\n{'Model pair':<30} {'LMC barrier':>12} {'LLFC barrier':>13} {'mean LLFC_CKA':>14} {'LLFC_CKA @0.5':>14}")
print("-" * 85)
for res, lbl in all_results:
    mid = res['llfc_cka'][len(alphas) // 2]
    print(f"{lbl:<30} {res['lmc_barrier']:>12.4f} {res['llfc_barrier']:>13.4f} "
          f"{np.mean(res['llfc_cka']):>14.4f} {mid:>14.4f}")

print("\nDone.")
