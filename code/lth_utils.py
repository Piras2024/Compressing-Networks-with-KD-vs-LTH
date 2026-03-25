import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import wandb

def get_prunable_layers(model):
    """
    Returns a list of modules for pruning (Conv1d and Linear).
    """
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            layers.append((module, 'weight'))
    return layers

def train_model(model, device, train_loader, test_loader, optimizer, scheduler, epochs, round_idx):
    """
    Standard training loop with validation, checkpointing, and W&B logging.
    """
    model.train()
    history = {'val_acc': []}
    criterion = nn.CrossEntropyLoss()
    best_test_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        val_acc = 100. * correct / len(test_loader.dataset)
        history['val_acc'].append(val_acc)

        if val_acc > best_test_acc:
            best_test_acc = val_acc
            # Save the best model for this LTH round
            torch.save(model.state_dict(), f'best_lth_model_round_{round_idx}.pth')
        
        # Log to Weights & Biases
        wandb.log({
            f"round_{round_idx}/val_acc": val_acc,
            f"round_{round_idx}/train_loss": total_loss / len(train_loader),
        })
        print(f"Round {round_idx} | Epoch {epoch+1} | Val Acc: {val_acc:.2f}%")
        
    return history

def remove_pruning_reparam(model):
    """
    Makes pruning permanent by removing the 'weight_orig' and 'weight_mask' 
    buffers and applying the mask to the weight.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                pass # Layer wasn't pruned or already removed

def count_sparsity(model):
    """Calculates global percentage of zero-weight parameters."""
    sum_zeros = 0
    sum_elements = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            sum_zeros += torch.sum(module.weight == 0)
            sum_elements += module.weight.nelement()
    
    if sum_elements == 0: return 0.0
    return 100.0 * float(sum_zeros) / float(sum_elements)

def count_active_parameters(model):
    """
    Prints a layer-by-layer breakdown of non-zero parameters and returns the total.
    """
    total_active = 0
    total_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            w = module.weight.data
            active = torch.count_nonzero(w).item()
            total = w.numel()
            
            total_active += active
            total_params += total
            
            print(f"Layer: {name:20} | Active: {active:8} / {total:8} ({100*active/total:.2f}%)")
            
    print("-" * 60)
    print(f"TOTAL ACTIVE PARAMETERS: {total_active:,} / {total_params:,}")
    print(f"GLOBAL DENSITY: {100 * total_active / total_params:.2f}%")
    
    return total_active