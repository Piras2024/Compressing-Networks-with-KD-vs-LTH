import torch
import numpy as np

def compute_cka_matrix(
    model_a,
    model_b,
    layers_a: dict,
    layers_b: dict,
    dataloader,
    device,
):
    """
    Computes a CKA similarity matrix between two models.
    """
    model_a.to(device).eval()
    model_b.to(device).eval()

    # ---------- Hook registration ----------
    def register_hooks(model, layer_dict):
        activations = {}

        def hook_fn(name):
            def hook(_, __, output):
                # Detach and move to CPU to avoid GPU OOM
                activations[name] = output.detach().cpu()
            return hook

        handles = []
        modules = dict(model.named_modules())

        for name, module_path in layer_dict.items():
            handles.append(
                modules[module_path].register_forward_hook(hook_fn(name))
            )

        return activations, handles

    acts_a, hooks_a = register_hooks(model_a, layers_a)
    acts_b, hooks_b = register_hooks(model_b, layers_b)

    # ---------- Storage ----------
    storage_a = {k: [] for k in layers_a}
    storage_b = {k: [] for k in layers_b}

    # ---------- Forward passes ----------
    with torch.no_grad():
        for batch in dataloader:
            # Check if dataloader returns (X, y) or just X
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)

            _ = model_a(x)
            _ = model_b(x)

            for k in storage_a:
                storage_a[k].append(acts_a[k])

            for k in storage_b:
                storage_b[k].append(acts_b[k])

    # ---------- Cleanup ----------
    for h in hooks_a + hooks_b:
        h.remove()

    # ---------- Concatenate ----------
    for k in storage_a:
        storage_a[k] = torch.cat(storage_a[k], dim=0)

    for k in storage_b:
        storage_b[k] = torch.cat(storage_b[k], dim=0)

    # ---------- CKA computation ----------
    def flatten(x):
        return x.view(x.size(0), -1)

    def linear_cka(X, Y):
        # Center the features
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)

        # Frobenius norm calculations
        # HSIC for linear kernels is ||X^T Y||_F^2
        hsic = torch.norm(X.T @ Y, p='fro') ** 2
        norm_x = torch.norm(X.T @ X, p='fro')
        norm_y = torch.norm(Y.T @ Y, p='fro')

        return (hsic / (norm_x * norm_y)).item()

    layer_names_a = list(layers_a.keys())
    layer_names_b = list(layers_b.keys())

    cka_matrix = np.zeros((len(layer_names_a), len(layer_names_b)))

    for i, la in enumerate(layer_names_a):
        Xa = flatten(storage_a[la])

        for j, lb in enumerate(layer_names_b):
            Xb = flatten(storage_b[lb])
            cka_matrix[i, j] = linear_cka(Xa, Xb)

    return cka_matrix, layer_names_a, layer_names_b