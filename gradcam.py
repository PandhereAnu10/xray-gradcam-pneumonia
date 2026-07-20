import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Returns a (H, W) heatmap in [0, 1], resized to match the input."""
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        # Global-average-pool the gradients over spatial dims - per-channel weight
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        weighted_activations = (weights * self.activations).sum(dim=1, keepdim=True)  # (1,1,h,w)
        cam = F.relu(weighted_activations)

        cam = F.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def overlay_heatmap(image_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blends a (H, W) heatmap in [0,1] onto an (H, W, 3) uint8 image using a
    simple red-hot colormap, so we don't need matplotlib's colormap machinery."""
    heat = np.zeros_like(image_uint8, dtype=np.float32)
    heat[..., 0] = cam * 255                      # red channel <- intensity
    heat[..., 1] = np.clip(cam - 0.5, 0, 1) * 2 * 180  # a little green/yellow at peaks
    blended = (1 - alpha) * image_uint8.astype(np.float32) + alpha * heat
    return np.clip(blended, 0, 255).astype(np.uint8)
