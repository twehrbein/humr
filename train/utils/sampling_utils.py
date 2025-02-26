import matplotlib.pyplot as plt
import torch
from typing import Literal
import torch.nn as nn
from einops import rearrange, reduce


_NORMALIZATION = Literal["image_half", "image", "mean_std", "none"]


class HeatmapSampler(nn.Module):

    def __init__(
        self,
        min_threshold: float = -1.0,
        temperature_scaling: float = 1.0,
        return_ordered: bool = False,
        normalize_poses: bool = False,
        normalization_method: _NORMALIZATION = "image_half",
        sample_with_replacement: bool = True,
    ) -> object:
        """

        Parameters
        ----------
        min_threshold
        temperature_scaling
        return_ordered
        normalize_poses
        normalization_method
        sample_with_replacement
        """
        super().__init__()
        self.min_threshold = min_threshold
        self.tau = temperature_scaling
        self.return_ordered = return_ordered
        self.normalize_poses = normalize_poses
        self.normalization_method = normalization_method
        self.sample_with_replacement = sample_with_replacement

    def normalize(self, x, y, heatmap_size=64):
        if self.normalization_method == "mean_std_per_sample":
            # Normalize the poses based on mean and std
            x = x - reduce(x, "b j n -> b () n", reduction="mean")
            y = y - reduce(y, "b j n -> b () n", reduction="mean")

            xy = rearrange([x, y], "d b j n -> b j n d")
            std = reduce(xy, "b j n d -> b () n", reduction=torch.std)

            x = x / std
            y = y / std

        elif self.normalization_method == "mean_std":
            # Normalize the poses based on mean and std over all samples
            x = x - reduce(x, "b j n -> b () ()", reduction="mean")
            y = y - reduce(y, "b j n -> b () ()", reduction="mean")

            xy = rearrange([x, y], "d b j n -> b j n d")
            std = reduce(xy, "b j n d -> b () ()", reduction=torch.std)

            x = x / std
            y = y / std

        elif self.normalization_method == "image_half":
            # Normalize the poses to a fixed interval
            x = (x - 0.5 * heatmap_size) / heatmap_size
            y = (y - 0.5 * heatmap_size) / heatmap_size
        else:
            # Normalize the poses to a fixed interval
            x = 2 * (x - 0.5 * heatmap_size) / heatmap_size
            y = 2 * (y - 0.5 * heatmap_size) / heatmap_size

        return x, y

    def sample(self, heatmap, num_samples=16, return_probs=False):
        heatmap_probs = heatmap.clone()
        # Enforce unlikely position to have zero probability

        if self.min_threshold < 0:
            heatmap_probs[heatmap_probs < 0] = 0
        else:
            heatmap_probs[heatmap_probs < self.min_threshold] = 0.0

        # Apply sharpening of scores
        heatmap_probs = torch.pow(heatmap_probs, self.tau)
        # handle if heatmap is all zeros
        hm_sum = heatmap_probs.sum(dim=(2, 3))
        heatmap_probs[hm_sum <= 0] = 0.001

        b, j, _, _ = heatmap_probs.shape
        heatmap_probs = rearrange(heatmap_probs, "b j w h -> (b j) (w h)")

        # Sample per joint and batch
        # The rows of input do not need to sum to one (in which case we use the values as weights),
        # but must be non-negative, finite and have a non-zero sum.
        samples = torch.multinomial(
            heatmap_probs, num_samples, replacement=self.sample_with_replacement
        )  # (bs * num_joints, num_samples) contains sampled indices to heatmap vector!

        if self.return_ordered or return_probs:  # Sorts the samples in decreasing likelihood order
            sample_probs = torch.gather(heatmap_probs, dim=-1, index=samples)
            if self.return_ordered:
                ordering = torch.argsort(sample_probs, dim=-1, descending=True)
                samples = torch.gather(samples, dim=-1, index=ordering)
                sample_probs = torch.gather(sample_probs, dim=-1, index=ordering)

            sample_probs = rearrange(sample_probs, "(b j) n -> b j n", b=b, j=j, n=num_samples)
            # "unsharpen" probabilities:
            sample_probs = torch.pow(sample_probs, 1 / self.tau)

        samples = rearrange(samples, "(b j) n -> b j n", b=b, j=j, n=num_samples).float()

        # Unravel index (heatmap vector to heatmap matrix!)
        sampled_joint_x = (samples % heatmap.shape[-1]).float()
        sampled_joint_y = torch.div(samples, heatmap.shape[-1], rounding_mode="floor").float()
        if self.normalize_poses:
            sampled_joint_x, sampled_joint_y = self.normalize(
                sampled_joint_x, sampled_joint_y, heatmap.shape[-1]
            )
        sampled_joint = torch.stack((sampled_joint_x, sampled_joint_y), dim=-2)
        sampled_joint = rearrange(
            sampled_joint, "b j d n -> b n (j d)", b=b, j=j, n=num_samples, d=2
        )

        if not return_probs:
            return sampled_joint  # Shape: B x N_samples x 2*N_joints
        else:
            sample_probs = rearrange(sample_probs, "b j n -> b n j", b=b, j=j, n=num_samples)
            return sampled_joint, sample_probs


def vertex_diversity_from_samples(vertices_samples):
    # (S, 6890, 3)
    directional_vertex_stddev = torch.sqrt(torch.var(vertices_samples, dim=0, correction=0))

    per_vertex_mean_stddev = directional_vertex_stddev.mean(dim=-1)
    vertex_uncertainty_norm = plt.Normalize(vmin=0.0, vmax=0.15, clip=True)
    vertex_uncertainty_colours = plt.cm.jet(
        vertex_uncertainty_norm(per_vertex_mean_stddev.cpu().numpy())
    )[:, :3]
    vertex_uncertainty_colours = (
        torch.from_numpy(vertex_uncertainty_colours[None]).to(vertices_samples.device).float()
    )
    return directional_vertex_stddev, vertex_uncertainty_colours


def expand_annotation(annot, num_samples):
    # annot: (B, ...) to (B * num_samples, ...)
    orig_shape = annot.shape
    batch_size = orig_shape[0]
    ndims = annot.ndim
    expand_dims = [-1, num_samples]

    annot = annot.unsqueeze(1)
    expand_dims.extend([-1] * (ndims - 1))
    return annot.expand(expand_dims).reshape(batch_size * num_samples, *orig_shape[1:])
