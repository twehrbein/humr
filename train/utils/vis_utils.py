import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from train.core.constants import COCO_17, COCO17_BONES
from train.utils.humaniflow_renderer import TexturedIUVRenderer, render_samples_visualization


def scatter_joint_hypos(img_np, pred_kpts, gt_kpts, nll, out_name, show=False, suptitle=None):
    # pred_kpts: (num_hypos, 17, 2), gt_kpts: (17, 2), nll: (#samples,)
    # normalize color between min amd max log_prob
    log_prob = nll * -1
    min_ll = log_prob.min()
    max_ll = log_prob.max()
    color = (log_prob - min_ll) / (max_ll - min_ll)

    def add_subplot(axs, row_idx, col_idx, joint_idx):
        x = pred_kpts[:, joint_idx, 0]
        y = pred_kpts[:, joint_idx, 1]
        axs[row_idx, col_idx].imshow(img_np)
        axs[row_idx, col_idx].set_title(f"{COCO_17[joint_idx]}")
        axs[row_idx, col_idx].scatter(x, y, c=color, s=5, cmap="viridis")
        axs[row_idx, col_idx].scatter(gt_kpts[joint_idx, 0], gt_kpts[joint_idx, 1], c="r")
        axs[row_idx, col_idx].axis("off")

    fig, axs = plt.subplots(3, 6, figsize=(22, 13))
    axs[0, 0].imshow(img_np)
    axs[0, 0].axis("off")
    for joint_idx in range(17):
        row_idx = (joint_idx + 1) // 6
        col_idx = (joint_idx + 1) % 6
        add_subplot(axs, row_idx, col_idx, joint_idx)
    if suptitle is not None:
        plt.suptitle(suptitle)
    fig.tight_layout()
    plt.axis("off")
    if show:
        plt.show()
    else:
        num_samples = pred_kpts.shape[0]
        plt.savefig(f"{out_name}_scatter_{num_samples}.jpg")
    plt.close()


def plot_multiple_hypos(img_np, pred_kpts, gt_kpts, log_prob, out_filename, sorting_metric):
    B = COCO17_BONES

    def add_subplot(axs, row_idx, col_idx, sample_idx):
        x = pred_kpts[sample_idx, :, 0]
        y = pred_kpts[sample_idx, :, 1]
        axs[row_idx, col_idx].imshow(img_np)
        axs[row_idx, col_idx].scatter(x, y)
        for id_b in range(len(B)):
            axs[row_idx, col_idx].plot(x[B[id_b]], y[B[id_b]], lw=1.5)
        axs[row_idx, col_idx].axis("off")
        axs[row_idx, col_idx].set_title(f"log_prob {log_prob[sample_idx]:.2f}")

    fig, axs = plt.subplots(3, 5, figsize=(22, 13))
    axs[0, 0].imshow(img_np)
    axs[0, 0].axis("off")
    # plot gt
    axs[0, 1].imshow(img_np)
    axs[0, 1].scatter(gt_kpts[:, 0], gt_kpts[:, 1], c="r")
    for id_b in range(len(B)):
        axs[0, 1].plot(gt_kpts[B[id_b], 0], gt_kpts[B[id_b], 1], lw=1.5)
    axs[0, 1].set_title("gt")
    axs[0, 1].axis("off")
    # plot z0 pred
    axs[0, 2].imshow(img_np)
    x = pred_kpts[0, :, 0]
    y = pred_kpts[0, :, 1]
    axs[0, 2].scatter(x, y, c="b")
    for id_b in range(len(B)):
        axs[0, 2].plot(x[B[id_b]], y[B[id_b]], lw=1.5)
    axs[0, 2].set_title(f"z0, log_prob {log_prob[0]:.2f}")
    axs[0, 2].axis("off")
    # plot best hypo
    best_idx = np.argmin(sorting_metric)
    worst_idx = np.argmax(sorting_metric)
    axs[0, 3].imshow(img_np)
    x = pred_kpts[best_idx, :, 0]
    y = pred_kpts[best_idx, :, 1]
    axs[0, 3].scatter(x, y, c="b")
    for id_b in range(len(B)):
        axs[0, 3].plot(x[B[id_b]], y[B[id_b]], lw=1.5)
    axs[0, 3].set_title(f"best, log_prob {log_prob[best_idx]:.2f}")
    axs[0, 3].axis("off")
    # plot worst hypo
    axs[0, 4].imshow(img_np)
    x = pred_kpts[worst_idx, :, 0]
    y = pred_kpts[worst_idx, :, 1]
    axs[0, 4].scatter(x, y, c="b")
    for id_b in range(len(B)):
        axs[0, 4].plot(x[B[id_b]], y[B[id_b]], lw=1.5)
    axs[0, 4].set_title(f"worst, log_prob {log_prob[worst_idx]:.2f}")
    axs[0, 4].axis("off")
    # scatter gt
    for hm_idx in range(5, 15):
        row_idx = hm_idx // 5
        col_idx = hm_idx % 5
        add_subplot(axs, row_idx, col_idx, hm_idx)
    fig.tight_layout()
    plt.axis("off")
    # plt.show()
    plt.savefig(out_filename + "_poses.jpg")
    plt.close()


def viz_mesh_xyz_std_dev(
    img,
    imgname,
    pred_vertices2D_mode,
    directional_vertex_stddev,
    num_samples,
    out_filename,
    max_std_dev=0.15,
):
    if out_filename[:-3] != "jpg" and out_filename[:-3] != "png":
        out_filename = out_filename + "_xyz-stddev.jpg"
    # plt.style.use('dark_background')
    titles = [
        "Input Image",
        "X-axis (Horizontal) Variance",
        "Y-axis (Vertical) Variance",
        "Z-axis (Depth) Variance",
    ]
    plt.figure(figsize=(22, 9))
    plt.suptitle(
        f"Mode prediction with sample (N={num_samples}) "
        f"std dev color clipped at {max_std_dev} meters.\n"
        f"Image: {imgname}"
    )
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.gca().axis("off")
        plt.title(titles[i])
        plt.imshow(img)
        if i > 0:
            plt.scatter(
                pred_vertices2D_mode[:, 0],
                pred_vertices2D_mode[:, 1],
                s=0.25,
                c=directional_vertex_stddev[:, i - 1],
                cmap="jet",
                norm=plt.Normalize(vmin=0.0, vmax=max_std_dev, clip=True),
            )
    plt.tight_layout()
    plt.savefig(out_filename)
    plt.close()


def viz_3d_samples(
    vertices_samples,
    pred_cam_t,
    focal_length,
    np_img,
    vertex_uncertainty_colours,
    out_filename,
    log_prob,
    center,
    scale,
    sorting_metric,
    img_res,
):
    orig_shape = np.array(np_img.shape)[:2]
    device = vertices_samples.device

    # rotate mesh around y axis (#hypos, N, 3)
    # first mean center xyz then rotate, then reshift
    verts_center = vertices_samples.mean(dim=1)  # (#hypos, 3)
    vertices_rot90_samples = vertices_samples - verts_center[:, None]
    theta = -90 * np.pi / 180
    rotation_matrix = torch.tensor(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        device=device,
        dtype=torch.float32,
    )
    vertices_rot90_samples = torch.matmul(vertices_rot90_samples, rotation_matrix.T)
    vertices_rot90_samples = vertices_rot90_samples + verts_center[:, None]

    pred_vertices_samples_all_rot = {"0": vertices_samples, "90": vertices_rot90_samples}

    # transform np_img to pytorch and resize
    img = torch.from_numpy(np_img).to(device).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)

    # Setting up RGB renderer for visualisation
    body_vis_renderer = TexturedIUVRenderer(
        device="cuda",
        batch_size=1,
        img_shape=orig_shape,
        projection_type="perspective",
        perspective_focal_length=focal_length,
        render_rgb=True,
        bin_size=-1,
    )

    samples_fig = render_samples_visualization(
        renderer=body_vis_renderer,
        num_vis_samples=9,
        samples_rows=3,
        samples_cols=6,
        visualise_wh=img_res,
        cropped_rgb_for_vis=img,
        pred_vertices_samples_all_rot=pred_vertices_samples_all_rot,
        vertex_colours=vertex_uncertainty_colours,
        cam_t=pred_cam_t[0:1],
        log_prob=log_prob,
        center=center,
        scale=scale,
        sorting_metric=sorting_metric,
    )
    # resize to larger res if width smaller than 2k?
    if samples_fig.shape[1] < 2200:
        h, w = samples_fig.shape[:2]
        ratio = 2200 / w
        samples_fig = cv2.resize(samples_fig, (2200, int(h * ratio)))
    if out_filename[:-3] != "jpg" and out_filename[:-3] != "png":
        out_filename = out_filename + "_samples.jpg"
    cv2.imwrite(out_filename, samples_fig[:, :, ::-1] * 255)
