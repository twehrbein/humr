import numpy as np
import torch
from scipy.io import loadmat
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    OrthographicCameras,
    PointLights,
    RasterizationSettings,
    MeshRasterizer,
    HardPhongShader,
    TexturesUV,
    TexturesVertex,
    BlendParams,
)
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import cv2
import os
from train.utils.image_utils import crop


def batch_add_rgb_background(backgrounds, rgb, seg):
    """
    :param backgrounds: (bs, 3, wh, wh)
    :param rgb: (bs, 3, wh, wh)
    :param seg: (bs, wh, wh)
    :return: rgb_with_background: (bs, 3, wh, wh)
    """
    background_pixels = (
        seg[:, None, :, :] == 0
    )  # Body pixels are > 0 and out of frame pixels are -1
    rgb_with_background = (
        rgb * (torch.logical_not(background_pixels)) + backgrounds * background_pixels
    )
    return rgb_with_background


def render_samples_visualization(
    renderer,
    num_vis_samples,
    samples_rows,
    samples_cols,
    visualise_wh,
    cropped_rgb_for_vis,
    pred_vertices_samples_all_rot,
    vertex_colours,
    cam_t,
    log_prob,
    center,
    scale,
    sorting_metric,
):
    samples_fig = np.zeros(
        (samples_rows * visualise_wh, samples_cols * visualise_wh, 3), dtype=np.float32
    )
    cam_t_zoom_out = cam_t.clone()  # zoom out to see more of the bodies
    cam_t_zoom_out[:, 2] *= 1.3
    for i in range(num_vis_samples):
        idx = i
        if sorting_metric is not None:
            if i == 1:
                # plot best hypo
                idx = np.argmin(sorting_metric)
            elif i == 2:
                # plot worst hypo
                idx = np.argmax(sorting_metric)
        body_vis_output_sample = renderer(
            vertices=pred_vertices_samples_all_rot["0"][[idx]],
            cam_t=cam_t,
            verts_features=vertex_colours,
        )
        body_vis_rgb_sample = batch_add_rgb_background(
            backgrounds=cropped_rgb_for_vis,
            rgb=body_vis_output_sample["rgb_images"].permute(0, 3, 1, 2).contiguous(),
            seg=body_vis_output_sample["iuv_images"][:, :, :, 0].round(),
        )
        body_vis_rgb_sample = body_vis_rgb_sample.cpu().detach().numpy()[0].transpose(1, 2, 0)

        body_vis_rgb_rot90_sample = (
            renderer(
                vertices=pred_vertices_samples_all_rot["90"][[idx]],
                cam_t=cam_t_zoom_out,
                verts_features=vertex_colours,
            )["rgb_images"]
            .cpu()
            .detach()
            .numpy()[0]
        )
        body_vis_rgb_sample = crop(body_vis_rgb_sample, center, scale, [visualise_wh, visualise_wh])
        body_vis_rgb_rot90_sample = crop(
            body_vis_rgb_rot90_sample, center, scale, [visualise_wh, visualise_wh]
        )
        # need to convert numpy array img to PIL image:
        pil_image = np.rint(body_vis_rgb_rot90_sample * 255).astype("uint8")
        pil_image = Image.fromarray(pil_image, "RGB")
        # load provided cv2 font
        font_path = os.path.join(cv2.__path__[0], "qt", "fonts", "DejaVuSans.ttf")
        font = ImageFont.truetype(font_path, size=12)
        draw = ImageDraw.Draw(pil_image)
        if i == 0:
            msg = f"z0, log prob: {log_prob[idx].item():.2f}"
        elif i == 1 and sorting_metric is not None:
            msg = f"best, log prob: {log_prob[idx].item():.2f}"
        elif i == 2 and sorting_metric is not None:
            msg = f"worst, log prob: {log_prob[idx].item():.2f}"
        else:
            msg = f"log prob: {log_prob[idx].item():.2f}"
        draw.text((3, 3), msg, (255, 255, 255), font=font)  # Coordinates  # Text  # Color
        body_vis_rgb_rot90_sample = np.array(pil_image, dtype=np.float32) / 255.0

        row = (2 * i) // samples_cols
        col = (2 * i) % samples_cols
        samples_fig[
            row * visualise_wh : (row + 1) * visualise_wh,
            col * visualise_wh : (col + 1) * visualise_wh,
        ] = body_vis_rgb_sample

        row = (2 * i + 1) // samples_cols
        col = (2 * i + 1) % samples_cols
        samples_fig[
            row * visualise_wh : (row + 1) * visualise_wh,
            col * visualise_wh : (col + 1) * visualise_wh,
        ] = body_vis_rgb_rot90_sample

    return samples_fig


class TexturedIUVRenderer(torch.nn.Module):
    def __init__(
        self,
        device,
        batch_size,
        img_shape=256,
        cam_t=None,
        cam_R=None,
        projection_type="perspective",
        perspective_focal_length=300,
        orthographic_scale=0.9,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=None,
        max_faces_per_bin=None,
        perspective_correct=False,
        cull_backfaces=False,
        clip_barycentric_coords=None,
        render_rgb=False,
        light_t=((0.0, -0.8, -2.0),),
        light_ambient_color=((0.5, 0.5, 0.5),),
        light_diffuse_color=((0.3, 0.3, 0.3),),
        light_specular_color=((0.0, 0.0, 0.0),),
        background_color=(0.0, 0.0, 0.0),
    ):
        """
        :param img_shape: (2, ) Size of rendered image.
        :param blur_radius: Float distance in the range [0, 2] used to expand the face
            bounding boxes for rasterization. Setting blur radius
            results in blurred edges around the shape instead of a
            hard boundary.
            Set to 0 (no blur) if rendering for visualisation purposes.
        :param faces_per_pixel: Number of faces to save per pixel, returning
            the nearest faces_per_pixel points along the z-axis.
            Set to 1 if rendering for visualisation purposes.
        :param bin_size: Size of bins to use for coarse-to-fine rasterization (i.e
            breaking image into tiles with size=bin_size before rasterising?).
            Setting bin_size=0 uses naive rasterization; setting bin_size=None
            attempts to set it heuristically based on the shape of the input (i.e. image_size).
            This should not affect the output, but can affect the speed of the forward pass.
            Heuristic based formula maps image_size -> bin_size as follows:
                image_size < 64 -> 8
                16 < image_size < 256 -> 16
                256 < image_size < 512 -> 32
                512 < image_size < 1024 -> 64
                1024 < image_size < 2048 -> 128
        :param max_faces_per_bin: Only applicable when using coarse-to-fine rasterization
            (bin_size > 0); this is the maxiumum number of faces allowed within each
            bin. If more than this many faces actually fall into a bin, an error
            will be raised. This should not affect the output values, but can affect
            the memory usage in the forward pass.
            Heuristic used if None value given:
                max_faces_per_bin = int(max(10000, meshes._F / 5))
        :param perspective_correct: Bool, Whether to apply perspective correction when computing
            barycentric coordinates for pixels.
        :param cull_backfaces: Bool, Whether to only rasterize mesh faces which are
            visible to the camera.  This assumes that vertices of
            front-facing triangles are ordered in an anti-clockwise
            fashion, and triangles that face away from the camera are
            in a clockwise order relative to the current view
            direction. NOTE: This will only work if the mesh faces are
            consistently defined with counter-clockwise ordering when
            viewed from the outside.
        :param clip_barycentric_coords: By default, turn on clip_barycentric_coords
            if blur_radius > 0.
            When blur_radius > 0, a face can be matched to a pixel that is outside the face,
            resulting in negative barycentric coordinates.
        """
        super().__init__()
        self.img_shape = img_shape

        # UV pre-processing for textures
        verts_uv_offset, verts_iuv, verts_map, faces_densepose = preprocess_densepose_UV(
            uv_path="data/utils/UV_Processed.mat", batch_size=batch_size
        )
        self.verts_uv_offset = verts_uv_offset.to(device)
        self.verts_iuv = verts_iuv.to(device)
        self.verts_map = verts_map.to(device)
        self.faces_densepose = faces_densepose.to(device)

        # Cameras - pre-defined here but can be specified in forward pass
        # if cameras will vary (e.g. random cameras)
        assert projection_type in ["perspective", "orthographic"], print(
            "Invalid projection type:", projection_type
        )
        self.projection_type = projection_type
        if cam_R is None:
            # Rotating 180° about z-axis to make pytorch3d camera convention
            # same as what I've been using so far in my perspective_project_torch/NMR/pyrender
            # (Actually pyrender also has a rotation defined in
            # the renderer to make it same as NMR.)
            cam_R = torch.tensor(
                [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], device=device
            ).float()
            cam_R = cam_R[None, :, :].expand(batch_size, -1, -1)
            # cam_R = torch.eye(3, device=device)[None].expand(batch_size, -1, -1)
        if cam_t is None:
            cam_t = torch.tensor([0.0, 0.2, 2.5]).float().to(device)[None, :].expand(batch_size, -1)
        # Pytorch3D camera is rotated 180° about z-axis to match my
        # perspective_project_torch/NMR's projection convention.
        # So, need to also rotate the given camera translation
        # (implemented below as elementwise-mul).
        cam_t = cam_t * torch.tensor([-1.0, -1.0, 1.0], device=cam_t.device).float()
        if projection_type == "perspective":
            principal_point = torch.tensor(
                [self.img_shape[1] / 2.0, self.img_shape[0] / 2.0],
                device=device,
                dtype=torch.float32,
            )
            self.cameras = PerspectiveCameras(
                device=device,
                R=cam_R,
                T=cam_t,
                focal_length=perspective_focal_length[None],
                principal_point=principal_point[None],
                image_size=((int(self.img_shape[0]), int(self.img_shape[1])),),
                in_ndc=False,
            )
        elif projection_type == "orthographic":
            self.cameras = OrthographicCameras(
                device=device,
                R=cam_R,
                T=cam_t,
                focal_length=orthographic_scale * (img_wh / 2.0),
                principal_point=((img_wh / 2.0, img_wh / 2.0),),
                image_size=((img_wh, img_wh),),
                in_ndc=False,
            )

        # Lights for textured RGB render - pre-defined here but can be specified
        # in forward pass if lights will vary (e.g. random cameras)
        self.render_rgb = render_rgb
        if self.render_rgb:
            self.lights_rgb_render = PointLights(
                device=device,
                location=light_t,
                ambient_color=light_ambient_color,
                diffuse_color=light_diffuse_color,
                specular_color=light_specular_color,
            )
        # Lights for IUV render - don't want lighting to affect the rendered image.
        self.lights_iuv_render = PointLights(
            device=device,
            ambient_color=[[1, 1, 1]],
            diffuse_color=[[0, 0, 0]],
            specular_color=[[0, 0, 0]],
        )

        # Rasterizer
        raster_settings = RasterizationSettings(
            image_size=(int(self.img_shape[0]), int(self.img_shape[1])),
            blur_radius=blur_radius,
            faces_per_pixel=faces_per_pixel,
            bin_size=bin_size,
            max_faces_per_bin=max_faces_per_bin,
            perspective_correct=perspective_correct,
            cull_backfaces=cull_backfaces,
            clip_barycentric_coords=clip_barycentric_coords,
        )
        self.rasterizer = MeshRasterizer(
            cameras=self.cameras, raster_settings=raster_settings
        )  # Specify camera in forward pass

        # Shader for textured RGB output and IUV output
        blend_params = BlendParams(background_color=background_color)
        self.iuv_shader = HardPhongShader(
            device=device,
            cameras=self.cameras,
            lights=self.lights_iuv_render,
            blend_params=blend_params,
        )
        if self.render_rgb:
            self.rgb_shader = HardPhongShader(
                device=device,
                cameras=self.cameras,
                lights=self.lights_rgb_render,
                blend_params=blend_params,
            )

        self.to(device)

    def to(self, device):
        # Rasterizer and shader have submodules which are not of type nn.Module
        self.rasterizer.to(device)
        if self.render_rgb:
            self.rgb_shader.to(device)
        self.iuv_shader.to(device)

    def forward(
        self,
        vertices,
        textures=None,
        cam_t=None,
        orthographic_scale=None,
        perspective_focal_length=None,
        lights_rgb_settings=None,
        verts_features=None,
    ):
        """
        Render a batch of textured RGB images and IUV images from a batch of meshes.

        Fragments output from rasterizer:
        pix_to_face:
          LongTensor of shape (B, image_size, image_size, faces_per_pixel)
          specifying the indices of the faces (in the packed faces)
          which overlap each pixel in the image.
        zbuf:
          FloatTensor of shape (B, image_size, image_size, faces_per_pixel)
          giving the z-coordinates of the nearest faces at each pixel in world
          coordinates, sorted in ascending z-order.
        bary_coords:
          FloatTensor of shape (B, image_size, image_size, faces_per_pixel, 3)
          giving the barycentric coordinates in NDC units of the nearest
          faces at each pixel, sorted in ascending z-order.
        pix_dists:
          FloatTensor of shape (B, image_size, image_size, faces_per_pixel)
          giving the signed Euclidean distance (in NDC units) in the x/y plane
          of each point closest to the pixel.

        :param vertices: (B, N, 3)
        :param textures: (B, tex_H, tex_W, 3)
        :param cam_t: (B, 3)
        :param orthographic_scale: (B, 2)
        :param perspective_focal_length: (B, 2)
        :param lights_rgb_settings: dict of lighting settings with location,
            ambient_color, diffuse_color and specular_color.
        :returns rgb_images: (B, img_wh, img_wh, 3)
        :returns iuv_images: (B, img_wh, img_wh, 3) IUV images give bodypart
            (I) + UV coordinate information. Parts are DP convention, indexed 1-24.
        :returns depth_images: (B, img_wh, img_wh)
        """
        if cam_t is not None:
            # Pytorch3D camera is rotated 180° about z-axis to
            # match my perspective_project_torch/NMR's projection convention.
            # So, need to also rotate the given camera translation
            # (implemented below as elementwise-mul).
            self.cameras.T = cam_t * torch.tensor([-1.0, -1.0, 1.0], device=cam_t.device).float()
        if perspective_focal_length is not None and self.projection_type == "perspective":
            self.cameras.focal_length = perspective_focal_length
        if orthographic_scale is not None and self.projection_type == "orthographic":
            self.cameras.focal_length = orthographic_scale * (self.img_wh / 2.0)
        if lights_rgb_settings is not None and self.render_rgb:
            self.lights_rgb_render.location = lights_rgb_settings["location"]
            self.lights_rgb_render.ambient_color = lights_rgb_settings["ambient_color"]
            self.lights_rgb_render.diffuse_color = lights_rgb_settings["diffuse_color"]
            self.lights_rgb_render.specular_color = lights_rgb_settings["specular_color"]
        # From SMPL verts indexing (0 to 6889) to DP verts indexing(0 to 7828),
        vertices = vertices[:, self.verts_map, :]
        textures_iuv = TexturesVertex(verts_features=self.verts_iuv)
        meshes_iuv = Meshes(verts=vertices, faces=self.faces_densepose, textures=textures_iuv)
        if self.render_rgb:
            if verts_features is not None:
                # From SMPL verts indexing (0 to 6889) to DP
                verts_features = verts_features[:, self.verts_map, :]
                textures_rgb = TexturesVertex(verts_features=verts_features)
            else:
                textures_rgb = TexturesUV(
                    maps=textures, faces_uvs=self.faces_densepose, verts_uvs=self.verts_uv_offset
                )
            meshes_rgb = Meshes(verts=vertices, faces=self.faces_densepose, textures=textures_rgb)

        # Rasterize
        fragments = self.rasterizer(meshes_iuv, cameras=self.cameras)
        zbuffers = fragments.zbuf[:, :, :, 0]

        # Render RGB and IUV outputs
        output = {}
        output["iuv_images"] = self.iuv_shader(
            fragments, meshes_iuv, lights=self.lights_iuv_render
        )[:, :, :, :3]
        if self.render_rgb:
            rgb_images = self.rgb_shader(fragments, meshes_rgb, lights=self.lights_rgb_render)[
                :, :, :, :3
            ]
            output["rgb_images"] = torch.clamp(rgb_images, max=1.0)

        # Get depth image
        output["depth_images"] = zbuffers
        return output


def preprocess_densepose_UV(uv_path, batch_size):
    DP_UV = loadmat(uv_path)
    faces_bodyparts = torch.Tensor(
        DP_UV["All_FaceIndices"]
    ).squeeze()  # (13774,) face to DensePose body part mapping
    faces_densepose = torch.from_numpy(
        (DP_UV["All_Faces"] - 1).astype(np.int64)
    )  # (13774, 3) face to vertices indices mapping
    verts_map = (
        torch.from_numpy(DP_UV["All_vertices"][0].astype(np.int64)) - 1
    )  # (7829,) DensePose vertex to SMPL vertex mapping
    u_norm = torch.Tensor(
        DP_UV["All_U_norm"]
    )  # (7829, 1)  # Normalised U coordinates for each vertex
    v_norm = torch.Tensor(
        DP_UV["All_V_norm"]
    )  # (7829, 1)  # Normalised V coordinates for each vertex

    # RGB texture images/maps are processed into a 6 x 4 grid (atlas) of 24 textures.
    # Atlas is ordered by DensePose body parts (down rows then across columns).
    # UV coordinates for vertices need to be offset to match the texture image grid.
    offset_per_part = {}
    already_offset = set()
    cols, rows = 4, 6
    for i, u in enumerate(np.linspace(0, 1, cols, endpoint=False)):
        for j, v in enumerate(np.linspace(0, 1, rows, endpoint=False)):
            part = rows * i + j + 1  # parts are 1-indexed in face_indices
            offset_per_part[part] = (u, v)
    u_norm_offset = u_norm.clone()
    v_norm_offset = v_norm.clone()
    # Also want to get a mapping between vertices and their corresponding DP
    # body parts (technically one-to-many but ignoring that here).
    vertex_parts = torch.zeros(u_norm.shape[0])
    for i in range(len(faces_densepose)):
        face_vert_idxs = faces_densepose[i]
        part = faces_bodyparts[i]
        offset_u, offset_v = offset_per_part[int(part.item())]
        for vert_idx in face_vert_idxs:
            # vertices are reused (at DensePose part boundaries),
            # but we don't want to offset multiple times
            if vert_idx.item() not in already_offset:
                # offset u value
                u_norm_offset[vert_idx] = u_norm_offset[vert_idx] / cols + offset_u
                # offset v value
                # this also flips each part locally, as each part is upside down
                v_norm_offset[vert_idx] = (1 - v_norm_offset[vert_idx]) / rows + offset_v
                # add vertex to our set tracking offsetted vertices
                already_offset.add(vert_idx.item())
        vertex_parts[face_vert_idxs] = part

    # invert V values
    v_norm = 1 - v_norm
    v_norm_offset = 1 - v_norm_offset

    # Combine body part indices (I), and UV coordinates
    verts_uv_offset = torch.cat([u_norm_offset[None], v_norm_offset[None]], dim=2).expand(
        batch_size, -1, -1
    )  # (batch_size, 7829, 2)
    verts_iuv = torch.cat([vertex_parts[None, :, None], u_norm[None], v_norm[None]], dim=2).expand(
        batch_size, -1, -1
    )  # (batch_size, 7829, 3)

    # Add a batch dimension to faces
    faces_densepose = faces_densepose[None].expand(batch_size, -1, -1)

    return verts_uv_offset, verts_iuv, verts_map, faces_densepose
