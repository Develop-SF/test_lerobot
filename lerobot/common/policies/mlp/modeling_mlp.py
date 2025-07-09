#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
import einops
import numpy as np

from lerobot.common.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.common.policies.mlp.configuration_mlp import MLPConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import populate_queues, get_output_shape


class MLPPolicy(PreTrainedPolicy):
    """
    MLP policy with ResNet backbone.
    """

    config_class = MLPConfig
    name = "mlp"

    def __init__(
        self,
        config: MLPConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.model = MLP(config)
        self.reset()

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    def reset(self):
        """This should be called whenever the environment is reset."""
        queues = {
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.robot_state_feature:
            queues[OBS_STATE] = deque(maxlen=self.config.n_obs_steps)
        if self.config.image_features:
            # We use a single queue for all images for simplicity.
            # During processing, this will be expanded into a list of tensors, one for each camera.
            queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        self._queues = queues

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()

        # Add image features to a single key for queueing.
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    @torch.no_grad
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()

        # during queues init, whether a key is used is already checked (using state or image)
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        actions = self.model(batch)
        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        batch = self.normalize_inputs(batch) 
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            images = []
            for key in self.config.image_features:
                img = batch[key]
                if img.ndim == 4:  # (B, C, H, W) for n_obs_steps=1
                    img = img.unsqueeze(1)  # (B, 1, C, H, W)
                images.append(img)
            batch[OBS_IMAGES] = torch.stack(images, dim=-4)  # (B, n_obs, num_cameras, C, H, W)

        batch = self.normalize_targets(batch)
        actions_hat = self.model(batch)

        l1_loss = F.l1_loss(batch[ACTION], actions_hat)
        loss_dict = {"l1_loss": l1_loss.item()}
        loss = l1_loss

        return loss, loss_dict


class SpatialSoftmax(nn.Module):
    """
    Spatial Soft Argmax operation described in "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
    """
    def __init__(self, input_shape, num_kp=None):
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape
        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))
    def forward(self, features: Tensor) -> Tensor:
        if self.nets is not None:
            features = self.nets(features)
        features = features.reshape(-1, self._in_h * self._in_w)
        attention = F.softmax(features, dim=-1)
        expected_xy = attention @ self.pos_grid
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)
        return feature_keypoints

class MLPRgbEncoder(nn.Module):
    """Encodes an RGB image into a 1D feature vector. Logic copied from DiffusionRgbEncoder."""
    def __init__(self, config):
        super().__init__()
        if config.crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if getattr(config, 'crop_is_random', False):
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        # No group norm logic for now (add if needed)
        images_shape = next(iter(config.image_features.values())).shape
        dummy_shape_h_w = config.crop_shape if config.crop_shape is not None else images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]
        self.pool = SpatialSoftmax(feature_map_shape, num_kp=getattr(config, 'spatial_softmax_num_keypoints', 32))
        self.feature_dim = getattr(config, 'spatial_softmax_num_keypoints', 32) * 2
        self.out = nn.Linear(self.feature_dim, self.feature_dim)
        self.relu = nn.ReLU()
    def forward(self, x: Tensor) -> Tensor:
        if self.do_crop:
            if self.training:
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        x = self.relu(self.out(x))
        return x

class MLP(nn.Module):
    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config

        # Instantiate state feature extractor
        if config.robot_state_feature:
            self.state_feature_extractor = nn.Sequential(
                nn.Linear(
                    config.robot_state_feature.shape[0] * self.config.n_obs_steps,
                    config.state_feature_dim,
                ),
                nn.ReLU(),
            )

        # Instantiate image feature extractor
        if config.image_features:
            num_images = len(config.image_features)
            if getattr(config, 'use_separate_rgb_encoder_per_camera', False):
                encoders = [MLPRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                image_feature_dim = encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = MLPRgbEncoder(config)
                image_feature_dim = self.rgb_encoder.feature_dim * num_images
        else:
            image_feature_dim = 0

        # Instantiate linear layers
        combined_feature_dim = (
            config.state_feature_dim if config.robot_state_feature else 0
        ) + len(config.image_features) * config.n_obs_steps * image_feature_dim

        linear_dim_list = (
            [combined_feature_dim]
            + config.hidden_dim_list
            + [config.action_feature.shape[0] * self.config.chunk_size]
        )
        linear_layers = []
        for linear_idx in range(len(linear_dim_list) - 1):
            input_dim = linear_dim_list[linear_idx]
            output_dim = linear_dim_list[linear_idx + 1]
            linear_layers.append(nn.Linear(input_dim, output_dim))
            if linear_idx < len(linear_dim_list) - 2:
                linear_layers.append(nn.ReLU())
        self.linear_layer_seq = nn.Sequential(*linear_layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch: dict[str, Tensor]):
        # Extract state features
        state_feature = None
        if self.config.robot_state_feature:
            state_seq = batch[OBS_STATE]
            # Assumes state_seq has shape (batch, n_obs_steps, state_dim)
            state_seq = state_seq.reshape(state_seq.shape[0], -1)
            state_feature = self.state_feature_extractor(state_seq)

        # Extract image features
        image_features = None
        if self.config.image_features:
            images_seq = batch[OBS_IMAGES]  # (B, n_obs, num_cameras, C, H, W)
            batch_size, n_obs, num_cameras, C, H, W = images_seq.shape
            if getattr(self.config, 'use_separate_rgb_encoder_per_camera', False):
                # For each camera, process all (B, n_obs, C, H, W) with its encoder
                images_per_camera = einops.rearrange(images_seq, 'b s n c h w -> n (b s) c h w')
                img_features_list = [
                    encoder(images)
                    for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                ]
                # (num_cameras, B*n_obs, D) -> (B, n_obs, num_cameras*D)
                img_features = torch.cat(img_features_list)
                img_features = einops.rearrange(
                    img_features, '(n b s) d -> b s (n d)', n=num_cameras, b=batch_size, s=n_obs
                )
            else:
                # Shared encoder: flatten batch, obs, camera
                images_flat = einops.rearrange(images_seq, 'b s n c h w -> (b s n) c h w')
                img_features = self.rgb_encoder(images_flat)
                # (B*n_obs*num_cameras, D) -> (B, n_obs, num_cameras*D)
                img_features = einops.rearrange(
                    img_features, '(b s n) d -> b s (n d)', b=batch_size, s=n_obs, n=num_cameras
                )
            image_features = img_features.reshape(batch_size, -1)

        # Concatenate features
        if state_feature is not None and image_features is not None:
            combined_feature = torch.cat([state_feature, image_features], dim=1)
        elif state_feature is not None:
            combined_feature = state_feature
        elif image_features is not None:
            combined_feature = image_features
        else:
            raise ValueError("At least one of state or image features must be provided.")

        # Apply linear layers
        action_seq = self.linear_layer_seq(combined_feature)

        # Reshape action_seq
        action_seq = action_seq.reshape(
            action_seq.shape[0], self.config.chunk_size, -1
        )
        return action_seq
