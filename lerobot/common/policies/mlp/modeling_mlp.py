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

from lerobot.common.constants import ACTION, OBS_IMAGES, STATE
from lerobot.common.policies.mlp.configuration_mlp import MlpConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy


class MlpPolicy(PreTrainedPolicy):
    """
    MLP policy with ResNet backbone.
    """

    config_class = MlpConfig
    name = "mlp"

    def __init__(
        self,
        config: MlpConfig,
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
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()

        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions = self.model(batch)
        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        batch = self.normalize_targets(batch)
        actions_hat = self.model(batch)

        l1_loss = F.l1_loss(batch[ACTION], actions_hat)
        loss_dict = {"l1_loss": l1_loss.item()}
        loss = l1_loss

        return loss, loss_dict


class MLP(nn.Module):
    def __init__(self, config: MlpConfig):
        super().__init__()
        self.config = config

        # Instantiate state feature extractor
        if config.state_feature:
            self.state_feature_extractor = nn.Sequential(
                nn.Linear(
                    config.state_feature.shape[0] * self.config.n_obs_steps,
                    config.state_feature_dim,
                ),
                nn.ReLU(),
            )

        # Instantiate image feature extractor
        if config.image_features:
            vision_backbone = getattr(torchvision.models, config.vision_backbone)
            backbone_model = vision_backbone(
                weights=config.pretrained_backbone_weights, norm_layer=FrozenBatchNorm2d
            )
            self.backbone = nn.Sequential(*list(backbone_model.children())[:-1])
            image_feature_dim = backbone_model.fc.in_features
        else:
            image_feature_dim = 0

        # Instantiate linear layers
        combined_feature_dim = (
            config.state_feature_dim if config.state_feature else 0
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
        if self.config.state_feature:
            state_seq = batch[STATE]
            # Assumes state_seq has shape (batch, n_obs_steps, state_dim)
            state_seq = state_seq.reshape(state_seq.shape[0], -1)
            state_feature = self.state_feature_extractor(state_seq)

        # Extract image features
        image_features = None
        if self.config.image_features:
            images_seq = batch[OBS_IMAGES]  # List of (B, n_obs, C, H, W)
            # Reshape to (B, num_images * n_obs, C, H, W)
            images_seq_cat = torch.cat(images_seq, dim=1)
            batch_size, num_obs_total, C, H, W = images_seq_cat.shape
            images_seq_flat = images_seq_cat.reshape(batch_size * num_obs_total, C, H, W)
            
            image_features_flat = self.backbone(images_seq_flat)
            image_features = image_features_flat.reshape(batch_size, num_obs_total, -1)
            image_features = image_features.reshape(batch_size, -1)


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
