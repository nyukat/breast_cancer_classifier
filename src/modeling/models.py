# Copyright (C) 2019 Nan Wu, Jason Phang, Jungkyu Park, Yiqiu Shen, Zhe Huang, Masha Zorin, 
#   Stanisław Jastrzębski, Thibault Févry, Joe Katsnelson, Eric Kim, Stacey Wolfson, Ujas Parikh, 
#   Sushma Gaddam, Leng Leng Young Lin, Kara Ho, Joshua D. Weinstein, Beatriu Reig, Yiming Gao, 
#   Hildegard Toth, Kristine Pysarenko, Alana Lewin, Jiyon Lee, Krystal Airola, Eralda Mema, 
#   Stephanie Chung, Esther Hwang, Naziya Samreen, S. Gene Kim, Laura Heacock, Linda Moy, 
#   Kyunghyun Cho, Krzysztof J. Geras
#
# This file is part of breast_cancer_classifier.
#
# breast_cancer_classifier is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# breast_cancer_classifier is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with breast_cancer_classifier.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================
"""
Defines architectures for breast cancer classification models. 
"""
import collections as col

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.modeling.layers as layers
from src.constants import VIEWS, VIEWANGLES


class SplitBreastModel(nn.Module):
    def __init__(self, input_channels):
        super(SplitBreastModel, self).__init__()

        self.four_view_resnet = FourViewResNet(input_channels)

        self.fc1_cc = nn.Linear(256 * 2, 256 * 2)
        self.fc1_mlo = nn.Linear(256 * 2, 256 * 2)
        self.output_layer_cc = layers.OutputLayer(256 * 2, (4, 2))
        self.output_layer_mlo = layers.OutputLayer(256 * 2, (4, 2))

        self.all_views_avg_pool = layers.AllViewsAvgPool()
        self.all_views_gaussian_noise_layer = layers.AllViewsGaussianNoise(0.01)

    def forward(self, x):
        h = self.all_views_gaussian_noise_layer(x)
        result = self.four_view_resnet(h)
        h = self.all_views_avg_pool(result)

        # Pool, flatten, and fully connected layers
        h_cc = torch.cat([h[VIEWS.L_CC], h[VIEWS.R_CC]], dim=1)
        h_mlo = torch.cat([h[VIEWS.L_MLO], h[VIEWS.R_MLO]], dim=1)

        h_cc = F.relu(self.fc1_cc(h_cc))
        h_mlo = F.relu(self.fc1_mlo(h_mlo))

        h_cc = self.output_layer_cc(h_cc)
        h_mlo = self.output_layer_mlo(h_mlo)

        h = {
            VIEWANGLES.CC: h_cc,
            VIEWANGLES.MLO: h_mlo,
        }

        return h


class ImageBreastModel(nn.Module):
    def __init__(self, input_channels):
        super(ImageBreastModel, self).__init__()

        self.four_view_resnet = FourViewResNet(input_channels)

        self.fc1_lcc = nn.Linear(256, 256)
        self.fc1_rcc = nn.Linear(256, 256)
        self.fc1_lmlo = nn.Linear(256, 256)
        self.fc1_rmlo = nn.Linear(256, 256)
        self.output_layer_lcc = layers.OutputLayer(256, (4, 2))
        self.output_layer_rcc = layers.OutputLayer(256, (4, 2))
        self.output_layer_lmlo = layers.OutputLayer(256, (4, 2))
        self.output_layer_rmlo = layers.OutputLayer(256, (4, 2))

        self.all_views_avg_pool = layers.AllViewsAvgPool()
        self.all_views_gaussian_noise_layer = layers.AllViewsGaussianNoise(0.01)

    def forward(self, x):
        h = self.all_views_gaussian_noise_layer(x)
        result = self.four_view_resnet(h)
        h = self.all_views_avg_pool(result)

        h_lcc = F.relu(self.fc1_lcc(h[VIEWS.L_CC]))
        h_rcc = F.relu(self.fc1_rcc(h[VIEWS.R_CC]))
        h_lmlo = F.relu(self.fc1_lmlo(h[VIEWS.L_MLO]))
        h_rmlo = F.relu(self.fc1_rmlo(h[VIEWS.R_MLO]))

        h_lcc = self.output_layer_lcc(h_lcc)
        h_rcc = self.output_layer_rcc(h_rcc)
        h_lmlo = self.output_layer_lmlo(h_lmlo)
        h_rmlo = self.output_layer_rmlo(h_rmlo)

        h = {
            VIEWS.L_CC: h_lcc,
            VIEWS.R_CC: h_rcc,
            VIEWS.L_MLO: h_lmlo,
            VIEWS.R_MLO: h_rmlo,
        }

        return h


class SingleImageBreastModel(nn.Module):
    def __init__(self, input_channels):
        super(SingleImageBreastModel, self).__init__()

        self.view_resnet = resnet22(input_channels)

        self.fc1 = nn.Linear(256, 256)
        self.output_layer = layers.OutputLayer(256, (2, 2))

        self.all_views_avg_pool = layers.AllViewsAvgPool()
        self.all_views_gaussian_noise_layer = layers.AllViewsGaussianNoise(0.01)

    def forward(self, x):
        h = self.all_views_gaussian_noise_layer.single_add_gaussian_noise(x)
        result = self.view_resnet(h)
        h = self.all_views_avg_pool.single_avg_pool(result)
        h = F.relu(self.fc1(h))
        h = self.output_layer(h)[:2]
        return h

    def load_state_from_shared_weights(self, state_dict, view):
        view_angle = view.lower().split("-")[-1]
        view_key = view.lower().replace("-", "")
        self.view_resnet.load_state_dict(
            filter_strip_prefix(state_dict, "four_view_resnet.{}.".format(view_angle))
        )
        self.fc1.load_state_dict(
            filter_strip_prefix(state_dict, "fc1_{}.".format(view_key))
        )
        self.output_layer.load_state_dict({
            "fc_layer.weight": state_dict["output_layer_{}.fc_layer.weight".format(view_key)][:4],
            "fc_layer.bias": state_dict["output_layer_{}.fc_layer.bias".format(view_key)][:4],
        })


class FourViewResNet(nn.Module):
    def __init__(self, input_channels):
        super(FourViewResNet, self).__init__()

        self.cc = resnet22(input_channels)
        self.mlo = resnet22(input_channels)
        self.model_dict = {}
        self.model_dict[VIEWS.L_CC] = self.l_cc = self.cc
        self.model_dict[VIEWS.L_MLO] = self.l_mlo = self.mlo
        self.model_dict[VIEWS.R_CC] = self.r_cc = self.cc
        self.model_dict[VIEWS.R_MLO] = self.r_mlo = self.mlo

    def forward(self, x):
        h_dict = {
            view: self.single_forward(x[view], view)
            for view in VIEWS.LIST
        }
        return h_dict

    def single_forward(self, single_x, view):
        return self.model_dict[view](single_x)


class ViewResNetV2(nn.Module):
    """
    Adapted fom torchvision ResNet, converted to v2
    """
    def __init__(self,
                 input_channels, num_filters,
                 first_layer_kernel_size, first_layer_conv_stride,
                 blocks_per_layer_list, block_strides_list, block_fn,
                 first_layer_padding=0,
                 first_pool_size=None, first_pool_stride=None, first_pool_padding=0,
                 growth_factor=2):
        super(ViewResNetV2, self).__init__()
        self.first_conv = nn.Conv2d(
            in_channels=input_channels, out_channels=num_filters,
            kernel_size=first_layer_kernel_size,
            stride=first_layer_conv_stride,
            padding=first_layer_padding,
            bias=False,
        )
        self.first_pool = nn.MaxPool2d(
            kernel_size=first_pool_size,
            stride=first_pool_stride,
            padding=first_pool_padding,
        )

        self.layer_list = nn.ModuleList()
        current_num_filters = num_filters
        self.inplanes = num_filters
        for i, (num_blocks, stride) in enumerate(zip(
                blocks_per_layer_list, block_strides_list)):
            self.layer_list.append(self._make_layer(
                block=block_fn,
                planes=current_num_filters,
                blocks=num_blocks,
                stride=stride,
            ))
            current_num_filters *= growth_factor
        self.final_bn = nn.BatchNorm2d(
            current_num_filters // growth_factor * block_fn.expansion
        )
        self.relu = nn.ReLU()

        # Expose attributes for downstream dimension computation
        self.num_filters = num_filters
        self.growth_factor = growth_factor

    def forward(self, x):
        h = self.first_conv(x)
        h = self.first_pool(h)
        for i, layer in enumerate(self.layer_list):
            h = layer(h)
        h = self.final_bn(h)
        h = self.relu(h)
        return h

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
        )

        layers_ = [
            block(self.inplanes, planes, stride, downsample)
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers_.append(block(self.inplanes, planes))

        return nn.Sequential(*layers_)


def resnet22(input_channels):
    return ViewResNetV2(
        input_channels=input_channels,
        num_filters=16,
        first_layer_kernel_size=7,
        first_layer_conv_stride=2,
        blocks_per_layer_list=[2, 2, 2, 2, 2],
        block_strides_list=[1, 2, 2, 2, 2],
        block_fn=layers.BasicBlockV2,
        first_layer_padding=0,
        first_pool_size=3,
        first_pool_stride=2,
        first_pool_padding=0,
        growth_factor=2
    )


def filter_strip_prefix(weights_dict, prefix):
    return {
        k.replace(prefix, ""): v
        for k, v in weights_dict.items()
        if k.startswith(prefix)
    }
