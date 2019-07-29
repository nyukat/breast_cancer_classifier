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
TensorFlow model definition and utils
"""

import tensorflow as tf

import src.modeling.layers_tf as layers
import src.utilities.tf_utils as tf_utils
from src.constants import VIEWS

DATA_FORMAT = "channels_first"


def single_image_breast_model(inputs, training):
    with tf.variable_scope("model"):
        h = layers.gaussian_noise_layer(
            inputs=inputs,
            std=0.01,
            training=training,
        )
        h = resnet22(
            inputs=h,
            training=training,
        )
        h = layers.avg_pool_layer(h)
        with tf.variable_scope("fc1"):
            h = tf.layers.dense(h, activation="relu", units=256)
        h = layers.output_layer(h, output_shape=(2, 2))
        return h


def construct_single_image_breast_model_match_dict(
        view_str, tf_variables, torch_weights, tf_torch_weights_map):
    """
    view_str: e.g. "r_mlo"
    """
    if isinstance(torch_weights, str):
        import torch
        torch_weights = torch.load(torch_weights)["model"]
    torch_weights = {k: w.numpy() for k, w in torch_weights.items()}
    match_dict = {}
    torch_resnet_prefix = "four_view_resnet.{}.".format(view_str)
    tf_var_dict = {var.name: var for var in tf_variables}
    for tf_var_name, tf_var in tf_var_dict.items():
        if "resnet" not in tf_var.name:
            continue
        lookup_key = tf_var_name.replace("model/", "")
        weight = torch_weights[torch_resnet_prefix + tf_torch_weights_map[lookup_key]]
        if len(weight.shape) == 4:
            weight = tf_utils.convert_conv_torch2tf(weight)
        assert tf_var.shape == weight.shape
        match_dict[tf_var] = weight
    short_view_str = view_str.replace("_", "")
    match_dict[tf_var_dict["model/fc1/dense/kernel:0"]] = \
        tf_utils.convert_fc_weight_torch2tf(torch_weights["fc1_{}.weight".format(short_view_str)])
    match_dict[tf_var_dict["model/fc1/dense/bias:0"]] = \
        torch_weights["fc1_{}.bias".format(short_view_str)]
    match_dict[tf_var_dict["model/output_layer/dense/kernel:0"]] = \
        tf_utils.convert_fc_weight_torch2tf(
            torch_weights["output_layer_{}.fc_layer.weight".format(short_view_str)])[:, :4]
    match_dict[tf_var_dict["model/output_layer/dense/bias:0"]] = \
        torch_weights["output_layer_{}.fc_layer.bias".format(short_view_str)][:4]
    assert len(match_dict) == len(tf_variables)
    return match_dict


def four_view_resnet(inputs, training):
    result_dict = {}
    for view in VIEWS.LIST:
        with tf.variable_scope("view_{}".format(view)):
            result_dict[view] = resnet22(inputs, training)
    return result_dict


def resnet22(inputs, training):
    return view_resnet_v2(
        inputs=inputs,
        training=training,
        num_filters=16,
        first_layer_kernel_size=7,
        first_layer_conv_stride=2,
        blocks_per_layer_list=[2, 2, 2, 2, 2],
        block_strides_list=[1, 2, 2, 2, 2],
        first_pool_size=3,
        first_pool_stride=2,
        growth_factor=2
    )


def view_resnet_v2(inputs, training, num_filters,
                   first_layer_kernel_size, first_layer_conv_stride,
                   blocks_per_layer_list, block_strides_list,
                   first_pool_size=None, first_pool_stride=None,
                   growth_factor=2):
    with tf.variable_scope("resnet"):
        with tf.variable_scope("first"):
            h = layers.conv2d_fixed_padding(
                inputs=inputs,
                filters=num_filters,
                kernel_size=first_layer_kernel_size,
                strides=first_layer_conv_stride,
                data_format=DATA_FORMAT,
                padding="valid",
                name="first_conv",
            )
            h = tf.layers.max_pooling2d(
                inputs=h,
                pool_size=first_pool_size,
                strides=first_pool_stride,
                padding='valid',
                data_format=DATA_FORMAT,
            )
        current_num_filters = num_filters
        for i, (num_blocks, stride) in enumerate(zip(
                blocks_per_layer_list, block_strides_list)):
            with tf.variable_scope("major_{}".format(i + 1)):
                h = make_layer(
                    inputs=h,
                    planes=current_num_filters,
                    training=training,
                    blocks=num_blocks,
                    stride=stride,
                )
                current_num_filters *= growth_factor
        with tf.variable_scope("final"):
            h = layers.batch_norm(
                inputs=h,
                training=training,
                data_format=DATA_FORMAT,
                name="bn",
            )
            h = tf.nn.relu(h)
    return h


def make_layer(inputs, planes, training, blocks, stride=1):
    with tf.variable_scope("block_0"):
        h = layers.basic_block_v2(
            inputs=inputs,
            planes=planes,
            training=training,
            data_format=DATA_FORMAT,
            strides=stride,
            downsample=True,
        )
    for i in range(1, blocks):
        with tf.variable_scope("block_{}".format(i)):
            h = layers.basic_block_v2(
                inputs=h,
                planes=planes,
                training=training,
                data_format=DATA_FORMAT,
                strides=1,
                downsample=False,
            )
    return h
