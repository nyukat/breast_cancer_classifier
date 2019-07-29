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
import numpy as np

import tensorflow as tf


def get_tf_variables(graph, batch_norm_key="bn"):
    param_variables = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    bn_running_variables = []
    for variable in graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        if batch_norm_key in variable.name \
                and "moving" in variable.name:
            bn_running_variables.append(variable)
    return param_variables + bn_running_variables


def construct_weight_assign_ops(match_dict):
    assign_list = []
    for tf_var, np_weights in match_dict.items():
        assign_list.append(tf_var.assign(np_weights))
    return assign_list


def convert_conv_torch2tf(w):
    # [C_out, C_in, H, W] => [H, W, C_in, C_out]
    return np.transpose(w, [2, 3, 1, 0])


def convert_fc_weight_torch2tf(w):
    return w.swapaxes(0, 1)
