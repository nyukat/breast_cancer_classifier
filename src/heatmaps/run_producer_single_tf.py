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
Generates benign and malignant heatmaps for cropped images using patch classifier.
"""
import argparse
import json
import numpy as np
import random

import tensorflow as tf
from tensorflow.contrib import slim
from nets.densenet import densenet121, densenet_arg_scope
import torch

import src.data_loading.loading as loading
import src.utilities.pickling as pickling
import src.utilities.saving_images as saving_images
import src.utilities.tf_utils as tf_utils
import src.utilities.tools as tools

import src.heatmaps.run_producer as run_producer


def construct_densenet_match_dict(tf_variables, torch_weights, tf_torch_weights_map):
    tf_var_dict = {var.name: var for var in tf_variables}
    match_dict = {}
    for tf_key, v in tf_var_dict.items():
        torch_w = torch_weights[tf_torch_weights_map[tf_key]].cpu().numpy()
        if tf_key == "densenet121/logits/weights:0":
            torch_w = tf_utils.convert_fc_weight_torch2tf(torch_w)[[np.newaxis, np.newaxis]]
        elif len(v.shape) == 4:
            torch_w = tf_utils.convert_conv_torch2tf(torch_w)
        elif len(v.shape) == 2:
            torch_w = tf_utils.convert_fc_weight_torch2tf(torch_w)
        match_dict[v] = torch_w
    return match_dict


def load_model_tf(parameters):
    # Setup model params
    if (parameters["device_type"] == "gpu") and tf.test.is_gpu_available():
        device_str = "/device:GPU:{}".format(parameters["gpu_number"])
    else:
        device_str = "/cpu:0"

    # Setup Graph
    graph = tf.Graph()
    with graph.as_default():
        with tf.device(device_str):
            x = tf.placeholder(tf.float32, [None, 256, 256, 3])
            with slim.arg_scope(densenet_arg_scope(weight_decay=0.0, data_format='NHWC')):
                densenet121_net, end_points = densenet121(
                    x,
                    num_classes=parameters["number_of_classes"],
                    data_format='NHWC',
                    is_training=False,
                )
            y_logits = densenet121_net[:, 0, 0, :]
            y = tf.nn.softmax(y_logits)

    # Load weights
    sess = tf.Session(graph=graph, config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    ))
    with open(parameters["tf_torch_weights_map_path"]) as f:
        tf_torch_weights_map = json.loads(f.read())

    with sess.as_default():
        torch_weights = torch.load(parameters["initial_parameters"])
        match_dict = construct_densenet_match_dict(
            tf_variables=tf_utils.get_tf_variables(graph, batch_norm_key="BatchNorm"),
            torch_weights=torch_weights,
            tf_torch_weights_map=tf_torch_weights_map
        )
        sess.run(tf_utils.construct_weight_assign_ops(match_dict))

    return sess, x, y


def prediction_by_batch_tf(minibatch_patches, sess, x, y, parameters):
    """
    Puts patches into a batch and gets predictions of patch classifier.
    """
    minibatch_x = np.stack((minibatch_patches,) * parameters['input_channels'], axis=-1).reshape(
        -1, parameters['patch_size'], parameters['patch_size'], parameters['input_channels']
    )

    output = sess.run(y, feed_dict={x: minibatch_x})
    return output


def get_all_prob_tf(all_patches, minibatch_size, sess, x, y, parameters):
    """
    Gets predictions for all sampled patches
    """
    all_prob = np.zeros((len(all_patches), parameters['number_of_classes']))

    for i, minibatch in enumerate(tools.partition_batch(all_patches, minibatch_size)):
        minibatch_prob = prediction_by_batch_tf(minibatch, sess, x, y, parameters)
        all_prob[i * minibatch_size: i * minibatch_size + minibatch_prob.shape[0]] = minibatch_prob

    return all_prob.astype(np.float32)


def produce_heatmaps(parameters):
    """
    Generate heatmaps for single example
    """
    random.seed(parameters['seed'])
    image_path = parameters["cropped_mammogram_path"]
    sess, x, y = load_model_tf(parameters)
    metadata = pickling.unpickle_from_file(parameters['metadata_path'])
    patches, case = run_producer.sample_patches_single(
        image_path=image_path,
        view=metadata["view"],
        horizontal_flip=metadata['horizontal_flip'],
        parameters=parameters,
    )
    all_prob = get_all_prob_tf(
        all_patches=patches,
        minibatch_size=parameters["minibatch_size"],
        sess=sess,
        x=x,
        y=y,
        parameters=parameters
    )
    heatmap_malignant, _ = run_producer.probabilities_to_heatmap(
        patch_counter=0,
        all_prob=all_prob,
        image_shape=case[0],
        length_stride_list=case[4],
        width_stride_list=case[3],
        patch_size=parameters['patch_size'],
        heatmap_type=parameters['heatmap_type'][0],
    )
    heatmap_benign, patch_counter = run_producer.probabilities_to_heatmap(
        patch_counter=0,
        all_prob=all_prob,
        image_shape=case[0],
        length_stride_list=case[4],
        width_stride_list=case[3],
        patch_size=parameters['patch_size'],
        heatmap_type=parameters['heatmap_type'][1],
    )
    heatmap_malignant = loading.flip_image(
        image=heatmap_malignant,
        view=metadata["view"],
        horizontal_flip=metadata["horizontal_flip"],
    )
    heatmap_benign = loading.flip_image(
        image=heatmap_benign,
        view=metadata["view"],
        horizontal_flip=metadata["horizontal_flip"],
    )
    saving_images.save_image_as_hdf5(
        image=heatmap_malignant,
        filename=parameters["heatmap_path_malignant"],
    )
    saving_images.save_image_as_hdf5(
        image=heatmap_benign,
        filename=parameters["heatmap_path_benign"],
    )


def main():
    parser = argparse.ArgumentParser(description='Produce Heatmaps')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--tf-torch-weights-map-path', required=True)
    parser.add_argument('--cropped-mammogram-path', required=True)
    parser.add_argument('--metadata-path', required=True)
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--heatmap-path-malignant', required=True)
    parser.add_argument('--heatmap-path-benign', required=True)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device-type', default="cpu", choices=['gpu', 'cpu'])
    parser.add_argument("--gpu-number", type=int, default=0)
    parser.add_argument("--use-hdf5", action="store_true")
    args = parser.parse_args()

    parameters = dict(
        device_type=args.device_type,
        gpu_number=args.gpu_number,

        patch_size=256,

        stride_fixed=70,
        more_patches=5,
        minibatch_size=args.batch_size,
        seed=args.seed,

        initial_parameters=args.model_path,
        input_channels=3,
        number_of_classes=4,
        tf_torch_weights_map_path=args.tf_torch_weights_map_path,

        cropped_mammogram_path=args.cropped_mammogram_path,
        metadata_path=args.metadata_path,
        heatmap_path_malignant=args.heatmap_path_malignant,
        heatmap_path_benign=args.heatmap_path_benign,

        heatmap_type=[0, 1],  # 0: malignant 1: benign

        use_hdf5=args.use_hdf5
    )
    produce_heatmaps(parameters)


if __name__ == "__main__":
    main()
