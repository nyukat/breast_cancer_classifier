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
Runs the image only model and image+heatmaps model for breast cancer prediction.
"""
import argparse
import numpy as np
import json
import tensorflow as tf

import src.utilities.pickling as pickling
import src.utilities.tf_utils as tf_utils
import src.utilities.tools as tools
import src.modeling.models_tf as models
import src.data_loading.loading as loading
from src.constants import INPUT_SIZE_DICT


class ModelInput:
    def __init__(self, image, heatmaps, metadata):
        self.image = image
        self.heatmaps = heatmaps
        self.metadata = metadata


def load_model(parameters):
    """
    Loads trained cancer classifier
    """
    # Setup model params
    input_channels = 3 if parameters["use_heatmaps"] else 1
    if (parameters["device_type"] == "gpu") and tf.test.is_gpu_available():
        device_str = "/device:GPU:{}".format(parameters["gpu_number"])
    else:
        device_str = "/cpu:0"
    view_str = parameters["view"].replace("-", "_").lower()
    h, w = INPUT_SIZE_DICT[parameters["view"]]

    # Setup Graph
    graph = tf.Graph()
    with graph.as_default():
        with tf.device(device_str):
            x = tf.placeholder(tf.float32, shape=[None, input_channels, h, w], name="inputs")
            y = models.single_image_breast_model(x, training=False)

    # Load weights
    sess = tf.Session(graph=graph, config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    ))
    with open(parameters["tf_torch_weights_map_path"]) as f:
        tf_torch_weights_map = json.loads(f.read())

    with sess.as_default():
        match_dict = models.construct_single_image_breast_model_match_dict(
            view_str=view_str,
            tf_variables=tf_utils.get_tf_variables(graph, batch_norm_key="bn"),
            torch_weights=parameters["model_path"],
            tf_torch_weights_map=tf_torch_weights_map,
        )
        sess.run(tf_utils.construct_weight_assign_ops(match_dict))

    return sess, x, y


def load_inputs(image_path, metadata_path,
                use_heatmaps, benign_heatmap_path=None, malignant_heatmap_path=None):
    """
    Load a single input example, optionally with heatmaps
    """
    if use_heatmaps:
        assert benign_heatmap_path is not None
        assert malignant_heatmap_path is not None
    else:
        assert benign_heatmap_path is None
        assert malignant_heatmap_path is None
    metadata = pickling.unpickle_from_file(metadata_path)
    image = loading.load_image(
        image_path=image_path,
        view=metadata["full_view"],
        horizontal_flip=metadata["horizontal_flip"],
    )
    if use_heatmaps:
        heatmaps = loading.load_heatmaps(
            benign_heatmap_path=benign_heatmap_path,
            malignant_heatmap_path=malignant_heatmap_path,
            view=metadata["full_view"],
            horizontal_flip=metadata["horizontal_flip"],
        )
    else:
        heatmaps = None
    return ModelInput(image=image, heatmaps=heatmaps, metadata=metadata)


def process_augment_inputs(model_input, random_number_generator, parameters):
    """
    Augment, normalize and convert inputs to np.ndarray
    """
    cropped_image, cropped_heatmaps = loading.augment_and_normalize_image(
        image=model_input.image,
        auxiliary_image=model_input.heatmaps,
        view=model_input.metadata["full_view"],
        best_center=model_input.metadata["best_center"],
        random_number_generator=random_number_generator,
        augmentation=parameters["augmentation"],
        max_crop_noise=parameters["max_crop_noise"],
        max_crop_size_noise=parameters["max_crop_size_noise"],
    )
    if parameters["use_heatmaps"]:
        return np.concatenate([
            cropped_image[:, :, np.newaxis],
            cropped_heatmaps,
        ], axis=2)
    else:
        return cropped_image[:, :, np.newaxis]


def batch_to_inputs(batch):
    """
    Convert list of input ndarrays to prepped inputs
    """
    return np.transpose(np.stack(batch), [0, 3, 1, 2])


def run(parameters):
    """
    Outputs the predictions as csv file
    """
    random_number_generator = np.random.RandomState(parameters["seed"])
    sess, x, y = load_model(parameters)

    model_input = load_inputs(
        image_path=parameters["cropped_mammogram_path"],
        metadata_path=parameters["metadata_path"],
        use_heatmaps=parameters["use_heatmaps"],
        benign_heatmap_path=parameters["heatmap_path_benign"],
        malignant_heatmap_path=parameters["heatmap_path_malignant"],
    )
    assert model_input.metadata["full_view"] == parameters["view"]

    all_predictions = []
    for data_batch in tools.partition_batch(range(parameters["num_epochs"]), parameters["batch_size"]):
        batch = []
        for _ in data_batch:
            batch.append(process_augment_inputs(
                model_input=model_input,
                random_number_generator=random_number_generator,
                parameters=parameters,
            ))
        x_data = batch_to_inputs(batch)
        with sess.as_default():
            y_hat = sess.run(y, feed_dict={x: x_data})
        predictions = np.exp(y_hat)[:, :, 1]
        all_predictions.append(predictions)
    agg_predictions = np.concatenate(all_predictions, axis=0).mean(0)
    predictions_dict = {
        "benign": float(agg_predictions[0]),
        "malignant": float(agg_predictions[1]),
    }
    print(json.dumps(predictions_dict))


def main():
    parser = argparse.ArgumentParser(description='Run image-only model or image+heatmap model')
    parser.add_argument('--view', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--tf-torch-weights-map-path', required=True)
    parser.add_argument('--cropped-mammogram-path', required=True)
    parser.add_argument('--metadata-path', required=True)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--use-heatmaps', action="store_true")
    parser.add_argument('--heatmap-path-malignant')
    parser.add_argument('--heatmap-path-benign')
    parser.add_argument('--use-augmentation', action="store_true")
    parser.add_argument('--use-hdf5', action="store_true")
    parser.add_argument('--num-epochs', default=1, type=int)
    parser.add_argument('--device-type', default="cpu", choices=['gpu', 'cpu'])
    parser.add_argument("--gpu-number", type=int, default=0)
    args = parser.parse_args()

    parameters = {
        "view": args.view,
        "model_path": args.model_path,
        "tf_torch_weights_map_path": args.tf_torch_weights_map_path,
        "cropped_mammogram_path": args.cropped_mammogram_path,
        "metadata_path": args.metadata_path,
        "device_type": args.device_type,
        "gpu_number": args.gpu_number,
        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "augmentation": args.use_augmentation,
        "num_epochs": args.num_epochs,
        "use_heatmaps": args.use_heatmaps,
        "heatmap_path_benign": args.heatmap_path_benign,
        "heatmap_path_malignant": args.heatmap_path_malignant,
        "use_hdf5": args.use_hdf5,
    }
    run(parameters)


if __name__ == "__main__":
    main()
