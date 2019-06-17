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
import random

import src.data_loading.loading as loading
import src.utilities.pickling as pickling
import src.utilities.saving_images as saving_images

import src.heatmaps.run_producer as run_producer


def produce_heatmaps(parameters):
    """
    Generate heatmaps for single example
    """
    random.seed(parameters['seed'])
    image_path = parameters["cropped_mammogram_path"]
    model, device = run_producer.load_model(parameters)
    metadata = pickling.unpickle_from_file(parameters['metadata_path'])
    patches, case = run_producer.sample_patches_single(
        image_path=image_path,
        view=metadata["view"],
        horizontal_flip=metadata['horizontal_flip'],
        parameters=parameters,
    )
    all_prob = run_producer.get_all_prob(
        all_patches=patches,
        minibatch_size=parameters["minibatch_size"],
        model=model,
        device=device,
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

        cropped_mammogram_path=args.cropped_mammogram_path,
        metadata_path=args.metadata_path,
        heatmap_path_malignant=args.heatmap_path_malignant,
        heatmap_path_benign=args.heatmap_path_benign,

        heatmap_type=[0, 1],  # 0: malignant 1: benign 0: nothing

        use_hdf5=args.use_hdf5
    )
    produce_heatmaps(parameters)


if __name__ == "__main__":
    main()
