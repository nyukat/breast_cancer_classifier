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
Runs search_windows_and_centers.py and extract_centers.py in the same directory
"""
import argparse

import src.utilities.pickling as pickling
import src.utilities.reading_images as reading_images
import src.optimal_centers.get_optimal_centers as get_optimal_centers


def get_optimal_center_single(cropped_mammogram_path, metadata_path):
    """
    Get optimal center for single example
    """
    metadata = pickling.unpickle_from_file(metadata_path)
    image = reading_images.read_image_png(cropped_mammogram_path)
    optimal_center = get_optimal_centers.extract_center(metadata, image)
    metadata["best_center"] = optimal_center
    pickling.pickle_to_file(metadata_path, metadata)


def main():
    parser = argparse.ArgumentParser(description='Compute and Extract Optimal Centers')
    parser.add_argument('--cropped-mammogram-path', required=True)
    parser.add_argument('--metadata-path', required=True)
    parser.add_argument('--num-processes', default=20)
    args = parser.parse_args()
    get_optimal_center_single(
        cropped_mammogram_path=args.cropped_mammogram_path,
        metadata_path=args.metadata_path,
    )


if __name__ == "__main__":
    main()
