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
import collections as col
import numpy as np
import os
import pandas as pd
import torch
import tqdm

import src.utilities.pickling as pickling
import src.utilities.tools as tools
import src.modeling.models as models
import src.data_loading.loading as loading
from src.constants import VIEWS, VIEWANGLES, LABELS, MODELMODES


def load_model(parameters):
    """
    Loads trained cancer classifier
    """
    input_channels = 3 if parameters["use_heatmaps"] else 1
    model_class = {
        MODELMODES.VIEW_SPLIT: models.SplitBreastModel,
        MODELMODES.IMAGE: models.ImageBreastModel,
    }[parameters["model_mode"]]
    model = model_class(input_channels)
    model.load_state_dict(torch.load(parameters["model_path"])["model"])

    if (parameters["device_type"] == "gpu") and torch.has_cudnn:
        device = torch.device("cuda:{}".format(parameters["gpu_number"]))
    else:
        device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    return model, device


def run_model(model, device, exam_list, parameters):
    """
    Returns predictions of image only model or image+heatmaps model.
    Prediction for each exam is averaged for a given number of epochs.
    """
    random_number_generator = np.random.RandomState(parameters["seed"])

    image_extension = ".hdf5" if parameters["use_hdf5"] else ".png"

    with torch.no_grad():
        predictions_ls = []
        for datum in tqdm.tqdm(exam_list):
            predictions_for_datum = []
            loaded_image_dict = {view: [] for view in VIEWS.LIST}
            loaded_heatmaps_dict = {view: [] for view in VIEWS.LIST}
            for view in VIEWS.LIST:
                for short_file_path in datum[view]:
                    loaded_image = loading.load_image(
                        image_path=os.path.join(parameters["image_path"], short_file_path + image_extension),
                        view=view,
                        horizontal_flip=datum["horizontal_flip"],
                    )
                    if parameters["use_heatmaps"]:
                        loaded_heatmaps = loading.load_heatmaps(
                            benign_heatmap_path=os.path.join(parameters["heatmaps_path"], "heatmap_benign",
                                                             short_file_path + ".hdf5"),
                            malignant_heatmap_path=os.path.join(parameters["heatmaps_path"], "heatmap_malignant",
                                                                short_file_path + ".hdf5"),
                            view=view,
                            horizontal_flip=datum["horizontal_flip"],
                        )
                    else:
                        loaded_heatmaps = None

                    loaded_image_dict[view].append(loaded_image)
                    loaded_heatmaps_dict[view].append(loaded_heatmaps)
            for data_batch in tools.partition_batch(range(parameters["num_epochs"]), parameters["batch_size"]):
                batch_dict = {view: [] for view in VIEWS.LIST}
                for _ in data_batch:
                    for view in VIEWS.LIST:
                        image_index = 0
                        if parameters["augmentation"]:
                            image_index = random_number_generator.randint(low=0, high=len(datum[view]))
                        cropped_image, cropped_heatmaps = loading.augment_and_normalize_image(
                            image=loaded_image_dict[view][image_index],
                            auxiliary_image=loaded_heatmaps_dict[view][image_index],
                            view=view,
                            best_center=datum["best_center"][view][image_index],
                            random_number_generator=random_number_generator,
                            augmentation=parameters["augmentation"],
                            max_crop_noise=parameters["max_crop_noise"],
                            max_crop_size_noise=parameters["max_crop_size_noise"],
                        )
                        if loaded_heatmaps_dict[view][image_index] is None:
                            batch_dict[view].append(cropped_image[:, :, np.newaxis])
                        else:
                            batch_dict[view].append(np.concatenate([
                                cropped_image[:, :, np.newaxis],
                                cropped_heatmaps,
                            ], axis=2))

                tensor_batch = {
                    view: torch.tensor(np.stack(batch_dict[view])).permute(0, 3, 1, 2).to(device)
                    for view in VIEWS.LIST
                }
                output = model(tensor_batch)
                batch_predictions = compute_batch_predictions(output, mode=parameters["model_mode"])
                pred_df = pd.DataFrame({k: v[:, 1] for k, v in batch_predictions.items()})
                pred_df.columns.names = ["label", "view_angle"]
                predictions = pred_df.T.reset_index().groupby("label").mean().T[LABELS.LIST].values
                predictions_for_datum.append(predictions)
            predictions_ls.append(np.mean(np.concatenate(predictions_for_datum, axis=0), axis=0))

    return np.array(predictions_ls)


def compute_batch_predictions(y_hat, mode):
    """
    Format predictions from different heads
    """

    if mode == MODELMODES.VIEW_SPLIT:
        assert y_hat[VIEWANGLES.CC].shape[1:] == (4, 2)
        assert y_hat[VIEWANGLES.MLO].shape[1:] == (4, 2)
        batch_prediction_tensor_dict = col.OrderedDict()
        batch_prediction_tensor_dict[LABELS.LEFT_BENIGN, VIEWANGLES.CC] = y_hat[VIEWANGLES.CC][:, 0]
        batch_prediction_tensor_dict[LABELS.LEFT_BENIGN, VIEWANGLES.MLO] = y_hat[VIEWANGLES.MLO][:, 0]
        batch_prediction_tensor_dict[LABELS.RIGHT_BENIGN, VIEWANGLES.CC] = y_hat[VIEWANGLES.CC][:, 1]
        batch_prediction_tensor_dict[LABELS.RIGHT_BENIGN, VIEWANGLES.MLO] = y_hat[VIEWANGLES.MLO][:, 1]
        batch_prediction_tensor_dict[LABELS.LEFT_MALIGNANT, VIEWANGLES.CC] = y_hat[VIEWANGLES.CC][:, 2]
        batch_prediction_tensor_dict[LABELS.LEFT_MALIGNANT, VIEWANGLES.MLO] = y_hat[VIEWANGLES.MLO][:, 2]
        batch_prediction_tensor_dict[LABELS.RIGHT_MALIGNANT, VIEWANGLES.CC] = y_hat[VIEWANGLES.CC][:, 3]
        batch_prediction_tensor_dict[LABELS.RIGHT_MALIGNANT, VIEWANGLES.MLO] = y_hat[VIEWANGLES.MLO][:, 3]
        batch_prediction_dict = col.OrderedDict([
            (k, np.exp(v.cpu().detach().numpy()))
            for k, v in batch_prediction_tensor_dict.items()
        ])
    elif mode == MODELMODES.IMAGE:
        assert y_hat[VIEWS.L_CC].shape[1:] == (2, 2)
        assert y_hat[VIEWS.R_CC].shape[1:] == (2, 2)
        assert y_hat[VIEWS.L_MLO].shape[1:] == (2, 2)
        assert y_hat[VIEWS.R_MLO].shape[1:] == (2, 2)
        batch_prediction_tensor_dict = col.OrderedDict()
        batch_prediction_tensor_dict[LABELS.LEFT_BENIGN, VIEWS.L_CC] = y_hat[VIEWS.L_CC][:, 0]
        batch_prediction_tensor_dict[LABELS.LEFT_BENIGN, VIEWS.L_MLO] = y_hat[VIEWS.L_MLO][:, 0]
        batch_prediction_tensor_dict[LABELS.RIGHT_BENIGN, VIEWS.R_CC] = y_hat[VIEWS.R_CC][:, 0]
        batch_prediction_tensor_dict[LABELS.RIGHT_BENIGN, VIEWS.R_MLO] = y_hat[VIEWS.R_MLO][:, 0]
        batch_prediction_tensor_dict[LABELS.LEFT_MALIGNANT, VIEWS.L_CC] = y_hat[VIEWS.L_CC][:, 1]
        batch_prediction_tensor_dict[LABELS.LEFT_MALIGNANT, VIEWS.L_MLO] = y_hat[VIEWS.L_MLO][:, 1]
        batch_prediction_tensor_dict[LABELS.RIGHT_MALIGNANT, VIEWS.R_CC] = y_hat[VIEWS.R_CC][:, 1]
        batch_prediction_tensor_dict[LABELS.RIGHT_MALIGNANT, VIEWS.R_MLO] = y_hat[VIEWS.R_MLO][:, 1]

        batch_prediction_dict = col.OrderedDict([
            (k, np.exp(v.cpu().detach().numpy()))
            for k, v in batch_prediction_tensor_dict.items()
        ])
    else:
        raise KeyError(mode)
    return batch_prediction_dict


def load_run_save(data_path, output_path, parameters):
    """
    Outputs the predictions as csv file
    """
    exam_list = pickling.unpickle_from_file(data_path)
    model, device = load_model(parameters)
    predictions = run_model(model, device, exam_list, parameters)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Take the positive prediction
    df = pd.DataFrame(predictions, columns=LABELS.LIST)
    df.to_csv(output_path, index=False, float_format='%.4f')


def main():
    parser = argparse.ArgumentParser(description='Run image-only model or image+heatmap model')
    parser.add_argument('--model-mode', default=MODELMODES.VIEW_SPLIT, type=str)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--image-path', required=True)
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--use-heatmaps', action="store_true")
    parser.add_argument('--heatmaps-path')
    parser.add_argument('--use-augmentation', action="store_true")
    parser.add_argument('--use-hdf5', action="store_true")
    parser.add_argument('--num-epochs', default=1, type=int)
    parser.add_argument('--device-type', default="cpu", choices=['gpu', 'cpu'])
    parser.add_argument("--gpu-number", type=int, default=0)
    args = parser.parse_args()

    parameters = {
        "device_type": args.device_type,
        "gpu_number": args.gpu_number,
        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "image_path": args.image_path,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "augmentation": args.use_augmentation,
        "num_epochs": args.num_epochs,
        "use_heatmaps": args.use_heatmaps,
        "heatmaps_path": args.heatmaps_path,
        "use_hdf5": args.use_hdf5,
        "model_mode": args.model_mode,
        "model_path": args.model_path,
    }
    load_run_save(
        data_path=args.data_path,
        output_path=args.output_path,
        parameters=parameters,
    )


if __name__ == "__main__":
    main()
