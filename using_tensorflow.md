## Using TensorFlow 

#### Single Image (TensorFlow)

We have ported our models to TensorFlow to facilitate usage by other teams. 

For the cancer classification model, the model has been ported entirely to TensorFlow within this repository.

For the patch classification model, we require the DenseNet code from [here](https://github.com/pudae/tensorflow-densenet). Simply git clone the linked repository (e.g. to `/path/to/tensorflow-densenet`) and add the path to your  `PYTHONPATH`. 

```bash
export PYTHONPATH=/path/to/tensorflow-densenet:$PYTHONPATH
```

Both the cancer classification models and patch classification models are called with the same Python scripts as the PyTorch versions, except with a `_tf` suffix (e.g. `run_models_single_tf.py`). See `run_single_tf.sh` for a full TensorFlow-based pipeline.  

In addition, the cancer classification models and patch classification models are also independent and interchangeable with their PyTorch equivalents, so you can use e.g. the PyTorch-generated heatmaps with the TensorFlow classifier. 

## Additional Prerequisites

* TensorFlow (1.13.1)
* [pudae/tensorflow-densenet (ab41685)](https://github.com/pudae/tensorflow-densenet)

## Running 

Using `run_single_tf.sh`, you should get the following output:

```bash
bash run_single_tf.sh "sample_data/images/0_L_CC.png" "L-CC"
``` 

```
Stage 4a: Run Classifier (Image)
{"benign": 0.04019179195165634, "malignant": 0.008045281283557415}
Stage 4b: Run Classifier (Image+Heatmaps)
{"benign": 0.05236595869064331, "malignant": 0.005510156974196434}
```

The outputs should be nearly identical to those of the PyTorch models.
