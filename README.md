# Combined Soft NMS Custom Tensorflow Op

This is an custom tensorflow op which performs soft non max suppression ([Improving Object Detection With One Line of Code](https://arxiv.org/abs/1704.04503)) on multiple batches and classes.
Basically, it is a merge between the two existing ops `tensorflow::ops::NonMaxSuppressionV5` and `tensorflow::ops::CombinedNonMaxSuppression` with minor adjustments.

## Setup

This build works only with Make.

### Build PIP Package

```bash
  make clean combined_soft_nms_op combined_soft_nms_pip_pkg
```

### Install and Test PIP Package

Once the pip package has been built, you can install it with,

```bash
python -m pip install artifacts/*.whl
```

Then test out the pip package

```bash
cd ..
python -c "import tensorflow as tf; from tensorflow_combined_soft_nms.python.ops.combined_soft_nms_ops import combined_soft_nms;  tf.print(combined_soft_nms(tf.random.uniform((2,2,1,4)), tf.random.uniform((2,2,3)), 5, 5, .5, .35, .5))"
```

And you should see the following output:

```bash
CombinedSoftNonMaxSuppression(nmsed_boxes=[[[0.932438612 0.701503754 0.459562063 0.256728888]
  [0.932438612 0.701503754 0.459562063 0.256728888]
  [0.932438612 0.701503754 0.459562063 0.256728888]
  [0.62079227 0.363695264 0.566833496 0.850822449]
  [0.62079227 0.363695264 0.566833496 0.850822449]]

 [[0.896504402 0.438400865 0.47448051 0.686301827]
  [0.896504402 0.438400865 0.47448051 0.686301827]
  [0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]]], nmsed_scores=[[0.925052881 0.80048728 0.659887195 0.398622751 0.397154897]
 [0.777113676 0.495211959 0 0 0]], nmsed_classes=[[2 0 1 0 2]
 [1 0 0 0 0]], valid_detections=[5 2])
```

## Colab Example

Here, you can find a colab with a real inference example.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waldemarmeier/combined-soft-nms-op/blob/master/examples/custom_soft_nms_op_example.ipynb)

## Backlog

- add a Python wrapper with a inline documentation
- more extensive unit tests
- optimzations
  - remove anchor anchor boxes where max prob is below threshold
  - run per class nms only for subset of anchor boxes where respective class has the max value
- GPU implementation (maybe not useful due to the traits of the algorithm)
