# Combined Soft NMS Custom Tensorflow Op

- tf provides a per class soft nms op
- making it work for all classes is pretty tedious and slow

- provides custom tf op for cpu 
  - basically a combination of existing combined and soft nms ops

## Setup

### Build PIP Package

Works only with make

With Makefile:

```bash
  make zero_out_pip_pkg
```

### Install and Test PIP Package

Once the pip package has been built, you can install it with,

```bash
pip3 install artifacts/*.whl
```

Then test out the pip package

```bash
cd ..
python3 -c "import tensorflow as tf;import tensorflow_zero_out;print(tensorflow_zero_out.zero_out([[1,2], [3,4]]))"
```

And you should see the op zeroed out all input elements except the first one:

```bash
[[1 0]
 [0 0]]
```

## example

- download & install easy-efficientdet
- download street image from somewhere
- download pre-trained coco model
- run inference

## issues

- bugs / a few detections seem unnecassary

## Backlog

- unit tests
- optimzations
  - remove anchor anchor boxes where max prob is below threshold
  - run per class nms only for subset of anchor boxes where respective class has the max value
- GPU implementation
