from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

combined_soft_nms_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_combined_soft_nms_ops.so'))
combined_soft_nms = combined_soft_nms_ops.combined_soft_non_max_suppression