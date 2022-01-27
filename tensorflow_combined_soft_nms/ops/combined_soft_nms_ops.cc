/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

namespace {

Status CombinedNMSShapeFn(InferenceContext* c) {
  // Get inputs and validate ranks
  ShapeHandle boxes;
  // boxes is a tensor of Dimensions [batch_size, num_anchors, q, 4]
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &boxes));
  ShapeHandle scores;
  // scores is a tensor of Dimensions [batch_size, num_anchors, num_classes]
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &scores));
  ShapeHandle max_output_size_per_class;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &max_output_size_per_class));
  ShapeHandle max_total_size;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &max_total_size));
  ShapeHandle unused_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused_shape));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused_shape));

  DimensionHandle unused;
  // boxes[0] and scores[0] are both batch_size
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(boxes, 0), c->Dim(scores, 0), &unused));
  // boxes[1] and scores[1] are both num_anchors
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(boxes, 1), c->Dim(scores, 1), &unused));
  // The boxes[3] is 4.
  TF_RETURN_IF_ERROR(c->WithValue(c->Dim(boxes, 3), 4, &unused));

  DimensionHandle d = c->Dim(boxes, 2);
  DimensionHandle class_dim = c->Dim(scores, 2);
  if (c->ValueKnown(d) && c->ValueKnown(class_dim)) {
    if (c->Value(d) != 1 && c->Value(d) != c->Value(class_dim)) {
      return errors::InvalidArgument(
          "third dimension of boxes must be either "
          "1 or equal to the third dimension of scores");
    }
  }
  DimensionHandle output_dim;
  DimensionHandle batch_dim = c->Dim(boxes, 0);

  TF_RETURN_IF_ERROR(c->MakeDimForScalarInput(3, &output_dim));
  if (c->ValueKnown(output_dim) && c->Value(output_dim) <= 0) {
    return errors::InvalidArgument("max_total_size should be > 0 ");
  }
  DimensionHandle size_per_class;
  TF_RETURN_IF_ERROR(c->MakeDimForScalarInput(2, &size_per_class));

  int64_t output_size;
  bool pad_per_class;
  TF_RETURN_IF_ERROR(c->GetAttr("pad_per_class", &pad_per_class));
  if (!pad_per_class) {
    output_size = c->Value(output_dim);
  } else {
    if (c->ValueKnown(size_per_class) && c->Value(size_per_class) <= 0) {
      return errors::InvalidArgument(
          "max_output_size_per_class must be > 0 "
          "if pad_per_class is set to true ");
    }
    output_size = std::min(c->Value(output_dim),
                           c->Value(size_per_class) * c->Value(class_dim));
  }
  c->set_output(0, c->MakeShape({batch_dim, output_size, 4}));
  c->set_output(1, c->MakeShape({batch_dim, output_size}));
  c->set_output(2, c->MakeShape({batch_dim, output_size}));
  c->set_output(3, c->Vector(batch_dim));
  return Status::OK();
}    

} // namespace

// register custom op
REGISTER_OP("CombinedSoftNonMaxSuppression")
    .Input("boxes: float")
    .Input("scores: float")
    .Input("max_output_size_per_class: int32")
    .Input("max_total_size: int32")
    .Input("iou_threshold: float")
    .Input("score_threshold: float")
    .Input("soft_nms_sigma: float")
    .Output("nmsed_boxes: float")
    .Output("nmsed_scores: float")
    .Output("nmsed_classes: float")
    .Output("valid_detections: int32")
    .Attr("pad_per_class: bool = false")
    .Attr("clip_boxes: bool = true")
    .SetShapeFn(CombinedNMSShapeFn);

} // namespace tensorflow
