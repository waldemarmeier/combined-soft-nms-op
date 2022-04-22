/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"

#include <queue>

// most of this code is just a combination of 
// 'combined-nms' and 'nmsv5' kernels code from tensorflow
namespace tensorflow {
namespace {

typedef Eigen::ThreadPoolDevice CPUDevice;

static inline void CheckScoreSizes(OpKernelContext* context, int num_boxes,
                                   const Tensor& scores) {
  // The shape of 'scores' is [num_boxes]
  OP_REQUIRES(context, scores.dims() == 1,
              errors::InvalidArgument(
                  "scores must be 1-D", scores.shape().DebugString(),
                  " (Shape must be rank 1 but is rank ", scores.dims(), ")"));
  OP_REQUIRES(
      context, scores.dim_size(0) == num_boxes,
      errors::InvalidArgument("scores has incompatible shape (Dimensions must "
                              "be equal, but are ",
                              num_boxes, " and ", scores.dim_size(0), ")"));
}

static inline void ParseAndCheckOverlapSizes(OpKernelContext* context,
                                             const Tensor& overlaps,
                                             int* num_boxes) {
  // the shape of 'overlaps' is [num_boxes, num_boxes]
  OP_REQUIRES(context, overlaps.dims() == 2,
              errors::InvalidArgument("overlaps must be 2-D",
                                      overlaps.shape().DebugString()));

  *num_boxes = overlaps.dim_size(0);
  OP_REQUIRES(context, overlaps.dim_size(1) == *num_boxes,
              errors::InvalidArgument("overlaps must be square",
                                      overlaps.shape().DebugString()));
}

static inline void ParseAndCheckBoxSizes(OpKernelContext* context,
                                         const Tensor& boxes, int* num_boxes) {
  // The shape of 'boxes' is [num_boxes, 4]
  OP_REQUIRES(context, boxes.dims() == 2,
              errors::InvalidArgument(
                  "boxes must be 2-D", boxes.shape().DebugString(),
                  " (Shape must be rank 2 but is rank ", boxes.dims(), ")"));
  *num_boxes = boxes.dim_size(0);
  OP_REQUIRES(context, boxes.dim_size(1) == 4,
              errors::InvalidArgument("boxes must have 4 columns (Dimension "
                                      "must be 4 but is ",
                                      boxes.dim_size(1), ")"));
}

static inline void CheckCombinedNMSScoreSizes(OpKernelContext* context,
                                              int num_boxes,
                                              const Tensor& scores) {
  // The shape of 'scores' is [batch_size, num_boxes, num_classes]
  OP_REQUIRES(context, scores.dims() == 3,
              errors::InvalidArgument("scores must be 3-D",
                                      scores.shape().DebugString()));
  OP_REQUIRES(context, scores.dim_size(1) == num_boxes,
              errors::InvalidArgument("scores has incompatible shape"));
}

static inline void ParseAndCheckCombinedNMSBoxSizes(OpKernelContext* context,
                                                    const Tensor& boxes,
                                                    int* num_boxes,
                                                    const int num_classes) {
  // The shape of 'boxes' is [batch_size, num_boxes, q, 4]
  OP_REQUIRES(context, boxes.dims() == 4,
              errors::InvalidArgument("boxes must be 4-D",
                                      boxes.shape().DebugString()));

  bool box_check = boxes.dim_size(2) == 1 || boxes.dim_size(2) == num_classes;
  OP_REQUIRES(context, box_check,
              errors::InvalidArgument(
                  "third dimension of boxes must be either 1 or num classes"));
  *num_boxes = boxes.dim_size(1);
  OP_REQUIRES(context, boxes.dim_size(3) == 4,
              errors::InvalidArgument("boxes must have 4 columns"));
}

struct ResultCandidate {
  int box_index;
  float score;
  int class_idx;
  float box_coord[4];
};

typedef std::function<ResultCandidate(const int, const float, const int, const float*)> 
  ResultCandiateCreator;


static inline ResultCandidate createResultCandidate(
                              const int box_idx,
                              const float score,
                              const int class_idx,
                              const float* boxes_data) {

    const float* box_coord_start_ptr = (boxes_data 
                                      + (box_idx * 4));
    return {
      box_idx,
      score,
      class_idx,
      {*box_coord_start_ptr, *(box_coord_start_ptr + 1),
       *(box_coord_start_ptr + 2), *(box_coord_start_ptr + 3)
       }
    };
}

static inline ResultCandidate createResultCandidate(
                              const int box_idx,
                              const float score,
                              const int class_idx,
                              const float* boxes_data,
                              const int q) {
        
    const float* box_coord_start_ptr = (boxes_data 
                                        + (box_idx * q * 4)
                                        + (class_idx * 4));
    return {
      box_idx,
      score,
      class_idx,
      {*box_coord_start_ptr, *(box_coord_start_ptr + 1),
       *(box_coord_start_ptr + 2), *(box_coord_start_ptr + 3)
       }
    };
}

static inline ResultCandiateCreator CreateResultCandiateCreatorFn(
  const int q
) {
  if (q > 1){
    return std::bind(static_cast<ResultCandidate (*)(const int, const float, const int, const float*, int)>(&createResultCandidate),
                     std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, q);
  } else {
    return std::bind(static_cast<ResultCandidate (*)(const int, const float, const int, const float*)>(&createResultCandidate), 
                      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
  }
}

// adjusted, considers q dimension in combined nms op
// Return intersection-over-union overlap between boxes i and j
template <typename T>
static inline float IOU(typename TTypes<T, 3>::ConstTensor boxes, const int q, 
                        const int i, int j) {

  auto boxes_ = [&](int i_, int j_) {
    return boxes(i_, q, j_);
  };

  const float ymin_i = Eigen::numext::mini<float>(boxes_(i, 0), boxes_(i, 2));
  const float xmin_i = Eigen::numext::mini<float>(boxes_(i, 1), boxes_(i, 3));
  const float ymax_i = Eigen::numext::maxi<float>(boxes_(i, 0), boxes_(i, 2));
  const float xmax_i = Eigen::numext::maxi<float>(boxes_(i, 1), boxes_(i, 3));
  const float ymin_j = Eigen::numext::mini<float>(boxes_(j, 0), boxes_(j, 2));
  const float xmin_j = Eigen::numext::mini<float>(boxes_(j, 1), boxes_(j, 3));
  const float ymax_j = Eigen::numext::maxi<float>(boxes_(j, 0), boxes_(j, 2));
  const float xmax_j = Eigen::numext::maxi<float>(boxes_(j, 1), boxes_(j, 3));
  const float area_i = (ymax_i - ymin_i) * (xmax_i - xmin_i);
  const float area_j = (ymax_j - ymin_j) * (xmax_j - xmin_j);
  if (area_i <= 0 || area_j <= 0) {
    return 0.0;
  }
  const float intersection_ymin = Eigen::numext::maxi<float>(ymin_i, ymin_j);
  const float intersection_xmin = Eigen::numext::maxi<float>(xmin_i, xmin_j);
  const float intersection_ymax = Eigen::numext::mini<float>(ymax_i, ymax_j);
  const float intersection_xmax = Eigen::numext::mini<float>(xmax_i, xmax_j);
  const float intersection_area =
      Eigen::numext::maxi<float>(intersection_ymax - intersection_ymin, 0.0) *
      Eigen::numext::maxi<float>(intersection_xmax - intersection_xmin, 0.0);
  return intersection_area / (area_i + area_j - intersection_area);
}

// also considers q
template <typename T>
static inline std::function<float(int, int)> CreateIOUSimilarityFn(
    const Tensor& boxes, const int q) {
  typename TTypes<T, 3>::ConstTensor boxes_data = boxes.tensor<T, 3>();
  return std::bind(&IOU<T>, boxes_data, q, std::placeholders::_1,
                   std::placeholders::_2);
}


void SelectResultPerBatch(std::vector<float>& nmsed_boxes,
                          std::vector<float>& nmsed_scores,
                          std::vector<float>& nmsed_classes,
                          std::vector<ResultCandidate>& result_candidate_vec,
                          std::vector<int>& final_valid_detections,
                          const int batch_idx,
                          int total_size_per_batch,
                          bool pad_per_class,
                          int max_size_per_batch,
                          bool clip_boxes,
                          int per_batch_size) {
  auto rc_cmp = [](const ResultCandidate rc_i, const ResultCandidate rc_j) {
    return rc_i.score > rc_j.score;
  };
  std::sort(result_candidate_vec.begin(), result_candidate_vec.end(), rc_cmp);

  int max_detections = 0;
  int result_candidate_size =
      std::count_if(result_candidate_vec.begin(), result_candidate_vec.end(),
                    [](ResultCandidate rc) { return rc.box_index > -1; });
  // If pad_per_class is false, we always pad to max_total_size
  if (!pad_per_class) {
    max_detections = std::min(result_candidate_size, total_size_per_batch);
  } else {
    max_detections = std::min(per_batch_size, result_candidate_size);
  }

  final_valid_detections[batch_idx] = max_detections;

  int curr_total_size = max_detections;
  int result_idx = 0;
  // Pick the top max_detections values
  while (curr_total_size > 0 && result_idx < result_candidate_vec.size()) {
    ResultCandidate next_candidate = result_candidate_vec[result_idx++];
    // Add to final output vectors
    if (clip_boxes) {
      const float box_min = 0.0;
      const float box_max = 1.0;
      nmsed_boxes.push_back(
          std::max(std::min(next_candidate.box_coord[0], box_max), box_min));
      nmsed_boxes.push_back(
          std::max(std::min(next_candidate.box_coord[1], box_max), box_min));
      nmsed_boxes.push_back(
          std::max(std::min(next_candidate.box_coord[2], box_max), box_min));
      nmsed_boxes.push_back(
          std::max(std::min(next_candidate.box_coord[3], box_max), box_min));
    } else {
      nmsed_boxes.push_back(next_candidate.box_coord[0]);
      nmsed_boxes.push_back(next_candidate.box_coord[1]);
      nmsed_boxes.push_back(next_candidate.box_coord[2]);
      nmsed_boxes.push_back(next_candidate.box_coord[3]);
    }
    nmsed_scores.push_back(next_candidate.score);
    nmsed_classes.push_back(next_candidate.class_idx);
    curr_total_size--;
  }

  nmsed_boxes.resize(per_batch_size * 4, 0);
  nmsed_scores.resize(per_batch_size, 0);
  nmsed_classes.resize(per_batch_size, 0);
}


//  adjusted version of 'DoNonMaxSuppressionOp'
template <typename T>
void DoNonMaxSuppressionOpV2(
  const int class_idx,
  const float* boxes_data,
  // const Tensor& scores,
  std::vector<T>& scores_data,
  const int num_boxes,
  // const Tensor& max_output_size,
  const int output_size,
  const T similarity_threshold,
  const T score_threshold,
  const T soft_nms_sigma,
  const ResultCandiateCreator create_result_cand_fn,
  const std::function<float(int, int)>& similarity_fn,
  typename std::vector<ResultCandidate>& candidate_vec,
  bool return_scores_tensor = false,
  bool pad_to_max_output_size = false,
  int* ptr_num_valid_outputs = nullptr) {

  struct Candidate {
    int box_index;
    T score;
    int suppress_begin_index;
  };

  auto cmp = [](const Candidate bs_i, const Candidate bs_j) {
    return ((bs_i.score == bs_j.score) && (bs_i.box_index > bs_j.box_index)) ||
           bs_i.score < bs_j.score;
  };
  std::priority_queue<Candidate, std::deque<Candidate>, decltype(cmp)>
      candidate_priority_queue(cmp);
  for (int i = 0; i < scores_data.size(); ++i) {
    if (scores_data[i] > score_threshold) {
      candidate_priority_queue.emplace(Candidate({i, scores_data[i], 0}));
    }
  }

  T scale = static_cast<T>(0.0);
  bool is_soft_nms = soft_nms_sigma > static_cast<T>(0.0);
  if (is_soft_nms) {
    scale = static_cast<T>(-0.5) / soft_nms_sigma;
  }

  auto suppress_weight = [similarity_threshold, scale,
                          is_soft_nms](const T sim) {
    const T weight = Eigen::numext::exp<T>(scale * sim * sim);
    return is_soft_nms || sim <= similarity_threshold ? weight
                                                      : static_cast<T>(0.0);
  };

  // @TODO remove following two if necessary
  // @TODO figure out how to deal with padding
  std::vector<int> selected;
  std::vector<T> selected_scores;
  
  float similarity;
  T original_score;
  Candidate next_candidate;

  int selection_idx = 0;
  while (selected.size() < output_size && !candidate_priority_queue.empty()) {
    next_candidate = candidate_priority_queue.top();
    original_score = next_candidate.score;
    candidate_priority_queue.pop();

    // Overlapping boxes are likely to have similar scores, therefore we
    // iterate through the previously selected boxes backwards in order to
    // see if `next_candidate` should be suppressed. We also enforce a property
    // that a candidate can be suppressed by another candidate no more than
    // once via `suppress_begin_index` which tracks which previously selected
    // boxes have already been compared against next_candidate prior to a given
    // iteration.  These previous selected boxes are then skipped over in the
    // following loop.
    bool should_hard_suppress = false;
    for (int j = static_cast<int>(selected.size()) - 1;
         j >= next_candidate.suppress_begin_index; --j) {
      similarity = similarity_fn(next_candidate.box_index, selected[j]);

      next_candidate.score *= suppress_weight(static_cast<T>(similarity));

      // First decide whether to perform hard suppression
      if (!is_soft_nms && static_cast<T>(similarity) > similarity_threshold) {
        should_hard_suppress = true;
        break;
      }

      // If next_candidate survives hard suppression, apply soft suppression
      if (next_candidate.score <= score_threshold) break;
    }
    // If `next_candidate.score` has not dropped below `score_threshold`
    // by this point, then we know that we went through all of the previous
    // selections and can safely update `suppress_begin_index` to
    // `selected.size()`. If on the other hand `next_candidate.score`
    // *has* dropped below the score threshold, then since `suppress_weight`
    // always returns values in [0, 1], further suppression by items that were
    // not covered in the above for loop would not have caused the algorithm
    // to select this item. We thus do the same update to
    // `suppress_begin_index`, but really, this element will not be added back
    // into the priority queue in the following.
    next_candidate.suppress_begin_index = selected.size();

    if (!should_hard_suppress) {
      if (next_candidate.score == original_score) {
        candidate_vec[output_size * class_idx + selection_idx] = create_result_cand_fn(
            next_candidate.box_index, next_candidate.score, class_idx, boxes_data);
        selection_idx++;
        // Suppression has not occurred, so select next_candidate
        selected.push_back(next_candidate.box_index);
        selected_scores.push_back(next_candidate.score);
        continue;
      }
      if (next_candidate.score > score_threshold) {
        // Soft suppression has occurred and current score is still greater than
        // score_threshold; add next_candidate back onto priority queue.
        candidate_priority_queue.push(next_candidate);
      }
    }
  }

  int num_valid_outputs = selected.size();
  // std::printf("%d", num_valid_outputs);

  if (pad_to_max_output_size) {
    selected.resize(output_size, 0);
    selected_scores.resize(output_size, static_cast<T>(0));
  }
  if (ptr_num_valid_outputs) {
    *ptr_num_valid_outputs = num_valid_outputs;
  }
}

void BatchedNonMaxSuppressionOp(
    OpKernelContext* context, 
    const Tensor& inp_boxes, 
    const Tensor& inp_scores,
    int num_boxes,
    const int max_size_per_class,
    const int total_size_per_batch,
    const float score_threshold, 
    const float iou_threshold,
    const float soft_nms_sigma,
    bool pad_per_class = false,
    bool clip_boxes = true) {
  const int num_batches = inp_boxes.dim_size(0);
  int num_classes = inp_scores.dim_size(2);
  int q = inp_boxes.dim_size(2);

  const float* scores_data =
      const_cast<float*>(inp_scores.flat<float>().data());
  const float* boxes_data = const_cast<float*>(inp_boxes.flat<float>().data());

  int boxes_per_batch = num_boxes * q * 4;
  int scores_per_batch = num_boxes * num_classes;
  const int size_per_class = std::min(max_size_per_class, num_boxes);
  std::vector<std::vector<ResultCandidate>> result_candidate_vec(
      num_batches,
      std::vector<ResultCandidate>(size_per_class * num_classes,
                                   {-1, -1.0, -1, {0.0, 0.0, 0.0, 0.0}}));

  // [num_batches, per_batch_size * 4]
  std::vector<std::vector<float>> nmsed_boxes(num_batches);
  // [num_batches, per_batch_size]
  std::vector<std::vector<float>> nmsed_scores(num_batches);
  // [num_batches, per_batch_size]
  std::vector<std::vector<float>> nmsed_classes(num_batches);
  // results
  // [num_batches]
  std::vector<int> final_valid_detections(num_batches);

  auto create_result_cand_fn = CreateResultCandiateCreatorFn(q);
  auto shard_nms = [&](int begin, int end) {
    for (int idx = begin; idx < end; ++idx) {
      int batch_idx = idx / num_classes;
      int class_idx = idx % num_classes;

      const Tensor& batch_boxes = inp_boxes.SubSlice(batch_idx);
      const int q_actual = (q > 1) ? class_idx : 0;
      auto similarity_fn = CreateIOUSimilarityFn<float>(batch_boxes, q_actual);

      // [num_boxes, num_cls]
      std::vector<float> scores_data_vec(num_boxes);
      
      for(int box_idx = 0; box_idx < num_boxes; box_idx++) {
        // start
        //  - go to batch_idx (batch_idx * scores_per_batch)
        //  - go to right box_idx (box_idx * num_classes)
        //  - go to right class class_idx
        scores_data_vec[box_idx] = *(scores_data  
                                      + scores_per_batch * batch_idx 
                                      + box_idx * num_classes
                                      + class_idx);
        // scores_data[box_idx] = inp_scores(batch_idx, box_idx, class_idx);
      }

      int num_valid_outputs = 0;

      DoNonMaxSuppressionOpV2<float>(
        class_idx,
        boxes_data + boxes_per_batch * batch_idx, // pointer to start of boxes
        scores_data_vec,
        num_boxes,
        size_per_class,
        iou_threshold,
        score_threshold, 
        soft_nms_sigma, 
        create_result_cand_fn,
        similarity_fn, 
        result_candidate_vec[batch_idx],
        true,
        false, 
        &num_valid_outputs
      );
    }
  };

  int length = num_batches * num_classes;
  // Input data boxes_data, scores_data
  int input_bytes = num_boxes * 10 * sizeof(float);
  int output_bytes = num_boxes * 10 * sizeof(float);
  int compute_cycles = Eigen::TensorOpCost::AddCost<int>() * num_boxes * 14 +
                       Eigen::TensorOpCost::MulCost<int>() * num_boxes * 9 +
                       Eigen::TensorOpCost::MulCost<float>() * num_boxes * 9 +
                       Eigen::TensorOpCost::AddCost<float>() * num_boxes * 8;
  // The cost here is not the actual number of cycles, but rather a set of
  // hand-tuned numbers that seem to work best.
  const Eigen::TensorOpCost cost(input_bytes, output_bytes, compute_cycles);
  const CPUDevice& d = context->eigen_device<CPUDevice>();
  d.parallelFor(length, cost, shard_nms);


  // remove this, just for debugging
  for (auto& cands : result_candidate_vec){
    for (auto& cand : cands) {

      if (cand.box_index == -1){
        continue;
      }

      // printf("box_idx: %d, score: %F, class_idx: %d, coords {%F, %F, %F, %F} \n", 
      //                     cand.box_index, cand.score, cand.class_idx, cand.box_coord[0],
      //                     cand.box_coord[1], cand.box_coord[2], cand.box_coord[3]);
    }
  }

  int per_batch_size = total_size_per_batch;
  if (pad_per_class) {
    per_batch_size =
        std::min(total_size_per_batch, max_size_per_class * num_classes);
  }

  Tensor* valid_detections_t = nullptr;
  TensorShape valid_detections_shape({num_batches});
  OP_REQUIRES_OK(context, context->allocate_output(3, valid_detections_shape,
                                                   &valid_detections_t));
  auto valid_detections_flat = valid_detections_t->template flat<int>();

  auto shard_result = [&](int begin, int end) {
    for (int batch_idx = begin; batch_idx < end; ++batch_idx) {      
      SelectResultPerBatch(
          nmsed_boxes[batch_idx], nmsed_scores[batch_idx],
          nmsed_classes[batch_idx], result_candidate_vec[batch_idx],
          final_valid_detections, batch_idx, total_size_per_batch,
          pad_per_class, max_size_per_class * num_classes, clip_boxes,
          per_batch_size);
      valid_detections_flat(batch_idx) = final_valid_detections[batch_idx];
    }
  };
  length = num_batches;
  // Input data boxes_data, scores_data
  input_bytes =
      num_boxes * 10 * sizeof(float) + per_batch_size * 6 * sizeof(float);
  output_bytes =
      num_boxes * 5 * sizeof(float) + per_batch_size * 6 * sizeof(float);
  compute_cycles = Eigen::TensorOpCost::AddCost<int>() * num_boxes * 5 +
                   Eigen::TensorOpCost::AddCost<float>() * num_boxes * 5;
  // The cost here is not the actual number of cycles, but rather a set of
  // hand-tuned numbers that seem to work best.
  const Eigen::TensorOpCost cost_result(input_bytes, output_bytes,
                                        compute_cycles);
  d.parallelFor(length, cost_result, shard_result);

  Tensor* nmsed_boxes_t = nullptr;
  TensorShape boxes_shape({num_batches, per_batch_size, 4});
  OP_REQUIRES_OK(context,
                 context->allocate_output(0, boxes_shape, &nmsed_boxes_t));
  auto nmsed_boxes_flat = nmsed_boxes_t->template flat<float>();

  Tensor* nmsed_scores_t = nullptr;
  TensorShape scores_shape({num_batches, per_batch_size});
  OP_REQUIRES_OK(context,
                 context->allocate_output(1, scores_shape, &nmsed_scores_t));
  auto nmsed_scores_flat = nmsed_scores_t->template flat<float>();

  Tensor* nmsed_classes_t = nullptr;
  OP_REQUIRES_OK(context,
                 context->allocate_output(2, scores_shape, &nmsed_classes_t));
  auto nmsed_classes_flat = nmsed_classes_t->template flat<float>();

  auto shard_copy_result = [&](int begin, int end) {
    for (int idx = begin; idx < end; ++idx) {
      int batch_idx = idx / per_batch_size;
      int j = idx % per_batch_size;
      nmsed_scores_flat(idx) = nmsed_scores[batch_idx][j];
      nmsed_classes_flat(idx) = nmsed_classes[batch_idx][j];
      for (int k = 0; k < 4; ++k) {
        nmsed_boxes_flat(idx * 4 + k) = nmsed_boxes[batch_idx][j * 4 + k];
      }
    }
  };
  length = num_batches * per_batch_size;
  // Input data boxes_data, scores_data
  input_bytes = 6 * sizeof(float);
  output_bytes = 6 * sizeof(float);
  compute_cycles = Eigen::TensorOpCost::AddCost<int>() * 2 +
                   Eigen::TensorOpCost::MulCost<int>() * 2 +
                   Eigen::TensorOpCost::DivCost<float>() * 2;
  const Eigen::TensorOpCost cost_copy_result(input_bytes, output_bytes,
                                             compute_cycles);
  d.parallelFor(length, cost_copy_result, shard_copy_result);
}

} // namespace


// start custom-op, should work like combined nms op
// pretty much just a customized version of CombinedNonMaxSuppressionOp
template <typename Device>
class CombinedSoftNonMaxSuppressionOp : public OpKernel {
 public:
  explicit CombinedSoftNonMaxSuppressionOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("pad_per_class", &pad_per_class_));
    OP_REQUIRES_OK(context, context->GetAttr("clip_boxes", &clip_boxes_));
  }

  void Compute(OpKernelContext* context) override {
    // boxes: [batch_size, num_anchors, q, 4]
    const Tensor& boxes = context->input(0);
    // scores: [batch_size, num_anchors, num_classes]
    const Tensor& scores = context->input(1);
    OP_REQUIRES(
        context, (boxes.dim_size(0) == scores.dim_size(0)),
        errors::InvalidArgument("boxes and scores must have same batch size"));

    // max_output_size: scalar
    const Tensor& max_output_size = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(max_output_size.shape()),
        errors::InvalidArgument("max_size_per_class must be 0-D, got shape ",
                                max_output_size.shape().DebugString()));
    const int max_size_per_class = max_output_size.scalar<int>()();
    OP_REQUIRES(context, max_size_per_class > 0,
                errors::InvalidArgument("max_size_per_class must be positive"));
    // max_total_size: scalar
    const Tensor& max_total_size = context->input(3);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(max_total_size.shape()),
        errors::InvalidArgument("max_total_size must be 0-D, got shape ",
                                max_total_size.shape().DebugString()));
    const int max_total_size_per_batch = max_total_size.scalar<int>()();
    OP_REQUIRES(context, max_total_size_per_batch > 0,
                errors::InvalidArgument("max_total_size must be > 0"));
    // Throw warning when `max_total_size` is too large as it may cause OOM.
    if (max_total_size_per_batch > pow(10, 6)) {
      LOG(WARNING) << "Detected a large value for `max_total_size`. This may "
                   << "cause OOM error. (max_total_size: "
                   << max_total_size.scalar<int>()() << ")";
    }
    // iou_threshold: scalar
    const Tensor& iou_threshold = context->input(4);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(iou_threshold.shape()),
                errors::InvalidArgument("iou_threshold must be 0-D, got shape ",
                                        iou_threshold.shape().DebugString()));
    const float iou_threshold_val = iou_threshold.scalar<float>()();

    // score_threshold: scalar
    const Tensor& score_threshold = context->input(5);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(score_threshold.shape()),
        errors::InvalidArgument("score_threshold must be 0-D, got shape ",
                                score_threshold.shape().DebugString()));
    const float score_threshold_val = score_threshold.scalar<float>()();

    OP_REQUIRES(context, iou_threshold_val >= 0 && iou_threshold_val <= 1,
                errors::InvalidArgument("iou_threshold must be in [0, 1]"));
    
    // soft_nms_sigma: scalar
    const Tensor& soft_nms_sigma = context->input(6);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(soft_nms_sigma.shape()),
        errors::InvalidArgument("soft_nms_sigma must be 0-D, got shape ",
                                score_threshold.shape().DebugString()));
    const float soft_nms_sigma_val = soft_nms_sigma.scalar<float>()();
    
    int num_boxes = 0;
    const int num_classes = scores.dim_size(2);
    ParseAndCheckCombinedNMSBoxSizes(context, boxes, &num_boxes, num_classes);
    CheckCombinedNMSScoreSizes(context, num_boxes, scores);

    if (!context->status().ok()) {
      return;
    }
    BatchedNonMaxSuppressionOp(context, boxes, scores, num_boxes,
                               max_size_per_class, max_total_size_per_batch,
                               score_threshold_val, iou_threshold_val, soft_nms_sigma_val,
                               pad_per_class_, clip_boxes_);
  }

 private:
  bool pad_per_class_;
  bool clip_boxes_;
};

REGISTER_KERNEL_BUILDER(
  Name("CombinedSoftNonMaxSuppression").Device(DEVICE_CPU), 
    CombinedSoftNonMaxSuppressionOp<CPUDevice>);

} // namespace tensorflow