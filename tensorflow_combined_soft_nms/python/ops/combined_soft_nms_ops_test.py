import numpy as np
import tensorflow as tf

tf.get_logger().setLevel('DEBUG')

from tensorflow.python.platform import test
try:
    # if package is installed
    from tensorflow_combined_soft_nms.python.ops.combined_soft_nms_ops import combined_soft_nms
except ImportError:
    from combined_soft_nms_ops import combined_soft_nms

NMS_SIGMA = .5
MAX_OUTPUT_SIZE_PER_CLASS = 100
MAX_TOTAL_SIZE = 200
IOU_THRESHOLD = .4
SCORE_THRESHOLD = .3


# Bounding boxes are supplied as [y1, x1, y2, x2], where (y1, x1) and 
# (y2, x2) are the coordinates of any diagonal pair of box corners and 
# the coordinates can be provided as normalized (i.e., lying in the 
# interval [0, 1]) or absolute.

class CombinedSoftNMSTest(test.TestCase):

    def test_batched_one_class_simple(self):
        
        # two batches with same data
        # one class
        # two overlapping bboxes
        boxes =  tf.constant([[[[.1, .1, .9, .6]], [[.1, .9, .9, .4]]], 
                              [[[.1, .1, .9, .6]], [[.1, .9, .9, .4]]]])
        scores = tf.constant([[[.9], [.8]], [[.9], [.8]]])

        with self.session():

            pred_boxes, pred_scores, pred_classes, pred_valid_detections = \
                combined_soft_nms(
                    boxes,
                    scores,
                    MAX_OUTPUT_SIZE_PER_CLASS,
                    MAX_TOTAL_SIZE,
                    iou_threshold=IOU_THRESHOLD,
                    score_threshold=SCORE_THRESHOLD,
                    soft_nms_sigma=NMS_SIGMA,
                    pad_per_class=False,
                    clip_boxes=True,
                    name=None
                )

            # test bounding box predictions
            pred_boxes_goal = tf.pad(boxes[:,:,0,:],
                                constant_values=.0,
                                paddings=[[0,0], [0, MAX_TOTAL_SIZE - boxes.shape[1]], [0, 0]])
            pred_scores_goal = tf.pad(tf.constant([[0.9, 0.70599747], [0.9, 0.70599747]]),
                                        constant_values=.0,
                                        paddings=[[0, 0], [0, MAX_TOTAL_SIZE - boxes.shape[1]]])
            pred_classes_goal = tf.zeros((2, MAX_TOTAL_SIZE), dtype=tf.float32)

            self.assertAllClose(pred_boxes,  pred_boxes_goal)            
            self.assertAllClose(pred_scores,  pred_scores_goal)
            self.assertAllEqual(pred_classes,  pred_classes_goal)
            self.assertAllEqual(pred_valid_detections, tf.constant([2, 2], dtype=tf.int32))
    
    def test_batched_one_class_simple_padded(self):
        
        # two batches with same data
        # one class
        # two overlapping bboxes
        max_size_per_class = 150
        max_total_size = 9999
        num_detections_goal = 2
        boxes =  tf.constant([[[[.1, .1, .9, .6]], [[.1, .9, .9, .4]]], 
                              [[[.1, .1, .9, .6]], [[.1, .9, .9, .4]]]])
        scores = tf.constant([[[.9], [.8]], [[.9], [.8]]])

        with self.session():

            pred_boxes, pred_scores, pred_classes, pred_valid_detections = \
                combined_soft_nms(
                    boxes,
                    scores,
                    max_size_per_class,
                    max_total_size,
                    iou_threshold=IOU_THRESHOLD,
                    score_threshold=SCORE_THRESHOLD,
                    soft_nms_sigma=NMS_SIGMA,
                    pad_per_class=True,
                    clip_boxes=True,
                    name=None
                )

            # test bounding box predictions
            pred_boxes_goal = tf.pad(boxes[:,:,0,:],
                                constant_values=.0,
                                paddings=[[0,0], [0, max_size_per_class - num_detections_goal], [0, 0]])
            pred_scores_goal = tf.pad(tf.constant([[0.9, 0.70599747], [0.9, 0.70599747]]),
                                        constant_values=.0,
                                        paddings=[[0, 0], [0, max_size_per_class - num_detections_goal]])
            pred_classes_goal = tf.zeros((num_detections_goal, max_size_per_class), dtype=tf.float32)

            self.assertAllClose(pred_boxes,  pred_boxes_goal)            
            self.assertAllClose(pred_scores,  pred_scores_goal)
            self.assertAllEqual(pred_classes,  pred_classes_goal)
            self.assertAllEqual(pred_valid_detections, tf.constant([2, 2], dtype=tf.int32))

    def test_batched_one_class_simple_filter(self):
        
        # one class
        # two overlapping bboxes
        # considering the overlap the second prediction score is below the score threshold
        boxes =  tf.constant([[[[.1, .1, .9, .6]], [[.1, .9, .9, .4]]],
                              [[[.1, .1, .9, .6]], [[.1, .9, .9, .4]]]])
        scores = tf.constant([[[.9], [.3]], [[.9], [.3]]])
                
        with self.session():

            pred_boxes, pred_scores, pred_classes, pred_valid_detections = \
                combined_soft_nms(
                    boxes,
                    scores,
                    MAX_OUTPUT_SIZE_PER_CLASS,
                    MAX_TOTAL_SIZE,
                    iou_threshold=IOU_THRESHOLD,
                    score_threshold=SCORE_THRESHOLD,
                    soft_nms_sigma=NMS_SIGMA,
                    pad_per_class=False,
                    clip_boxes=True,
                    name=None
                )

            # test bounding box predictions
            pred_boxes_goal = tf.pad(boxes[:,:1,0,:],
                                constant_values=.0,
                                paddings=[[0,0], [0, MAX_TOTAL_SIZE - boxes.shape[1] + 1], [0, 0]])
            pred_scores_goal = tf.pad(tf.constant([[.9, .0], [.9, .0]] ),
                                        constant_values=.0,
                                        paddings=[[0, 0], [0, MAX_TOTAL_SIZE - boxes.shape[1]]])
            pred_classes_goal = tf.zeros((2, MAX_TOTAL_SIZE), dtype=tf.float32)

            self.assertAllClose(pred_boxes,  pred_boxes_goal)            
            self.assertAllClose(pred_scores,  pred_scores_goal)
            self.assertAllEqual(pred_classes,  pred_classes_goal)
            self.assertAllEqual(pred_valid_detections, tf.constant([1, 1, ], dtype=tf.int32))
    
    def test_batched_one_class_simple_reverse(self):
        
        # one class
        # two overlapping bboxes
        # second predictions has a higher score
        boxes =  tf.constant([[[[.1, .1, .9, .6]], [[.1, .9, .9, .4]]], 
                              [[[.1, .1, .9, .6]], [[.1, .9, .9, .4]]]])
        pred_boxes_goal = tf.constant([[[[.1, .9, .9, .4]], [[.1, .1, .9, .6]]],
                                       [[[.1, .9, .9, .4]], [[.1, .1, .9, .6]]]])
        scores = tf.constant([[[.8], [.9]], [[.8], [.9]]])

        with self.session():

            pred_boxes, pred_scores, pred_classes, pred_valid_detections = \
                combined_soft_nms(
                    boxes,
                    scores,
                    MAX_OUTPUT_SIZE_PER_CLASS,
                    MAX_TOTAL_SIZE,
                    iou_threshold=IOU_THRESHOLD,
                    score_threshold=SCORE_THRESHOLD,
                    soft_nms_sigma=NMS_SIGMA,
                    pad_per_class=False,
                    clip_boxes=True,
                    name=None
                )

            # test bounding box predictions
            pred_boxes_goal = tf.pad(pred_boxes_goal[:,:,0,:],
                                    constant_values=.0,
                                    paddings=[[0,0], [0, MAX_TOTAL_SIZE - boxes.shape[1]], [0, 0]])
            pred_scores_goal = tf.pad(tf.constant([[0.9, 0.70599747], [0.9, 0.70599747]]),
                                    constant_values=.0,
                                    paddings=[[0, 0], [0, MAX_TOTAL_SIZE - boxes.shape[1]]])
            pred_classes_goal = tf.zeros((2, MAX_TOTAL_SIZE), dtype=tf.float32)

            self.assertAllClose(pred_boxes,  pred_boxes_goal)            
            self.assertAllClose(pred_scores,  pred_scores_goal)
            self.assertAllEqual(pred_classes,  pred_classes_goal)
            self.assertAllEqual(pred_valid_detections, tf.constant([2, 2, ], dtype=tf.int32))

    def test_batched_multi_class_simple_reverse(self):
        
        # one class
        # two overlapping bboxes
        # second predictions has a higher score
        boxes =  tf.constant([[[[.1, .1, .9, .6]], [[.1, .9, .9, .4]], 
                               [[.1, .1, .9, .6]], [[.1, .9, .9, .4]], 
                               [[.1, .1, .9, .6]], [[.1, .9, .9, .4]],
                               [[.1, .1, .9, .6]],],
                               # next batch
                              [[[.1, .1, .9, .6]], [[.1, .9, .9, .4]],
                               [[.1, .1, .9, .6]], [[.1, .9, .9, .4]],
                               [[.1, .1, .9, .6]], [[.1, .9, .9, .4]],
                               [[.1, .1, .9, .6]],]])

        pred_boxes_goal = tf.constant([[[[.1, .9, .9, .4]],
                                        [[.1, .9, .9, .4]],
                                        [[.1, .9, .9, .4]],
                                        [[.1, .1, .9, .6]],
                                        [[.1, .1, .9, .6]],
                                        [[.1, .1, .9, .6]],],
                                    # next batch
                                       [[[.1, .9, .9, .4]],
                                        [[.1, .9, .9, .4]],
                                        [[.1, .9, .9, .4]],
                                        [[.1, .1, .9, .6]],
                                        [[.1, .1, .9, .6]],
                                        [[.1, .1, .9, .6]],]])

        scores = tf.constant([[[.8, .0, .0],  # cls 1
                               [.9, .0, .0],  # cls 1
                               [.0, .8, .0],  # cls 2
                               [.0, .9, .0],  # cls 2
                               [.0, .0, .8],  # cls 3
                               [.0, .0, .9],  # cls 3
                               [.0, .0, .29],], # cls 3
                            # next batch
                              [[.8, .0, .0],  # cls 1
                               [.9, .0, .0],  # cls 1
                               [.0, .8, .0],  # cls 2
                               [.0, .9, .0],  # cls 2
                               [.0, .0, .8],  # cls 3
                               [.0, .0, .9],  # cls 3
                               [.0, .0, .29],]])
        print(boxes.shape, scores.shape)
        with self.session():

            pred_boxes, pred_scores, pred_classes, pred_valid_detections = \
                combined_soft_nms(
                    boxes,
                    scores,
                    MAX_OUTPUT_SIZE_PER_CLASS,
                    MAX_TOTAL_SIZE,
                    iou_threshold=IOU_THRESHOLD,
                    score_threshold=SCORE_THRESHOLD,
                    soft_nms_sigma=NMS_SIGMA,
                    pad_per_class=False,
                    clip_boxes=True,
                    name=None
                )

            # test bounding box predictions
            pred_boxes_goal = tf.pad(pred_boxes_goal[:,:,0,:],
                                    constant_values=.0,
                                    paddings=[[0,0], [0, MAX_TOTAL_SIZE - boxes.shape[1] + 1], [0, 0]])
            pred_scores_goal = tf.pad(tf.constant([[.9, .9, .9, .70599747, .70599747, .70599747], 
                                                   [.9, .9, .9, .70599747, .70599747, .70599747]]),
                                    constant_values=.0,
                                    paddings=[[0, 0], [0, MAX_TOTAL_SIZE - boxes.shape[1] + 1]])
            pred_classes_goal = tf.pad(tf.constant([[.0, 1.0, 2.0, .0, 1.0, 2.0], 
                                                    [.0, 1.0, 2.0, .0, 1.0, 2.0]]),
                                    constant_values=.0,
                                    paddings=[[0, 0], [0, MAX_TOTAL_SIZE - boxes.shape[1] + 1]])

            self.assertAllClose(pred_boxes,  pred_boxes_goal,)            
            self.assertAllClose(pred_scores,  pred_scores_goal,)
            self.assertAllEqual(pred_classes,  pred_classes_goal,)
            self.assertAllEqual(pred_valid_detections, tf.constant([6, 6, ], dtype=tf.int32),)

    def test_batched_multi_class_simple_reverse_padded(self):
        
        # one class
        # two overlapping bboxes
        # second predictions has a higher score
        max_size_per_class = 150
        max_total_size = 400 # smaller than num_cls * max_size_per_class = 3 * 150 = 450
                             # so output is trimmed to max_total_size = 400
        boxes =  tf.constant([[[[.1, .1, .9, .6]], [[.1, .9, .9, .4]], 
                               [[.1, .1, .9, .6]], [[.1, .9, .9, .4]], 
                               [[.1, .1, .9, .6]], [[.1, .9, .9, .4]],
                               [[.1, .1, .9, .6]],],
                               # next batch
                              [[[.1, .1, .9, .6]], [[.1, .9, .9, .4]],
                               [[.1, .1, .9, .6]], [[.1, .9, .9, .4]],
                               [[.1, .1, .9, .6]], [[.1, .9, .9, .4]],
                               [[.1, .1, .9, .6]],]])

        pred_boxes_goal = tf.constant([[[[.1, .9, .9, .4]],
                                        [[.1, .9, .9, .4]],
                                        [[.1, .9, .9, .4]],
                                        [[.1, .1, .9, .6]],
                                        [[.1, .1, .9, .6]],
                                        [[.1, .1, .9, .6]],],
                                    # next batch
                                       [[[.1, .9, .9, .4]],
                                        [[.1, .9, .9, .4]],
                                        [[.1, .9, .9, .4]],
                                        [[.1, .1, .9, .6]],
                                        [[.1, .1, .9, .6]],
                                        [[.1, .1, .9, .6]],]])

        scores = tf.constant([[[.8, .0, .0],  # cls 1
                               [.9, .0, .0],  # cls 1
                               [.0, .8, .0],  # cls 2
                               [.0, .9, .0],  # cls 2
                               [.0, .0, .8],  # cls 3
                               [.0, .0, .9],  # cls 3
                               [.0, .0, .29],], # cls 3
                            # next batch
                              [[.8, .0, .0],  # cls 1
                               [.9, .0, .0],  # cls 1
                               [.0, .8, .0],  # cls 2
                               [.0, .9, .0],  # cls 2
                               [.0, .0, .8],  # cls 3
                               [.0, .0, .9],  # cls 3
                               [.0, .0, .29],]])
        print(boxes.shape, scores.shape)
        with self.session():

            pred_boxes, pred_scores, pred_classes, pred_valid_detections = \
                combined_soft_nms(
                    boxes,
                    scores,
                    max_size_per_class,
                    max_total_size,
                    iou_threshold=IOU_THRESHOLD,
                    score_threshold=SCORE_THRESHOLD,
                    soft_nms_sigma=NMS_SIGMA,
                    pad_per_class=True,
                    clip_boxes=True,
                    name=None
                )

            # test bounding box predictions
            pred_boxes_goal = tf.pad(pred_boxes_goal[:,:,0,:],
                                    constant_values=.0,
                                    paddings=[[0,0], [0, max_total_size - boxes.shape[1] + 1], [0, 0]])
            pred_scores_goal = tf.pad(tf.constant([[.9, .9, .9, .70599747, .70599747, .70599747], 
                                                   [.9, .9, .9, .70599747, .70599747, .70599747]]),
                                    constant_values=.0,
                                    paddings=[[0, 0], [0, max_total_size - boxes.shape[1] + 1]])
            pred_classes_goal = tf.pad(tf.constant([[.0, 1.0, 2.0, .0, 1.0, 2.0], 
                                                    [.0, 1.0, 2.0, .0, 1.0, 2.0]]),
                                    constant_values=.0,
                                    paddings=[[0, 0], [0, max_total_size - boxes.shape[1] + 1]])

            self.assertAllClose(pred_boxes,  pred_boxes_goal,)            
            self.assertAllClose(pred_scores,  pred_scores_goal,)
            self.assertAllEqual(pred_classes,  pred_classes_goal,)
            self.assertAllEqual(pred_valid_detections, tf.constant([6, 6, ], dtype=tf.int32),)


    def test_empty(self):
        
        # no detections
        batch_size = 2
        num_boxes = 250
        boxes =  tf.zeros((batch_size, num_boxes, 1, 4), dtype=tf.float32)
        scores = tf.zeros((batch_size, num_boxes, 1), dtype=tf.float32)

        with self.session():

            pred_boxes, pred_scores, pred_classes, pred_valid_detections = \
                combined_soft_nms(
                    boxes,
                    scores,
                    MAX_OUTPUT_SIZE_PER_CLASS,
                    MAX_TOTAL_SIZE,
                    iou_threshold=IOU_THRESHOLD,
                    score_threshold=SCORE_THRESHOLD,
                    soft_nms_sigma=NMS_SIGMA,
                    pad_per_class=False,
                    clip_boxes=True,
                    name="test_empty"
                )

            # test bounding box predictions
            pred_boxes_goal = tf.zeros((batch_size, MAX_TOTAL_SIZE, 4), dtype=tf.float32)
            pred_scores_goal = tf.zeros((batch_size, MAX_TOTAL_SIZE), dtype=tf.float32)
            pred_classes_goal = tf.zeros((batch_size, MAX_TOTAL_SIZE), dtype=tf.float32)

            self.assertAllClose(pred_boxes,  pred_boxes_goal)            
            self.assertAllClose(pred_scores,  pred_scores_goal)
            self.assertAllEqual(pred_classes,  pred_classes_goal)
            self.assertAllEqual(pred_valid_detections, tf.constant([0, 0, ], dtype=tf.int32))


if __name__ == '__main__':
    test.main()