# from . import combined_soft_nms

import numpy as np

from tensorflow.python.platform import test
try:
    # if package is installed
    from tensorflow_combined_soft_nms.python.ops.combined_soft_nms_ops import combined_soft_nms
except ImportError:
    from combined_soft_nms_ops import combined_soft_nms

NMS_SIGMA=.5
MAX_OUTPUT_SIZE_PER_CLASS = 100
MAX_TOTAL_SIZE = 100


# Bounding boxes are supplied as [y1, x1, y2, x2], where (y1, x1) and 
# (y2, x2) are the coordinates of any diagonal pair of box corners and 
# the coordinates can be provided as normalized (i.e., lying in the 
# interval [0, 1]) or absolute.

class CombinedSoftNMSTest(test.TestCase):

    def testZeroOut(self):

        boxes =  [[[[.1, .1, .9, .6]], [[.1, .9, .9, .4]]]]
        scores = [[[.9]], [ [.8]]]

        with self.test_session():
            self.assertAllClose(
                 np.array([[1, 0], [0, 0]]),  np.array([[1, 0], [0, 0]])
            )
            

if __name__ == '__main__':
    test.main()