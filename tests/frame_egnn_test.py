import unittest
import torch
import math
from frame_egnn.frame_egnn import quat_exponential

class FrameEGNNTest(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()
    
    def test_quat_exponential(self):
        imag_quat = torch.Tensor([[1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0],
                                  [0.0, 0.0, 1.0]])
        exp = quat_exponential(imag_quat)
        result = torch.Tensor([[math.sin(1.0), 0.0,           0.0,           math.cos(1)],
                               [0.0,           math.sin(1.0), 0.0,           math.cos(1)],
                               [0.0,           0.0,           math.sin(1.0), math.cos(1)]])
        self.assertTrue(torch.equal(exp, result))


if __name__ == '__main__':
    unittest.main()