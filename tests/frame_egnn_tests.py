import unittest
import torch
import math
from frame_egnn.frame_egnn import quat_exponential

class FrameEGNNTest(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()
    
    def test_quat_exponential(self):
        imag_quat = torch.Tensor([1.0, 0.0, 0.0]).unsqueeze(0)
        exp = quat_exponential(imag_quat)
        self.assertTrue(torch.equal(exp, torch.Tensor([math.sin(1), 0.0, 0.0, math.cos(1)]).unsqueeze(0)))


if __name__ == '__main__':
    unittest.main()