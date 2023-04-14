from tokenize import Single

from models.BSDual_CNN import Dual_CNN
from models.USingleCNN import Single_CNN

__models__ = {
    "Unidirectional-SingleBranch-CNN": Single_CNN,
    "Bidirectional-DualBranch-CNN": Dual_CNN
}
