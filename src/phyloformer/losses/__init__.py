from .mae import MAE, MAELoss
from .mre import MRE, MRELoss
from .quartet_mre import QuartetMRELoss, QuartetMarginMRELoss
from .quartet_push_close import QuartetPushCloseMRELoss
from .quartet_siamese import QuartetSiameseLoss
from .quartet_siamese_mre import QuartetSiameseMRELoss
from .quartet_softmax import QuartetSoftmaxLoss

__all__ = [
    "MAE",
    "MAELoss",
    "MRE",
    "MRELoss",
    "QuartetMRELoss",
    "QuartetMarginMRELoss",
    "QuartetPushCloseMRELoss",
    "QuartetSiameseLoss",
    "QuartetSiameseMRELoss",
    "QuartetSoftmaxLoss",
]
