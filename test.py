from cycler import V
import torch

from .models import Detector
from .utils import load_detection_data, DetectionSuperTuxDataset, PR, point_close
from . import dense_transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from os import path
train_dataset = DetectionSuperTuxDataset('dense_data/train', min_size=0)
image, *dets = train_dataset[0]
dets = Detector().detect(image)
print(dets.shape)