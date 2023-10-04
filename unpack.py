import torch
import torchvision
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
import torchvision.transforms as transforms

data = fetch_openml(name = 'cifar_10_small')

x = pd.DataFrame(np.array(data.data))
y = pd.DataFrame(data.target.astype("uint8"))

x.to_csv("inputs.csv")
y.to_csv("labels.csv")