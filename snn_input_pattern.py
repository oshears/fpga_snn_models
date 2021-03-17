# import numpy
import numpy as np

# import modules from pytorch
import torch
from torchvision import transforms

# import modules from bindsnet
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder, BernoulliEncoder, RankOrderEncoder
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
from bindsnet.network.monitors import Monitor
from bindsnet.network import load

# miscellaneous imports
import os
import argparse

# create an argument parser to interpret command line arguments
parser = argparse.ArgumentParser()

# --encoding specifies the type of encoding (Poisson, Bernoulli or RankOrder)
parser.add_argument("--encoding", type=str, default="Poisson")

# parse the arguments
args = parser.parse_args()

# time specifies the simulation time of the SNN
time = 100

# dt specifies the timestep size for the simulation time
dt = 1

# intensity specifies the maximum intensity of the input data
intensity = 128

# determine number of worker threads to load data
n_workers = 0
# if n_workers == -1:
#     n_workers = gpu * 4 * torch.cuda.device_count()

# report the selected encoding scheme, neural model and learning technique
print("Encoding Scheme:",args.encoding)

# assign a value to the encoder based on the input argument
encoder = None
if args.encoding == "Poisson":
    encoder = PoissonEncoder(time=time,dt=dt)
if args.encoding == "Bernoulli":
    encoder = BernoulliEncoder(time=time,dt=dt)
if args.encoding == "RankOrder":
    encoder = RankOrderEncoder(time=time,dt=dt)

# load the MNIST test dataset
# use the encoder to convert the input into spikes
test_dataset = MNIST(
    encoder,
    None,
    root=os.path.join(".", "data", "MNIST"),
    download=True,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# create a dataloader to iterate over and batch the test data
test_dataloader = DataLoader( test_dataset, batch_size=1, shuffle=False, num_workers=n_workers, pin_memory=True, )

collected_labels = []

samples = 10

# iterate over each batch
for step, batch in enumerate(test_dataloader):

    # get next input sample
    inputs = {"X": batch["encoded_image"]}
    label = batch["label"]
    
    if label not in collected_labels:
        print( f"Label: {label}" )
        print(batch["encoded_image"].shape)
        collected_labels.append(label)
        convertedTensor = batch["encoded_image"].reshape((100,784)).long()
        print(convertedTensor.shape)
        fmt = "%-1.1d"
        np.savetxt(f"./input_data/{label}.txt", convertedTensor, fmt)
    
    if len(collected_labels) == samples:
        break