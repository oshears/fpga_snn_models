# import numpy
import numpy as np
import matplotlib.pyplot as plt

# import modules from pytorch
import torch
from torchvision import transforms

# import modules from bindsnet
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder, BernoulliEncoder, RankOrderEncoder
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
from bindsnet.network.monitors import Monitor
from bindsnet.network import load, Network
from bindsnet.network.nodes import Input, IFNodes
from bindsnet.network.topology import Connection, LocalConnection
from bindsnet.analysis.plotting import plot_spikes, plot_voltages, plot_input, plot_weights

# miscellaneous imports
import os
import argparse


SCALE_FACTOR = 2**31
TIMESTEPS = 100
NUM_NEURONS = 100
NUM_INPUTS = 784

network = Network(learning=False)


input_layer = Input(n=NUM_INPUTS)
if_layer = IFNodes(n=NUM_NEURONS,reset = 5 * SCALE_FACTOR,thresh = 12 * SCALE_FACTOR,refrac=0,tc_trace=20.0)

w = torch.zeros((NUM_INPUTS, NUM_NEURONS))

for neuron_idx in range(NUM_NEURONS):
    fh = open(f"../networks/if_Poisson_1_32bit_weights/{neuron_idx}.txt")
    for synapse_idx in range(NUM_INPUTS):
        line = fh.readline()
        val = int(line,16)
        w[synapse_idx,neuron_idx] = val

        #if (val != 0):
        #    print(f"Neuron: {neuron_idx} Synapse: {synapse_idx} Weight: {val}")
    # print(w[:,neuron_idx])

input_connection = Connection(
            source=input_layer,
            target=if_layer,
            w=w,
            wmin=0,
            wmax=1 * SCALE_FACTOR,
            norm=78.4 * SCALE_FACTOR,
        )

# create a monitor to record the spiking activity of the output layer (Y)
output_spikes_monitor = Monitor(if_layer, state_vars=["s","v"], time=TIMESTEPS)



network.add_layer(input_layer,name="X")
network.add_layer(if_layer,name="Y")
network.add_connection(input_connection,source="X",target="Y")
network.add_monitor(output_spikes_monitor, name="Y")


spike_inputs = torch.zeros((TIMESTEPS,NUM_INPUTS))
fh = open("../input_data/tensor([0]).txt")
for time_idx in range(TIMESTEPS):
    line = fh.readline()
    spikes = line.split(" ")

    #print(line)
    for spike_idx in range(NUM_INPUTS):
        spike_val = int(spikes[spike_idx])

        spike_inputs[time_idx,spike_idx] = spike_val

    # print(spike_inputs[time_idx])



spike_inputs_dict = {"X" : spike_inputs}
network.run(inputs=spike_inputs_dict, time=TIMESTEPS, input_time_dim=0)



output_spikes = output_spikes_monitor.get("s").reshape((TIMESTEPS,NUM_NEURONS))
assignments = torch.load('../networks/if_Poisson_1_32bit_snn_assignments.pt',map_location=torch.device('cpu'))




# potentials = output_spikes_monitor.get("v").reshape((TIMESTEPS,NUM_NEURONS))
# # print(f"Weights: {w[:,0]}")
# for time_idx in range(TIMESTEPS):  # range(TIMESTEPS):
#     # for spike_idx in range(NUM_INPUTS):
#     #     print(f"Time: {time_idx} Spike: {spike_idx} Value: {spike_inputs[time_idx,spike_idx]}")
#     print(f"Time: {time_idx} Neuron: 0 Spikes: {output_spikes[time_idx,0]}")
#     print(f"Time: {time_idx} Neuron: 0 Potential: {potentials[time_idx,0]}")

# count = output_spikes[:,0].sum()
# print(f"Neuron[0] : {count}")
for neuron_idx in range(NUM_NEURONS):
    count = output_spikes[:,neuron_idx].sum()
    print(f"Neuron[{neuron_idx}] : {count}")

plot_voltages({"IF Layer" : output_spikes_monitor.get("v")})
plot_spikes({"IF Layer" : output_spikes_monitor.get("s")})
plt.show()
# input()