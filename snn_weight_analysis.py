from bindsnet.network import load
import argparse

# create an argument parser to interpret command line arguments
parser = argparse.ArgumentParser()

parser.add_argument("--weight_size", type=int, default=16)

# parse the arguments
args = parser.parse_args()

# hexSize = args.weight_size / 4
hexSize = int(32 / 4)

networkFile = "./fpga_snn_models/networks/diehlAndCook_Poisson_64_32bit_snn.pt"
weightFileDirectory = "./fpga_snn_models/networks/diehlAndCook_Poisson_64_32bit_weights"

network = load("./fpga_snn_models/networks/diehlAndCook_Poisson_64_32bit_snn.pt")

# extract connections
excitatoryConnectionWeights = network.connections["X","Y"].w
# inhibitoryConnectionWeights = network.connections["Y","Y"].w

# print(f"Excitatory Connections: { excitatoryConnection.w } ")
# print(f"Inhibitory Connections: { inhibitoryConnection.w } ")

# test the adjusted weights

# for each hidden layer neuron
for neuronIdx in range(excitatoryConnectionWeights.shape[1]):
    # new file
    neuronFile = open(f"{weightFileDirectory}/{neuronIdx}.txt","w")
    #neuronFileNum = open(f"{weightFileDirectory}/{neuronIdx}_numerical.txt","w")

    # for each input neuron
    for inputIdx in range(excitatoryConnectionWeights.shape[0]):
        # write the weight value to the file
        weightValue = int(excitatoryConnectionWeights[inputIdx][neuronIdx].numpy())
        hexWeightValue = hex(weightValue)[2:].zfill(hexSize).upper()
        neuronFile.write(f"{hexWeightValue}\n")
        #neuronFileNum.write(f"{excitatoryConnectionWeights[inputIdx][neuronIdx].numpy()}\n")

    neuronFile.close()
    #neuronFileNum.close()