from bindsnet.network import load

networkFile = "./fpga_snn_models/networks/diehlAndCook_Poisson_64_snn.pt"
weightFileDirectory = "./fpga_snn_models/networks/diehlAndCook_Poisson_64_weights"

network = load("./fpga_snn_models/networks/diehlAndCook_Poisson_64_snn.pt")

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

    # for each input neuron
    for inputIdx in range(excitatoryConnectionWeights.shape[0]):
        # write the weight value to the file
        weightValue = int(excitatoryConnectionWeights[inputIdx][neuronIdx].numpy())
        hexWeightValue = hex(weightValue)[2:].zfill(4).upper()
        neuronFile.write(f"{hexWeightValue}\n")

    neuronFile.close()