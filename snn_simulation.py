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

# miscellaneous imports
import os
import argparse

# import local modules
from models.diehl_cook_snn import DiehlAndCookNetwork,DiehlAndCookNetworkInt



# create an argument parser to interpret command line arguments
parser = argparse.ArgumentParser()

# --encoding specifies the type of encoding (Poisson, Bernoulli or RankOrder)
parser.add_argument("--encoding", type=str, default="Poisson")

# parse the arguments
args = parser.parse_args()

# declare global variables

# n_neurons specifies the number of neurons per layer
n_neurons = 100

# batch_size specifies the number of training samples to collect weight changes from before updating the weights
batch_size = 64

# n_train specifies the number of training samples
n_train = 60000

# n_test specifies the number of testing samples
n_test = 10000

# update_steps specifies the number of batches to process before reporting an update
update_steps = 10

# time specifies the simulation time of the SNN
time = 100

# dt specifies the timestep size for the simulation time
dt = 1

# intensity specifies the maximum intensity of the input data
intensity = 128

# gpu setting
gpu = True

# update_interavl specifies the number of samples processed before updating accuracy estimations
update_interval = update_steps * batch_size

# setup CUDA
device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")

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


# build network based on the input argument
network = DiehlAndCookNetworkInt(n_inpt=784,inpt_shape=(1, 28, 28),batch_size=batch_size,n_neurons=100)

# run the network using the GPU/CUDA
if gpu:
    network.to("cuda")

# load the MNIST training dataset
# use the encoder to convert the input into spikes
dataset = MNIST(
    encoder,
    None,
    root=os.path.join(".", "data", "MNIST"),
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# create a dataloader to iterate over and batch the training data
train_dataloader = DataLoader( dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True, )


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
test_dataloader = DataLoader( test_dataset, batch_size=256, shuffle=False, num_workers=n_workers, pin_memory=True, )

# declare variables needed for estimating the network accuracy
n_classes = 10

# assignments stores the label that each output neuron corresponds to
assignments = -torch.ones(n_neurons, device=device)

# proportions stores the ratio of the number of times each of the output neurons produced a spike for the corresponding class relative to other classes
proportions = torch.zeros((n_neurons, n_classes), device=device)

# rates stores the number of times each of the output neurons produced a spike for the corresponding class
rates = torch.zeros((n_neurons, n_classes), device=device)

# create a dictionary to store all assignment and proportional assignment accuracy values
accuracy = {"all": [], "proportion": []}

# create a monitor to record the spiking activity of the output layer (Y)
output_spikes_monitor = Monitor(network.layers["Y"], state_vars=["s"], time=int(time / dt))

# add the monitor to the network
network.add_monitor(output_spikes_monitor, name="Y")

# create a tensor to store the spiking activity for all neurons for the duration of the update_interval 
spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)

# train the network
print("\nBegin training.\n")

# create a list to store the sample labels for each batch in the update interval
labels = []

# iterate through each batch of data
for step, batch in enumerate(train_dataloader):

    # get next input sample
    inputs = {"X": batch["encoded_image"]}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # if it is time to print out an accuracy estimate
    if step % update_steps == 0 and step > 0:

        # convert the array of labels into a tensor
        label_tensor = torch.tensor(labels, device=device)

        # get network predictions based on the spiking activity, previous assignments and number of classes
        all_activity_pred = all_activity( spikes=spike_record, assignments=assignments, n_labels=n_classes )

        # get network predictions based on the spiking activity, previous assignments, proportional assignments and number of classes
        proportion_pred = proportion_weighting( spikes=spike_record, assignments=assignments, proportions=proportions, n_labels=n_classes, )

        # compute the network accuracy based on the prediction results and append to the assignment accuracy dictionary
        accuracy["all"].append( 100 * torch.sum(label_tensor.long() == all_activity_pred).item() / len(label_tensor) )
        
        # compute the network accuracy based on the proportional prediction results and append to the assignment accuracy dictionary
        accuracy["proportion"].append( 100 * torch.sum(label_tensor.long() == proportion_pred).item() / len(label_tensor) )

        # report the network accuracy at the current time
        print( "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)" % ( accuracy["all"][-1], np.mean(accuracy["all"]), np.max(accuracy["all"]), ) )
        print("Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f" " (best)" % ( accuracy["proportion"][-1], np.mean(accuracy["proportion"]), np.max(accuracy["proportion"]), ) )

        # display how many samples are remaining
        print("Progress:",step*batch_size,"/",n_train)

        # update the neuron assignments, proportional assignments and spiking rates
        assignments, proportions, rates = assign_labels( spikes=spike_record, labels=label_tensor, n_labels=n_classes, rates=rates,)

        # reset the list of labels
        labels = []

    # append the labels of the current batch to the list of labels
    labels.extend(batch["label"].tolist())

    # run the network on the input
    network.run(inputs=inputs, time=time, input_time_dim=1)

    # get the spikes produced by the current batch
    s = output_spikes_monitor.get("s").permute((1, 0, 2))

    # store the spikes inside of the spike record list at the current batch's index (relative to the number of batches in the update interval)
    spike_record[(step * batch_size) % update_interval : (step * batch_size % update_interval) + s.size(0)] = s

    # reset the network before running it again
    network.reset_state_variables()  

print("Training complete.\n")

# save the network
filename = f"./networks/diehlAndCook_{args.encoding}_{batch_size}_snn.pt"
network.save(filename)

# write out network assignments and proportions
networkAssignmentFile = open(f"./networks/diehlAndCook_{args.encoding}_{batch_size}_snn_assignments.json","w")
networkAssignmentFile.write(assignments)
networkAssignmentFile.write(proportions)
networkAssignmentFile.close()

# create a dictionary to store all assignment and proportional assignment accuracy values for the test data
accuracy = {"all": 0, "proportion": 0}

# run the network for each test sample
print("\nBegin testing\n")

# put the network into test mode
network.train(mode=False)

# iterate over each batch
for step, batch in enumerate(test_dataloader):

    # get next input sample and send to the GPU
    inputs = {"X": batch["encoded_image"].cuda()}

    # run the network on the input
    network.run(inputs=inputs, time=time, input_time_dim=1)

    # get the spikes produced by the current batch
    spike_record = output_spikes_monitor.get("s").permute((1, 0, 2))

    # convert the array of labels into a tensor
    label_tensor = torch.tensor(batch["label"], device=device)

    # get network predictions based on the spiking activity, previous assignments and number of classes
    all_activity_pred = all_activity( spikes=spike_record, assignments=assignments, n_labels=n_classes )

    # get network predictions based on the spiking activity, previous assignments, proportional assignments and number of classes
    proportion_pred = proportion_weighting( spikes=spike_record, assignments=assignments, proportions=proportions, n_labels=n_classes, )

    # compute the network accuracy based on the prediction results and add the results to the accuracy dictionary
    accuracy["all"] += float(torch.sum(label_tensor.long() == all_activity_pred).item())

    # compute the network accuracy based on the proportional prediction results and add the results to the accuracy dictionary
    accuracy["proportion"] += float( torch.sum(label_tensor.long() == proportion_pred).item() )

    # if it is time to print out an accuracy estimate
    if step % update_steps == 0 and step > 0:
        # print out the assignment and proportional assignment accuracy
        print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test))
        print("Proportion weighting accuracy: %.2f" % (accuracy["proportion"] / n_test))

        #print out how many test samples are remaining
        print("Progress:",step*256,"/",n_test)

    # reset the network before running it again
    network.reset_state_variables()

# print out the final assignment and proportional assignment accuracies
print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test))
print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test))
print("Testing complete.\n")
