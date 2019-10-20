""" Implementation of Topological Quantum Processor with Spiking Neural Networks deployed on the Loihi chip. """

import glob
import nengo
import numpy as np
import nengo_dl
import nengo_loihi
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


""" input data """
# TODO: write the loading for input data and define an output
num_invocation = 10
N_INPUT = 16
N_LAYER_1 = 64
N_LAYER_2 = 32
N_LAYER_3 = 32
N_OUTPUT = 5

input_1 = []
for i in range(num_invocation):
    for j in range(N_INPUT):
        input_1[i][j] = 0

# give path to the dataset folder
data_path = './dataset'

print("loading the dataset")
# Get folder path containing h5 files
file_list = glob.glob(data_path + '/*.h5')
dataset = []
for file_path in file_list:
    dataset.append(file_path)
print("dataset loaded")

# creating a train and test dataset
test_d = []
train_d = []


""" spiking model for TPQ inspired by implementation on FPGA """
max_rate = 100
amplitude = 1/max_rate
presentation_time = 0.1

# model for Jet classification
with nengo.Network(label="Topological Quantum Processor") as model:
    nengo_loihi.add_params(model)
    model.config[nengo.Connection].synapse = None

    model.config[nengo.Ensemble].max_rates = nengo.dists.Choice([max_rate])
    model.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])

    # Choose a type of spiking neuron
    neuron_type = nengo.SpikingRectifiedLinear(amplitude=amplitude)
    # neuron_type = nengo.LIF(tau_rc=0.02, tau_ref=0.001, amplitude=amplitude)
    # neuron_type = nengo.AdaptiveLIF(amplitude=amplitude)
    # neuron_type = nengo.Izhikevich()

    inp = nengo.Node(nengo.processes.PresentInput(test_d[0], presentation_time), size_out=N_INPUT, label="Input")

    layer_1 = nengo.Ensemble(n_neurons=N_LAYER_1, dimensions=1, neuron_type=neuron_type, label="Layer 1")
    model.config[layer_1].on_chip = False
    nengo.Connection(inp, layer_1.neurons, transform=nengo_dl.dists.Glorot())
    p1 = nengo.Probe(layer_1.neurons)

    layer_2 = nengo.Ensemble(n_neurons=N_LAYER_2, dimensions=1, neuron_type=neuron_type, label="Layer 2")
    nengo.Connection(layer_1.neurons, layer_2.neurons, transform=nengo_dl.dists.Glorot())
    p2 = nengo.Probe(layer_2.neurons)

    layer_3 = nengo.Ensemble(n_neurons=N_LAYER_3, dimensions=1, neuron_type=neuron_type, label="Layer 3")
    nengo.Connection(layer_2.neurons, layer_3.neurons, transform=nengo_dl.dists.Glorot())
    p3 = nengo.Probe(layer_3.neurons)

    out = nengo.Node(size_in=N_OUTPUT, label="Output")
    nengo.Connection(layer_3.neurons, out, transform=nengo_dl.dists.Glorot())

    out_p = nengo.Probe(out)
    out_p_filt = nengo.Probe(out, synapse=nengo.Alpha(0.01))


def crossentropy(outputs, targets):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=targets))


def classification_error(outputs, targets):
    return 100 * tf.reduce_mean(
        tf.cast(tf.not_equal(tf.argmax(outputs[:, -1], axis=-1), tf.argmax(targets[:, -1], axis=-1)), tf.float32))


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False,  title=None, cmap=plt.cm.Blues):
    # This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes,
           title=title, ylabel='True label', xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


dt = 0.001  # simulation timestep
step = int(presentation_time / dt)
presentation_time = 0.1  # input presentation time
train_data = {inp: train_d[0][:, None, :], out_p: train_d[1][:, None, :]}

# for the test data evaluation we will be running the network over time
# using spiking neurons, so we need to repeat the input/target data
# for a number of timesteps (based on the presentation_time)
minibatch_size = 200
test_data = {inp: np.tile(test_d[0][:minibatch_size*2, None, :], (1, step, 1)),
             out_p_filt: np.tile(test_d[1][:minibatch_size*2, None, :], (1, step, 1))}

""" training and simulation of the model """
do_training = True
with nengo_dl.Simulator(model, minibatch_size=minibatch_size, seed=0) as sim:
    if do_training:
        output = {out_p_filt: classification_error}
        loss = sim.loss(test_data, output)
        print("error before training: %.2f%%" % loss)
        # run training
        sim.train(train_data, tf.train.RMSPropOptimizer(learning_rate=0.001),
                  objective={out_p: crossentropy}, n_epochs=50)
        print("error after training: %.2f%%" % sim.loss(test_data, {out_p_filt: classification_error}))
        sim.save_params("./TQP_params")
    else:
        print("error before training: %.2f%%" % sim.loss(test_data, {out_p_filt: classification_error}))
        sim.load_params("./model_files/TPQ_file.ckpt")
        print("parameters loaded")
        print("error after training: %.2f%%" % sim.loss(test_data, {out_p_filt: classification_error}))

    sim.run_steps(int(presentation_time / dt), data={inp: test_data[inp][:minibatch_size]})
    sim.freeze_params(model)

for conn in model.all_connections:
    conn.synapse = 0.005

if do_training:
    with nengo_dl.Simulator(model, minibatch_size=minibatch_size) as sim:
        print("error w/ synapse: %.2f%%" % sim.loss(test_data, {out_p_filt: classification_error}))


""" deployment of the model on neuromorphic hardware """
n_presentations = 50
with nengo_loihi.Simulator(model, dt=dt, precompute=False) as sim:
    # if running on Loihi, increase the max input spikes per step
    if 'loihi' in sim.sims:
        sim.sims['loihi'].snip_max_spikes_per_step = 120

    # run the simulation on Loihi
    sim.run(n_presentations * presentation_time)

    # check classification error
    step = int(presentation_time / dt)
    output = sim.data[out_p_filt][step - 1::step]

    error_percentage = 100 * (np.mean(np.argmax(output, axis=-1) !=
                             np.argmax(test_data[out_p_filt][:n_presentations, -1], axis=-1)))

    predicted = np.argmax(output, axis=-1)
    correct = np.argmax(test_data[out_p_filt][:n_presentations, -1], axis=-1)

    predicted = np.array(predicted, dtype=int)
    correct = np.array(correct, dtype=int)

    print("Predicted labels: ", predicted)
    print("Correct labels: ", correct)
    print("loihi error: %.2f%%" % error_percentage)

    np.set_printoptions(precision=2)

    # TODO: define an output and visualize a confusion matrix of accuracy
    # Plot non-normalized confusion matrix
    ### plot_confusion_matrix(correct, predicted, classes=class_names, title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    ### plot_confusion_matrix(correct, predicted, classes=class_names, normalize=True, title='Normalized confusion matrix')
    ### plt.show()
