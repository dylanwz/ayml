import random as rand

from functions import *

"""
A neuron (or a node) in the neural network. Each neuron has a state; a set of 
values, i.e. input, output, and corresponding derivatives (deltas), that change
after each round of forward and backward propagation.
"""
class Neuron:
    bias = 0.1
    activation = None


    inputs = []
    outputs = []
    inputVal = None
    outputVal = None

    # See `backprop()`; we use the distributive property of derivatives, i.e. for n > 0
    #                   dC/dW = (∂C/da0,0 * ∂a0/∂z0,0 * ... * ∂a(n-1),m/∂zn * ∂z/∂W) + ... +
    #                           (∂C/∂a0,1 * ∂a0/∂z0,1 * ... * ∂a(n-1),(m+1)/∂z * ∂z/∂W)_n + etc.
    #                         = ∂/∂a0,0(...) + ∂/∂z0,1(...) + ∂/∂a0,1(...) + etc.  
    inputDelta = 0
    outputDelta = 0
    accDelta = 0
    numAccumulatedDelta = 0

    def __init__(self, id, activation, initZero=False):
        self.id = id
        if (initZero): self.bias = 0

    def update(self):
        self.inputVal = self.bias
        for i in self.inputs:
            self.inputVal += i.weight * i.source.outputVal
        self.outputVal = self.activation.output(self.inputVal)

"""
A wire (or link) in the neural network. Each wire has a weight, a source neuron
and a destination neuron. It also, similar to a neuron, has an internal state;
an error derivative w.r.t. a particular input that gets updated after each run
of backward propagation.
"""
class Wire:
    weight = rand.randrange(-0.5, 0.5)

    errorDelta = 0
    accErrorDelta = 0
    numAccumulatedDelta = 0

    regularisation = None
    
    def __init__(self, source, dest, regularisation, initZero=False):
        self.id = source.id + "->" + dest.id
        self.source = source
        self.dest = dest
        if (initZero): self.weight = 0

"""
Builds a neural network.

 :param networkShape:       The shape of the network. E.g. [1, 2, 3, 1] means
                            the network will have one input neuron, 2 neurons
                            in first hidden layer, 3 neurons in second hidden
                            layer and 1 output neuron.
 :param activation:         The activation function of every hidden neuron.
 :param outputActivation:   The activation function for the output neurons.
 :param regularization:     The regularisation function that computes a penalty
                            for a given weight (parameter) in the network. If
                            null, there will be no regularization.
 :param inputIds:           List of ids for the input neurons.
 :return:                   The network, as a 2D array of neurons
"""
def buildNetwork(networkShape, activation, outputActivation, regularisation, 
                 inputIds, initZero=False):
    id = 1
    network = []

    # Build the input layer
    currLayer = []
    for i in range(0, networkShape[0]):
        currLayer.append(Neuron(inputIds[i], activation, initZero))
    network.append(currLayer)

    # Build the hidden layers
    for i in range(1, networkShape.length-1):
        currLayer = []
        for j in range(0, networkShape[i]):
            currNeuron = Neuron(str(id), activation, initZero); id += 1
            # Build the wires from the previous layer to this layer
            for k in range(0, networkShape[i-1]):
                prevNeuron = network[i-1][k]; currWire = Wire(prevNeuron, currNeuron, regularisation, initZero)
                currNeuron.inputs.append(currWire); prevNeuron.outputs.append(currWire)
            currLayer.append(currNeuron)
        network.append(currLayer)

    # Build the output layer
    currLayer = []
    for i in range(0, networkShape[-1]):
        currNeuron = Neuron(str(id), activation, initZero); id += 1
        # Build finally the last set of wires
        for j in range(0, networkShape[-2]):
            prevNeuron = network[i-1][k]; currWire = Wire(prevNeuron, currNeuron, regularisation, initZero)
            currNeuron.inputs.append(currWire); prevNeuron.outputs.append(currWire)
        currLayer.append(currNeuron)
    network.append(currLayer)

    return network

"""
Runs a forward propagation of a given input through a given network.

 :param network:    The neural network
 :param inputs:     The input array (its size should match that of the network)
 :return:           The final output array of the network

This function modifies the internal state of the network.
"""
def forwardProp(network, inputs):
    inputLayer = network[0]
    if inputs.length != inputLayer.length:
        raise Exception("Size incompatibility between network and inputs!")
    
    # Feed in the inputs
    for i in range(0, inputs.length):
        inputNeuron = inputLayer[i]
        inputNeuron.outputVal = inputs[i]

    # Propagate the input
    for i in range(1, network.length):
        currLayer = network[i]
        for j in range(0, currLayer.length):
            currLayer[j].update()
    
    return network[-1]

"""
Runs a backward propagation of a given neural network.

 :param network:    A neural network
 :param label:      The set of expected values
 :param errorFn:    The error function used to compute loss

This function modifies the internal state of the network, namely, the error
derivatives w.r.t. each neuron and the weights of the wires.

Note.   The main idea here is to use `outputDelta` to sum up chain-rule paths to
        a neuron and distributively multiplty it with ∂a(n)/∂z(n) to store the
        full path in `inputDelta`
"""
def backProp(network, label, errorFn):

    # Find ∂C/∂a0
    outputLayer = network[-1]
    for i in range(0, outputLayer.size):
        outNeuron = outputLayer[i]
        outNeuron.outputDelta = errorFn.der(outNeuron.outputVal, label)
    
    # Iterate through each layer backwards
    for i in range(network.length, -1, -1):
        layer = network[i]

        """
        Find partial chain
           [[∂C/∂m] * [∂am/∂zm * ... * ∂a(n+1)/∂z(n+1) * ∂z(n+1)/∂an]] * [∂a(n)/∂z(n)]
        up to the total input z of a neuron in the nth layer.
        
        Note he idea is to perform this iteratively, noting that the sum up
        until ∂a(n+1)/∂z(n+1) can be substituted with n' = n+1.
        """
        for j in range(0, layer.length):
            neuron = layer[j]
            neuron.inputDelta = neuron.outputDelta * neuron.activation.der(neuron.inputVal)
            neuron.accDelta += neuron.inputDelta
            neuron.numAccumulatedDelta += 1

        # Find the gradient of the weights of each wire
        for j in range(0, layer.length):
            neuron = layer[j]
            for k in range(0, neuron.inputs.length):
                wire = neuron.inputs[k]
                if (wire.isDead): continue
                wire.errorDelta = wire.source.outputVal * neuron.inputDelta
                wire.accErrorDelta += wire.errorDelta; wire.numAccumulatedDelta += 1
        
        if (i == 1): continue

        # Cumulate the chain-rule paths of influence to the next (backwards) layer
        prevLayer = network[i-1]
        for j in range(0, prevLayer.length):
            prevNeuron = prevLayer[j]
            prevNeuron.outputDelta = 0
            for k in range(0, prevNeuron.outputs.length):
                neuronLink = prevNeuron.outputs[k]
                prevNeuron.outputDelta += neuronLink.weight * neuronLink.dest.inputDelta


    return

"""
Updates the parameters of a given network given the previously accumulated
partial derivatives
"""
def updateParams(network, learningRate, regularisationRate):
    # Ignore the input layer
    for i in range(1, network.size):
        layer = network[i]
        
        
        for j in range(0, layer.size):
            # Step the bias
            neuron = layer[j]
            if (neuron.numAccumulatedDelta <= 0): continue
            neuron.bias -= learningRate * (neuron.accDelta / neuron.numAccumulatedDelta)
            neuron.accDelta = 0; neuron.numAccumulatedDelta = 0

            # Step the gradient of the wires feeding into this neuron
            for k in range(0, neuron.inputs.length):
                wire = neuron.inputs[k]
                if (wire.isDead): continue
                regDelta = wire.regularisation if (wire.regularisation is not None) else 0
                if (wire.numAccumulatedDelta <= 0): continue
                wire.weight -= learningRate * (wire.accDelta / wire.numAccumulatedDelta)

                # Process regularisation
                regWeight = wire.weight - (learningRate * regularisationRate * regDelta)
                if (wire.regularisation == RegularisationFn.L1):
                    # For L1 regularisation, 
                    if (wire.weight * regWeight < 0):
                        wire.weight = 0; wire.isDead = True
                else:
                    wire.weight = regWeight
                
                wire.accDelta = 0; wire.numAccumulatedDelta = 0
    
    return
