import random as rand

from utils.functions import *

"""
A neuron (or a node) in the neural network. Each neuron has a state; a set of 
values, i.e. input, output, and corresponding derivatives (deltas), that change
after each round of forward and backward propagation.
"""
class Neuron:
    def __init__(self, id, activation, initZero=False):
        self.id = id
        self.activation = activation
        self.bias = 0 if (initZero) else 0.1

        self.inputs = []
        self.outputs = []
        self.inputVal = None
        self.outputVal = None

        # See `backprop()`; we use the distributive property of derivatives
        self.inputDelta = 0
        self.outputDelta = 0
        self.accDelta = 0
        self.numAccumulatedDelta = 0

    def update(self):
        self.inputVal = self.bias
        for i in self.inputs:
            self.inputVal += i.weight * i.source.outputVal
        self.outputVal = self.activation.out(self.inputVal)

"""
A wire (or link) in the neural network. Each wire has a weight, a source neuron
and a destination neuron. It also, similar to a neuron, has an internal state;
an error derivative w.r.t. a particular input that gets updated after each run
of backward propagation.
"""
class Wire:    
    def __init__(self, source, dest, regularisation, initZero=False):
        self.id = source.id + "->" + dest.id
        self.source = source
        self.dest = dest
        self.regularisation = regularisation
        self.weight = 0 if (initZero) else rand.uniform(-0.5, 0.5)

        self.lossDelta = 0
        self.accDelta = 0
        self.numAccumulatedDelta = 0

        self.isDead = False

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
 :return:                   The network, as a 2D array of neurons
"""
def buildNetwork(networkShape, activation, outputActivation, regularisation, initZero=False):
    network = []

    # Build the input layer
    currLayer = []
    for i in range(0, networkShape[0]):
        currLayer.append(Neuron("I"+str(i), activation, initZero))
    network.append(currLayer)

    # Build the hidden layers
    id = 1
    for i in range(1, len(networkShape)-1):
        currLayer = []
        for _ in range(0, networkShape[i]):
            currNeuron = Neuron(str(id), activation, initZero); id += 1
            # Build the wires from the previous layer to this layer
            for k in range(0, networkShape[i-1]):
                prevNeuron = network[i-1][k]
                currWire = Wire(prevNeuron, currNeuron, regularisation, initZero)
                currNeuron.inputs.append(currWire); prevNeuron.outputs.append(currWire)
            currLayer.append(currNeuron)
        network.append(currLayer)

    # Build the output layer
    id = 1
    currLayer = []
    for i in range(0, networkShape[-1]):
        currNeuron = Neuron("O"+str(id), outputActivation, initZero); id += 1
        # Build finally the last set of wires
        for j in range(0, networkShape[-2]):
            prevNeuron = network[-1][j]
            currWire = Wire(prevNeuron, currNeuron, regularisation, initZero)
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
    if len(inputs) != len(inputLayer):
        raise Exception("Size incompatibility between network and inputs!")
    
    # Feed in the inputs
    for i in range(0, len(inputs)):
        inputNeuron = inputLayer[i]
        inputNeuron.outputVal = inputs[i]

    # Propagate the input
    for i in range(1, len(network)):
        currLayer = network[i]
        for j in range(0, len(currLayer)):
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
def backProp(network, label, lossFn):
    totalLoss = 0
    # Find ∂C/∂a0 >> outputDelta for each outputNeuron
    outputLayer = network[-1]
    length = len(outputLayer)
    for i in range(0, length):
        outNeuron = outputLayer[i]
        outNeuron.outputDelta = lossFn.der(outNeuron.outputVal, label[i], length)
        totalLoss += lossFn.out(outNeuron.outputVal, label[i], length)
    
    # Iterate through each layer backwards
    for i in range(len(network) - 1, 0, -1):
        layer = network[i]

        """
        Find partial chain
           [[∂C/∂m] * [∂am/∂zm * ... * ∂a(n+1)/∂z(n+1) * ∂z(n+1)/∂an]] * [∂a(n)/∂z(n)]
        up to the total input z of a neuron in the nth layer.
        
        Note the idea is to perform this iteratively, noting that the sum up
        until ∂a(n+1)/∂z(n+1) can be substituted with n' = n+1.
        """
        for j in range(0, len(layer)):
            neuron = layer[j]
            neuron.inputDelta = neuron.outputDelta * neuron.activation.der(neuron.inputVal)
            neuron.accDelta += neuron.inputDelta
            neuron.numAccumulatedDelta += 1
            #if (neuron.accDelta > 0): print(f"Neuron AccDelta: {neuron.accDelta}")

        # Find the gradient of the weights of each wire
        for j in range(0, len(layer)):
            neuron = layer[j]
            for k in range(0, len(neuron.inputs)):
                wire = neuron.inputs[k]
                if (wire.isDead): continue
                wire.lossDelta = wire.source.outputVal * neuron.inputDelta
                wire.accDelta += wire.lossDelta; wire.numAccumulatedDelta += 1
                #print(f"Wire AccDelta: {wire.accDelta}")  
        
        if (i == 1): continue

        # Cumulate the chain-rule paths of influence to the next (backwards) layer
        prevLayer = network[i-1]
        for j in range(0, len(prevLayer)):
            prevNeuron = prevLayer[j]
            prevNeuron.outputDelta = 0
            for k in range(0, len(prevNeuron.outputs)):
                wire = prevNeuron.outputs[k]
                prevNeuron.outputDelta += wire.weight * wire.dest.inputDelta

    return totalLoss
    

"""
Updates the parameters of a given network given the previously accumulated
partial derivatives
"""
def updateParams(network, learningRate, regLambda):
    # Ignore the input layer
    for i in range(1, len(network)):
        layer = network[i]
        
        for j in range(0, len(layer)):
            # Step the bias
            neuron = layer[j]
            if (neuron.numAccumulatedDelta <= 0): continue
            neuron.bias -= learningRate * (neuron.accDelta / neuron.numAccumulatedDelta)
            neuron.accDelta = 0; neuron.numAccumulatedDelta = 0

            # Step the gradient of the wires feeding into this neuron
            for k in range(0, len(neuron.inputs)):
                wire = neuron.inputs[k]
                if (wire.isDead): continue
                if (wire.numAccumulatedDelta <= 0): continue
                wire.weight -= learningRate * (wire.accDelta / wire.numAccumulatedDelta)

                # Process regularisation; derive the penalty component of the
                # cost w.r.t. the weight, i.e. ∂C/∂W = ∂(L+P)/∂W = ∂L/∂W + ∂P/∂W (=dP(W)) [P_W : W⊆R->R],
                # added to desensitise the parameters (lower weight between neurons) (fixing the 'ridge').
                regDer = wire.regularisation.der if (wire.regularisation is not None) else 0
                regWeight = wire.weight - (learningRate * regLambda * regDer)
                if (wire.regularisation == Regularisations.L1):
                    # For L1 regularisation, check if the desensitisation then
                    # crosses 0; if so, kill. (Useful for useless parameters).
                    if (wire.weight * regWeight < 0):
                        wire.weight = 0; wire.isDead = True
                else: # Otw for L2, 
                    wire.weight = regWeight
                
                wire.accDelta = 0; wire.numAccumulatedDelta = 0

    return

def printNetwork(network, limit=50):
    count = 0
    for i in range(1, len(network)):
        layer = network[i]
        for j in range(0, len(layer)):
            neuron = network[i][j]
            # if (neuron.bias != 0.0 and count < limit):
            #     print(f"Neuron: {neuron.bias}")
            #     count += 1
            if (neuron.outputVal != 0.0):
                print(f"Neuron Out: {neuron.outputVal}")
                count += 1
            # for k in range(0, len(neuron.outputs)):
            #     if (neuron.outputs[k].weight != 0.0 and count < limit):
            #         print(f"Wire: {neuron.outputs[k].weight}")
            #         count += 1
                
def printWeights(network, limit=50):
    count = 0
    for i in range(0, len(network)):
        layer = network[i]
        for j in range(0, len(layer)):
            neuron = network[i][j]
            # if (neuron.bias != 0.0 and count < limit):
            #     print(f"Neuron: {neuron.bias}")
            #     count += 1
            # if (neuron.outputVal != 0.0):
            #     print(f"Neuron Out: {neuron.outputVal}")
            #     count += 1
            for k in range(0, len(neuron.outputs)):
                if (neuron.outputs[k].weight != 0.0 and count < limit):
                    print(f"Wire {neuron.outputs[k].id}: {neuron.outputs[k].weight}")
                    count += 1