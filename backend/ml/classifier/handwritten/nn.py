import random as rand

class Neuron:
    bias = 0.1
    activation = None


    inputs = []
    outputs = []
    inputVal = None
    outputVal = None

    inputDelta = 0
    outputDelta = 0
    accInputDelta = 0
    numAccumulatedDelta = 0

    def __init__(self, id, activation, initZero=False):
        self.id = id
        if (initZero): self.bias = 0

    def update(self):
        self.inputVal = self.bias
        for i in self.inputs:
            self.inputVal += i.weight * i.source.output
        self.outputVal = self.activation.output(self.inputVal)


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


def buildNetwork(networkShape, activation, outputActivation, regularisation, inputIds, initZero=False):
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

