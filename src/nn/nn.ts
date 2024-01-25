import * as fns from "./functions";

export class Node {
  id: string;
  bias: number;
  activation: fns.Activation;
  totalInput: number;
  output: number;
  inputs: Link[];
  outputs: Link[];

  // Final derivatives; `derPath` should end up as `∂a/∂z` so that `∂z/∂w` is
  //                    performed in `updateParams`
  derA: number;
  derZ: number;
  accDer: number;
  numAccDer: number;


  constructor(id: string, activation: fns.Activation) {
    this.id = id;
    this.activation = activation;
    this.totalInput = 0;
    this.output = 0;
    this.inputs = [];
    this.outputs = [];
    this.bias = 0.1;

    this.derA = 0; this.derZ = 0; this.accDer = 0; this.numAccDer = 0;
  }

  initBias(bias: fns.Bias) {
    this.bias = bias.gen(this.inputs.length);;
  }

  /**
   * Note; thhis update algorithm for forward propagation is from
   * https://github.com/tensorflow/playground/blob/master/src/nn.ts.
   */
  updateOutput(): number {
    this.totalInput = this.bias;
    for (let j = 0; j < this.inputs.length; j++) {
      let link = this.inputs[j];
      this.totalInput += link.weight * link.source.output;
    }
    this.output = this.activation.output(this.totalInput);
    return this.output;
  }
}

export class Link {
  id: string;
  
  source: Node;
  dest: Node;
  weight: number;

  der: number;
  accDer: number;
  numAccDer: number;

  constructor(id: string, source: Node, dest: Node) {
    this.id = id;

    this.source = source;
    this.dest = dest;
    this.weight = Math.random() - 0.5;

    this.der = 0; this.accDer = 0; this.numAccDer = 0;
  }
}

export function buildNetwork(
  shape: number[],
  activation: fns.Activation,
  outputActivation: fns.Activation): Node[][]
{
  let network: Node[][] = [];
  let id = 1;

  // First build the input layer
  let inputLayer = [];
  for (let i = 0; i < shape[0]; i++) {
    const node = new Node(String(id), activation);
    node.initBias(fns.Bias.ZERO);
    id++;
    inputLayer.push(node);
  }
  network.push(inputLayer);
  
  // Next build the hidden layers and output layer
  for (let i = 1; i < shape.length; i++) {
    let layer = [];
    for (let j = 0; j < shape[i]; j++) {
      const node = new Node(String(id), i === shape.length - 1 ? outputActivation : activation);
      id++;
      // Links: add links between this node and each previous node
      for (let k = 0; k < shape[i - 1]; k++) {
        const prev = network[i - 1][k];
        const link = new Link(prev.id + " -> " + node.id, prev, node);
        node.inputs.push(link);
        prev.outputs.push(link);
      }
      node.initBias(fns.Bias.HE);
      layer.push(node);
    }
    network.push(layer);
  }

  return network;
}

export function forwardProp(network: Node[][], inputs: number[]) {
  let inputLayer = network[0];
  if (inputs.length != inputLayer.length) {
    throw new Error(`Size error: incompatibility between given input size ${inputs.length} and network input size ${inputLayer.length}!`);
  }

  // First: update the input layer
  for (let i = 0; i < inputs.length; i++) {
    const node = inputLayer[i];
    node.output = inputs[i];
  }
  
  for (let i = 1; i < network.length; i++) {
    const layer = network[i];
    for (let j = 0; j < layer.length; j++) {
      const node = layer[j];
      node.updateOutput();
    }
  }

  return network[network.length - 1];
}

export function backProp(network: Node[][], labels: number[], lossFn: fns.Loss) {
  const outputLayer = network[network.length - 1];
  if (labels.length != outputLayer.length) {
    throw new Error(`Size error: incompatibility between given input size ${labels.length} and network input size ${outputLayer.length}!`);
  }

  //    A. Compute ∂C/∂a(L), for each output neuron
  const numOutputs = outputLayer.length;
  for (let i = 0; i < numOutputs; i++) {
    const node = outputLayer[i];
    const label = labels[i];
    node.derA = lossFn.der(node.output, label, numOutputs);
  }

  /**   B.  Iterate through the layers backwards, and for each node of each layer...
   *          1)  Perform   ∂a(L)/∂z(L) = act'(z(L))      using the `derA` value stored to give `derZ`;
   *          2)  Perform   ∂z(L)/∂W(L_{i->j}) = a(L-1)   to each weight extending into that node.
   *          3)  Perform   Σ ( ∂z(L)/∂a(L-1) ) = W       over each node in the current layer,
   *              and store in the next (backward) layer's nodes' `derA`.
   *    Reset temporary `derPath` value backpropagation before each assignment.
  */
  for (let i = network.length - 1; i > 0; i--) {
    
    const layer = network[i];
    for (let j = 0; j < layer.length; j++) {
      const node = layer[j];
      node.derZ = node.derA * node.activation.der(node.totalInput);
      node.accDer += node.derZ;
      node.numAccDer++;
    }

    for (let j = 0; j < layer.length; j++) {
      const node = layer[j];
      for (let k = 0; k < node.inputs.length; k++) {
        const link = node.inputs[k];
        link.der = node.derZ * link.source.output;
        link.accDer += link.der;
        link.numAccDer++;
      }
    }

    if (i === 1) {
      continue;
    }

    const prevLayer = network[i - 1];
    for (let j = 0; j < prevLayer.length; j++) {
      const prevNode = prevLayer[j];
      prevNode.derA = 0;
      for (let k = 0; k < prevNode.outputs.length; k++) {
        const link = prevNode.outputs[k];
        prevNode.derA += link.dest.derZ * link.weight;
      }
    }
  }
}

export function updateParams(network: Node[][], learningRate: number) {
  for (let i = 1; i < network.length; i++) {
    const layer = network[i];
    for (let j = 0; j < layer.length; j++) {
      const node = layer[j];
      if (node.numAccDer > 0) {
        node.bias -= learningRate * (node.accDer / node.numAccDer);
        node.accDer = 0;
        node.numAccDer = 0;
      }
      for (let k = 0; k < node.inputs.length; k++) {
        const link = node.inputs[k];
        if (link.numAccDer > 0) {
          link.weight -= learningRate * (link.accDer / link.numAccDer);
          link.accDer = 0; 
          link.numAccDer = 0;
        }
      }
    }
  }
}