import * as fns from "@/nn/functions";

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
  derPath: number;
  accDer: number;
  numAccDer: number;


  constructor(id: string, activation: fns.Activation) {
    this.id = id;
    
    this.bias = 0.1;
    this.activation = activation;
    this.totalInput = 0;
    this.output = 0;
    this.inputs = [];
    this.outputs = [];

    this.derPath = 0; this.accDer = 0; this.numAccDer = 0;

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

  accDer: number;
  numAccDer: number;

  constructor(id: string, source: Node, dest: Node) {
    this.id = id;

    this.source = source;
    this.dest = dest;
    this.weight = Math.random() - 0.5;

    this.accDer = 0; this.numAccDer = 0;
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
    const node = new Node(String(id), activation); id++;
    inputLayer.push(node);
  }
  network.push(inputLayer);
  
  // Next build the hidden layers and output layer
  for (let j = 1; j < shape.length; j++) {
    let layer = [];
    for (let c = 0; c < shape[j]; c++) {
      const node = new Node(String(id), j === shape.length - 1 ? outputActivation : activation); id++;
      // Links: add links between this node and each previous node
      for (let p = 0; p < shape[j - 1]; p++) {
        const prev = network[j - 1][p];
        const link = new Link(prev.id + " -> " + node.id, prev, node);
        node.inputs.push(link);
        prev.outputs.push(link);
      }
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
  let totalLoss = 0;
  
  const outputLayer = network[network.length -1];
  if (labels.length != outputLayer.length) {
    throw new Error(`Size error: incompatibility between given input size ${labels.length} and network input size ${outputLayer.length}!`);
  }

  // First; compute the loss w.r.t. each output neuron
  const numOutputs = outputLayer.length;
  for (let i = 0; i < numOutputs; i++) {
    const node = outputLayer[i];
    const label = labels[i];
    node.derPath = lossFn.der(node.output, label, numOutputs) * node.activation.der(node.totalInput);
    node.accDer += node.derPath; node.numAccDer++;
    totalLoss += lossFn.output(node.output, label, numOutputs);
  }
  totalLoss = totalLoss / numOutputs;

  // Iterate over the hidden layers backwards
  for (let i = network.length - 2; i >= 0; i--) {
    const layer = network[i];
    for (let j = 0; j < layer.length; j++) {
      const node = layer[j];
      let sum = 0;
      for (let k = 0; k < node.outputs.length; k++) {
        const link = node.outputs[k];
        const prevNode = link.dest;
        sum += prevNode.derPath * link.weight;

        link.accDer += node.output * prevNode.derPath;
        link.numAccDer++;
      }
      node.derPath = sum * node.activation.der(node.output);
      node.accDer += node.derPath; node.numAccDer++;
    }
  }

  return totalLoss;
}

export function updateParams(network: Node[][], learningRate: number) {
  for (let i = 1; i < network.length; i++) {
    const layer = network[i];
    for (let j = 0; j < layer.length; j++) {
      const node = layer[j];
      if (node.numAccDer > 0) {
        node.bias -= learningRate * node.accDer / node.numAccDer;
        node.accDer = 0; node.numAccDer = 0;
      }
      for (let k = 0; k < node.inputs.length; k++) {
        const link = node.inputs[k];
        if (link.numAccDer > 0) {
          link.weight -= learningRate * link.accDer / link.numAccDer;
          link.accDer = 0; link.numAccDer = 0;
        }
      }
    }
  }
}