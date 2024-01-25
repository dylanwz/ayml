// To run, change `"module": "esnext" to "module": "commonjs" and
// comment out `moduleResolution` in `tsconfig.json`.

import * as nn from "../../src/nn/nn";
import * as fn from "../../src/nn/functions"
import { flattenedTestData, flattenedTrainData, trainLabels, testLabels } from "./generateMNIST";



// testBuildNetworkLinksForward();
// testForwardProp();
// testBackProp();
// testUpdate();
testMNIST();

function testBuildNetworkLinksForward() {
  const network = nn.buildNetwork([4, 2, 2, 4], fn.Activations.LINEAR, fn.Activations.LINEAR);
  for (let i = 0; i < network.length; i++) {
    console.log(`Layer: ${i}`);
    for (let j = 0; j < network[i].length; j++) {
      const node = network[i][j];
      console.log(`Node: ${node.id}`);
      if (i < network.length - 1) {
        for (let k = 0; k < node.outputs.length; k++) {
          console.log(`Link: ${node.outputs[k].id}`);
        }
      }
    }
  }
}

function testBuildNetworkLinksBackward() {
  const network = nn.buildNetwork([4, 2, 2, 4], fn.Activations.LINEAR, fn.Activations.LINEAR);
  for (let i = 1; i < network.length; i++) {
    console.log(`Layer: ${i}`);
    for (let j = 0; j < network[i].length; j++) {
      const node = network[i][j];
      console.log(`Node: ${node.id}`);
      for (let k = 0; k < node.inputs.length; k++) {
        console.log(`Link: ${node.inputs[k].id}`);
      }
    }
  }
}

function testForwardProp() {
  const network = nn.buildNetwork([2,1,2], fn.Activations.RELU, fn.Activations.LINEAR);
  for (let i = 0; i < 10; i++) {
    let node = nn.forwardProp(network, [Math.random(),Math.random()]);
    let nodeVals = node.map((n) => n.output);
    console.log(nodeVals);
  }
  let node = nn.forwardProp(network, [1,1]);
  let nodeVals = node.map((n) => n.output);
  console.log(nodeVals);
  for (let i = 0; i < network.length; i++) {
    console.log(`Layer: ${i}`);
    for (let j = 0; j < network[i].length; j++) {
      const node = network[i][j];
      console.log(`Node: ${node.id}, Bias: ${node.bias}`);
      if (i < network.length - 1) {
        for (let k = 0; k < node.outputs.length; k++) {
          console.log(`Link: ${node.outputs[k].id}, Weight: ${node.outputs[k].weight}`);
        }
      }
    }
  }
}

// function testBackProp() {
//   const network = nn.buildNetwork([2,1,2], fn.Activations.RELU, fn.Activations.SIGMOID);
//   for (let i = 0; i < 10; i++) {
//     let node = nn.forwardProp(network, [Math.random(),Math.random()]);
//     let nodeVals = node.map((n) => n.output);
//     console.log(nodeVals);
//   }
//   nn.forwardProp(network, [1,1]);
//   nn.backProp(network, [0,0], fn.Loss.SQUARE);
//   for (let i = 0; i < network.length; i++) {
//     console.log(`Layer: ${i}`);
//     for (let j = 0; j < network[i].length; j++) {
//       const node = network[i][j];
//       console.log(`Node: ${node.id}, Bias: ${node.bias}, AccDer: ${node.accDer}`);
//       if (i < network.length - 1) {
//         for (let k = 0; k < node.outputs.length; k++) {
//           console.log(`Link: ${node.outputs[k].id}, Weight: ${node.outputs[k].weight}, AccDer: ${node.outputs[k].accDer}`);
//         }
//       }
//     }
//   }
// }

// function testUpdate() {
//   const network = nn.buildNetwork([2,1,2], fn.Activations.RELU, fn.Activations.LINEAR);
//   for (let i = 0; i < 10; i++) {
//     let node = nn.forwardProp(network, [Math.random(),Math.random()]);
//     let nodeVals = node.map((n) => n.output);
//     console.log(nodeVals);
//   }
//   nn.forwardProp(network, [1,1]);
//   nn.backProp(network, [0,0], fn.Loss.SQUARE);
//   nn.updateParams(network, 0.03);
//   for (let i = 0; i < network.length; i++) {
//     console.log(`Layer: ${i}`);
//     for (let j = 0; j < network[i].length; j++) {
//       const node = network[i][j];
//       console.log(`Node: ${node.id}, Bias: ${node.bias}, AccDer: ${node.accDer}`);
//       if (i < network.length - 1) {
//         for (let k = 0; k < node.outputs.length; k++) {
//           console.log(`Link: ${node.outputs[k].id}, Weight: ${node.outputs[k].weight}, AccDer: ${node.outputs[k].accDer}`);
//         }
//       }
//     }
//   }
// }

function testMNIST() {
  const network = nn.buildNetwork([784, 128, 10], fn.Activations.RELU, fn.Activations.SIGMOID);
  for (let i = 0; i < flattenedTrainData.length; i++) {
    console.log(`Training: ${i}`);
    nn.forwardProp(network, flattenedTrainData[i]);
    let label = [0,0,0,0,0,0,0,0,0,0];
    label[trainLabels[i]] = 1;
    nn.backProp(network, label, fn.Loss.SQUARE);
    if ((i+1) % 5 === 0) {
      nn.updateParams(network, 0.8);
    }
  }

  let corr = 0;
  let num = 0;
  let errors = 0;
  let finalResu: number[] = [];
  for (let i = 0; i < flattenedTestData.length; i++) {
    console.log(`Testing: ${i}`);
    const respo = nn.forwardProp(network, flattenedTestData[i]);
    if (flattenedTestData[i] === flattenedTestData[i-1]) {
      errors++;
    }
    const resu = respo.map((n) => n.output);
    console.log(`Result: ${resu} and label: ${testLabels[i]}`);
    if (findMaxIndex(resu) === testLabels[i]) {
      corr++;
    }
    num++;
    finalResu = respo.map((n) => n.totalInput);
  }
  console.log(corr/num);
  console.log(errors);

  // for (let i = 0; i < network.length; i++) {
  //   for (let j = 0; j < network[i].length; j++) {
  //     const node = network[i][j];
  //     console.log(`Node: ${node.id}, Input: ${node.totalInput}, Output: ${node.output}$, Bias: ${node.bias}`);
  //     if (i < network.length - 1) {
  //       for (let k = 0; k < node.outputs.length; k++) {
  //         console.log(`Link: ${node.outputs[k].id}, Weight: ${node.outputs[k].weight}`);
  //       }
  //     }
  //   }
  // }
  // console.log(finalResu);
  // console.log(corr/num);
}

function testMNISTGen() {
  console.log(flattenedTestData);
}

function findMaxIndex(arr: number[]): number {
  if (arr.length === 0) {
    throw new Error("Array is empty");
  }

  let maxIndex = 0;
  let maxValue = arr[0];

  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > maxValue) {
      maxIndex = i;
      maxValue = arr[i];
    }
  }

  return maxIndex;

}