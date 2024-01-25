export interface Bias {
  gen: (size: number) => number;
}

export class Bias {
  public static ZERO: Bias = {
    gen: (size: number) => 0
  }

  // For ReLU; vanishing/exploding gradient problem
  public static HE: Bias = {
    gen: (size: number) => gaussianRandom(0, Math.sqrt(2/size))
  }
}

// Standard Normal variate using Box-Muller transform.
function gaussianRandom(mean=0, stdev=1) {
  const u = 1 - Math.random(); // Converting [0,1) to (0,1]
  const v = Math.random();
  const z = Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
  // Transform to the desired mean and standard deviation:
  return z * stdev + mean;
}


/**
 * The remaining contents of this file is from TensorFlow;
 * https://github.com/tensorflow/playground/blob/master/src/nn.ts
 */

/**
 * An Loss function and its derivative.
 */
export interface Loss {
  output: (output: number, target:  number, size: number) => number;
  der: (output: number, target: number, size: number) => number;
}

/** A node's activation function and its derivative. */
export interface Activation {
  output: (input: number) => number;
  der: (input: number) => number;
}

/** Built-in Loss functions */
export class Loss {
  public static SQUARE: Loss = {
    output: (output: number, target: number, size: number) => (Math.pow(output - target, 2))/size,
    der: (output: number, target: number, size: number) => (2 * (output - target)) / size
  };
}

/** Polyfill for TANH */
(Math as any).tanh = (Math as any).tanh || function(x: number) {
  if (x === Infinity) {
    return 1;
  } else if (x === -Infinity) {
    return -1;
  } else {
    let e2x = Math.exp(2 * x);
    return (e2x - 1) / (e2x + 1);
  }
};

/** Built-in activation functions */
export class Activations {
  public static TANH: Activation = {
    output: x => (Math as any).tanh(x),
    der: x => {
      let output = Activations.TANH.output(x);
      return 1 - output * output;
    }
  };
  public static RELU: Activation = {
    output: x => Math.max(0, x),
    der: x => x <= 0 ? 0 : 1
  };
  public static SIGMOID: Activation = {
    output: x => 1 / (1 + Math.exp(-x)),
    der: x => {
      let output = Activations.SIGMOID.output(x);
      return output * (1 - output);
    }
  };
  public static LINEAR: Activation = {
    output: x => x,
    der: x => 1
  };
}