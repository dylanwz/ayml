class NetworkFunction {
    out: Function;
    der: Function;

    constructor(out: Function, der: Function) {
        this.out = out;
        this.der = der;
    }
}

export const functionFactory = (fn: string) => {
    switch(fn.toLowerCase()) {
        case("square"):
            const squareSum = function(arr1: number[], arr2: number[]) {
                var sum = 0;
                for (let i = 0; i < arr1.length; i++) {
                    sum += Math.pow(arr1[i] - arr2[i], 2);
                }
                return sum / arr1.length;
            };    
            return new NetworkFunction(
                (activations: number[], labels: number[]) => 1/activations.length * squareSum(activations, labels),
                (activations: number[], labels: number[], i: number) => 2*(activations[i] - labels[i])/activations.length);
        
        case("tanh"):
            return new NetworkFunction(
                (activation: number) => Math.tanh(activation),
                (val: number) => 1);

        case("relu"):
            return new NetworkFunction(
                (activation: number) => Math.max(0, activation),
                (val: number) => val <= 0 ? 0 : 1);

        case("sigmoid"):
            return new NetworkFunction(
                (activation: number) => 1/(1 + Math.exp(-1 * activation)),
                (val: number) => 1 - Math.pow(1/(1+Math.exp(-1 * val)), 2));

        case("linear"):
            return new NetworkFunction(
                (activation: number) => activation,
                (val: number) => 1);
        
        case("l1"):
            return new NetworkFunction(
                (slope: number) => Math.abs(slope),
                (slope: number) => slope < 0 ? -1 : (slope > 0 ? 1 : 0));

        case("l2"):
            return new NetworkFunction(
                (slope: number) => 1/2 * Math.pow(slope, 2),
                (slope: number) => slope);

    }
};