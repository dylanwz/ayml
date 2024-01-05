import math

class Function:
    def __init(self, out, der):
        self.out = out
        self.der = der

# Defined Loss Functions
class Loss:
    SQUARE = Function(
        lambda activation,label : 1/(len(activation)) * sum(map(lambda x,y : pow(x-y,2), activation,label)),
        lambda activation,label,index : (2*(activation[index] - label[index]))/len(activation)
    )
    
# Defined Activation Functions
class Activations:
    TANH = Function(
        lambda activation : math.tahn(activation),
        lambda val : 1 - pow(math.tahn(val), 2)
    )

    RELU = Function(
        lambda activation : math.max(0, activation),
        lambda val : 0 if val <= 0 else 1
    )

    SIGMOID = Function(
        lambda activation : 1 / (1 + math.exp(-1 * activation)),
        lambda val : 1 - pow(1 / (1 + math.exp(-1 * val)), 2)
    )

    LINEAR = Function(
        lambda activation : activation,
        lambda val : 1
    )

# Defined Regularisation Functions
class Regularisations:
    
    # The idea here is to use
    # ∂C/∂W = ∂(L+P)/∂W = ∂L/∂W + ∂P/∂W (=dP(W)) [P_W : W⊆R->R],
    # i.e. the penalty P is for a 'slope' (e.g. C = L + slope^2). Then the gradient of this cost
    # is the cumulative gradient of L and P w.r.t. each parameter. Let's consider a given one, W.
    # The 'slope' of W here is just the weight of the wire. So P = weight^2, for instance, and we
    # can derive directly w.r.t. the weight.

    L1 = Function(
        lambda slope : abs(slope),
        lambda slope : -1 if slope < 0 else (1 if slope > 0 else 0)
    )

    L2 = Function(
        lambda slope : 1/2 * pow(slope, 2),
        lambda slope : slope
    )
