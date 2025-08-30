######################################################################################
#                           Based on Andrej Karpathy Video                           #
#   The spelled-out intro to neural networks and backpropagation: building micrograd #
#                     https://www.youtube.com/watch?v=VMj-3S1tku0&t=2047s            #
######################################################################################

import random
import matplotlib
matplotlib.use('TkAgg')
from BackPropagation import Value
from BackPropagation import draw_dot

matplotlib.use('TkAgg')

class Neuron:

    def __init__(self, noConnection, idNeuron, idLayer, non_lin=True):
        self.weights = [Value(random.uniform(-1,1)) for _ in range(noConnection)]
        self.biases = Value(0)
        self.activationFunc = non_lin
        #necessary for the final output to be something different than a value between 0 and 1
        self.id= "IDL:" + str(idLayer) + ":IDN:" + str(idNeuron)

    def __call__(self, x):
        act = sum(w1 * x1 for w1, x1 in zip(self.weights, x)) + self.biases
        return act.tanh() if self.activationFunc else act

    #Parameters on which the network can be optimized.
    def parameters(self):
        return self.weights + [self.biases]

class Layer:

    def __init__(self, noConnection, noNeurons, idLayer ,**kwargs):
        self.neurons = [Neuron(noConnection,idNeuron, idLayer, **kwargs) for idNeuron in range(noNeurons)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out #If not done, draw_dot does not work because it is
    #expecting an Value object and not a [Value]

    def parameters(self):
        res = []
        for neuron in self.neurons:
            res.extend(neuron.parameters())
        return res

class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], i,non_lin=i!=len(nouts) - 1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x) # L'entrée de l'un est la sortie de l'autre.
        return x

    def parameters(self):
        res = []
        for layer in self.layers:
            res.extend(layer.parameters())
        return res


#draw_dot(Ml(x)).render()
dataset = [
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0]
]

truth = [
    0,
    1,
    1,
    0
]
Ml = MLP (2, [4 , 4 , 1])

loss = 0
lossEvol = []
comp = []
trainingRound = 1000
for i in range(trainingRound):
    loss = 0
    for tr, result in zip(truth, dataset):
        out = Ml(result)

        #compute loss
        loss += (tr - out)**2
        loss.backward()

        if (trainingRound -1) == i:
            comp.append([tr, out.data])
            draw_dot(out).render()

        #adjust weight
        for p in Ml.parameters():
            p.data += (-0.01 * p.grad)

        out.cleanGrad()

    lossEvol.append([i, loss])

print(" === XOR Function === ")
print(comp)