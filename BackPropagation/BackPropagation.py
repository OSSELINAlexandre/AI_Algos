######################################################################################
#                           Based on Andrej Karpathy Video                           #
#   The spelled-out intro to neural networks and backpropagation: building micrograd #
#                     https://www.youtube.com/watch?v=VMj-3S1tku0&t=2047s            #
######################################################################################

import matplotlib
from graphviz import Digraph
import math
matplotlib.use('TkAgg')


#Copied from Micrograd, no reflexion done
def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


#Copied from Micrograd, no reflexion done
def draw_dot(root, format='svg', rankdir='LR'):
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})  # , node_attr={'rankdir': 'TB'})

    for n in nodes:
        dot.node(name=str(id(n)), label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

class Value:
    def __init__(self, x, _children=(), _op='', label=''):
        self.data = x
        self._op = _op
        self.grad = 0
        self._backward = lambda: None
        self.label = label
        self._prev = set(_children)

    def cleanGrad(self):
        self.grad = 0
        for child in self._prev:
            child.cleanGrad()

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        #Need to implement transformation from primitive to Class value, otherwise NeuronNetwork would not work
        other = other if isinstance(other, Value) else Value (other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad = out.grad
            other.grad = out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        #Need to implement transformation from primitive to Class value, otherwise NeuronNetwork would not work
        other = other if isinstance(other, Value) else Value (other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad
        out._backward = _backward
        return out

    #necessary function to simplify the process in NeuronNetwork for Loss computation and
    #creation of object that can optimize output based on gradient (tr - out) **2
    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1
    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __radd__(self, other):
        return self + other

    def __pow__(self, other):
        out = Value(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        o = Value (t, (self,) , label="o", _op="tanh")
        def _backward():
            self.grad = (1 - t**2) * o.grad

        o._backward = _backward
        return o

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


def basicNeuron():
    x1 = Value(2.0 , label='x1')
    w1 = Value(0.3 , label='w1')
    x2 = Value(1.5 , label='x2')
    w2 = Value(1.5 , label='w2')
    b = Value(-1.5, label='b')

    x1w1 = x1 * w1; x1w1.label="x1w1"
    x2w2 = x2 * w2; x2w2.label="x2w2"
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label="x1w1x2w2"
    f = x1w1x2w2 + b; f.label="funcNeur"
    o = f.tanh()
    o.grad = 1.0
    o.backward()
    return o

output = basicNeuron()

draw_dot(output).render()

