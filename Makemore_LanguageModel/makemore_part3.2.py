#Interlude : hyperparameters optimizations to compute the best solution
import torch
import torch.nn.functional as F
import Utilities as ut
import matplotlib.pyplot as plt

g = torch.Generator().manual_seed(2147483647)

class Linear:

    def __init__(self, fan_in, fan_out, bias=True):
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.weight = torch.randn((fan_in, fan_out)) / fan_in ** 0.5 #Kaiming init copy
        self.bias = torch.randn(fan_out) if bias else None
        self.out = 0

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


n_bloc_size = 3
myUtil = ut.Utilies(contextLen=n_bloc_size, sz_train=0.9)
myUtil.dataSetConstructor()

n_in = 64
n_embedding = 5
n_hidden = 100
n_out = 27

C = torch.randn((27, n_embedding))

myNeuralNetwork = [Linear(n_bloc_size * n_embedding , n_hidden), Tanh(),
                   Linear(n_hidden                  , n_hidden), Tanh(),
                   Linear(n_hidden                  , n_hidden), Tanh(),
                   Linear(n_hidden                  , n_hidden), Tanh(),
                   Linear(n_hidden                  , n_hidden)]

with torch.no_grad():
    myNeuralNetwork[-1].weight.data *= 0.1
    for layer in myNeuralNetwork:
        if (isinstance(layer, Linear)):
            layer.weight.data *= 5/3

parameters = [C] + [p for layer in myNeuralNetwork for p in layer.parameters()]
lossi = []
learning_rate = 0.1
ud = []
for i in range(1000):
    for p in parameters:
        p.requires_grad=True

    idx = torch.randint(0, myUtil.trainingSize, (n_in,))
    emb = C[myUtil.inputTraining[idx]]

    x = emb.view(-1, n_embedding * n_bloc_size)

    for layer in myNeuralNetwork:
        x = layer(x)

    loss = F.cross_entropy(x, myUtil.truthTraining[idx])

    for layer in myNeuralNetwork:
        layer.out.retain_grad()

    for p in parameters:
        p.grad = None
    loss.backward()


    for p in parameters:
        p.data += - learning_rate * p.grad
    with torch.no_grad():
        ud.append([((learning_rate * p.grad).std() / p.data.std()).log10().item() for p in parameters])

    lossi.append(loss.item())

# === Loss vizualisation ===
plt.figure(figsize=(15,7))
plt.xlabel("iterations")
plt.ylabel("loss")
plt.title("Loss evolution for network")
plt.plot(lossi)
plt.savefig(f'pngForCode/pt3_A_lossFuncFig.png', dpi=300, bbox_inches='tight')


# === Tanh activation visualization ===
plt.figure(figsize=(20,5))
plt.xlabel("Values")
plt.title("Activation histogram")
legends = []
for i, lay in enumerate(myNeuralNetwork):
    if isinstance(lay, Tanh):
        t = lay.out
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f"layer {i} ({lay.__class__.__name__})")

plt.legend(legends)
plt.savefig(f'pngForCode/pt3_B_ActivationFig.png', dpi=300, bbox_inches='tight')

# === Gradient descent Tanh visualization ===
plt.figure(figsize=(20,5))
plt.xlabel("Values")
plt.title("Grandient descent values")
legends = []
for i, lay in enumerate(myNeuralNetwork):
    if isinstance(lay, Tanh):
        t = lay.out.grad
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f"layer {i} ({lay.__class__.__name__})")

plt.legend(legends)
plt.savefig(f'pngForCode/pt3_C_GradientDescentFig.png', dpi=300, bbox_inches='tight')

# === Gradient descent Params visualization ===
plt.figure(figsize=(20, 4)) # width and height of the plot
plt.title("Grandient descent values from params")
legends = []
for i,p in enumerate(parameters):
  t = p.grad
  if p.ndim == 2:
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'{i} {tuple(p.shape)}')
plt.legend(legends)
plt.savefig(f'pngForCode/pt3_D_GradientDescent_FromParam.png', dpi=300, bbox_inches='tight')

# === Learning rate evolution visualization ===
plt.figure(figsize=(20, 4))
plt.title("Learning rate compared to the weight modified viz")
legends = []
for i,p in enumerate(parameters):
  if p.ndim == 2:
    plt.plot([ud[j][i] for j in range(len(ud))])
    legends.append('param %d' % i)

plt.plot([0, len(ud)], [-3, -3], 'k')
plt.savefig(f'pngForCode/pt3_E_LearningRateViz.png', dpi=300, bbox_inches='tight')


