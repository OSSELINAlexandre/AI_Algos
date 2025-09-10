######################################################################################
#                           Based on Andrej Karpathy Video                           #
#                           Building makemore Part 2: MLP                            #
#                     https://www.youtube.com/watch?v=TCH_1BHY58I&t                  #
######################################################################################
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

#creation of generator for easier understanding through iteration
g = torch.Generator().manual_seed(2147483647)

#Bulding dictionary
words = open("names.txt").read().splitlines()
intToString = {i:s for i,s in enumerate(sorted(set("".join(words))))}
stringToNum = {s:i for i,s in enumerate(sorted(set("".join(words))))}
intToString[26] = "."
stringToNum["."] = 26

#Bulding the dataset of words, and its context
charPredict = []
charContext = []
sizeBloc = 3

sizeTraining = int(len(words) * 0.9)

for word in words[:sizeTraining]:
    wordF = word + "."
    context = [26] * sizeBloc
    for w in wordF:
        charPredict.append(stringToNum[w])
        context = context[1:] + [stringToNum[w]]
        charContext.append(context)

context = torch.tensor(charContext)
tensChar = torch.tensor(charPredict, dtype= torch.long)

charPredictVal = []
charContextVal = []
for word in words[sizeTraining:]:
    wordF = word + "."
    con = [26] * sizeBloc
    for w in wordF:
        charPredictVal.append(stringToNum[w])
        con = con[1:] + [stringToNum[w]]
        charContextVal.append(con)


contextVal = torch.tensor(charContextVal)
tensCharVal = torch.tensor(charPredictVal, dtype= torch.long)

### Architecture of MLP
#Embedding Layer
embeddingTens = torch.randn(size=(27,2),requires_grad=True,dtype=torch.float, generator=g)

#Layer tanh
W = torch.randn((6,100), requires_grad=True, generator=g)
b = torch.randn(100, requires_grad=True,generator=g)

#Output layer
Wout = torch.randn(100, 27, requires_grad=True, generator=g)
Bout = torch.randn(27, requires_grad=True, generator=g)

#Size of the model in parameters:
paramList = [embeddingTens, W, b, Wout, Bout]

for p in paramList:
    p.requires_grad = True

# Finding the right learning rate

#lri = torch.linspace(0.001, 1, 51000)
#lossi = []

#Training time !!
for i in range(5000):
    ix = torch.randint(0 , context.shape[0], (32, ), generator=g)

    emb = embeddingTens[context[ix]]

    tanhLayer = torch.tanh(emb.view(-1, 6) @ W + b)
    out = tanhLayer @ Wout + Bout
    loss = F.cross_entropy(out, tensChar[ix])

    for p in paramList:
        p.grad = None

    loss.backward()

    #from png, we can see that 0.8 grad seems to be the optimal without
    #too much noise
    for p in paramList:
        p.data += -( 0.8 * p.grad)
    print(">", loss.item())

#plt.plot(lri, lossi)
#plt.savefig('loss.png', dpi=300, bbox_inches='tight')
#Getting now the global loss to see if it is good
emb = embeddingTens[contextVal]
tanhLayer = torch.tanh(emb.view(-1, 6) @ W + b)
out = tanhLayer @ Wout + Bout
loss = F.cross_entropy(out, tensCharVal)
print("===>", loss.item())