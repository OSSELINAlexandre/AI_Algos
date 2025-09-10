#Section that duplicates the exemple of MLP.
#The purpose is to be a synthesis and a summary as clear as possible
#of all the manipulated objects
import torch
import matplotlib.pyplot as plt

words = open(file="names.txt").read().splitlines()
stoi = {i:s for s, i in enumerate(sorted(set("".join(words))))}
itos = {s:i for s, i in enumerate(sorted(set("".join(words))))}
stoi["."] = 26
itos[26]= "."
g = torch.Generator().manual_seed(2147483647)
size_bloc = 3
context = []
allContext = []
allContextConverted = []
output = []

for word in words:
    context = size_bloc * ["."]
    wordF = word + "."
    for w in wordF:
        allContext.append(context)
        allContextConverted.append([stoi[c] for c in context])
        output.append(stoi[w])
        context = context[1:] + [w]

#Creating the tensors & the architecture

#Embedding
C = torch.randn((27,2), generator=g, requires_grad=True)
#Tanh layer
tan = torch.randn((6,100), generator=g, requires_grad=True, dtype=torch.float)
bTan = torch.randn(100, generator=g, requires_grad=True, dtype=torch.float)
#Output layer
out = torch.rand((100, 27), generator=g, requires_grad=True, dtype=torch.float)
bOut = torch.randn(27, generator=g, requires_grad=True, dtype=torch.float)

yEncoding = torch.tensor(output)
contextTens = torch.tensor(allContextConverted, dtype=torch.long)

allParams = [C, tan, bTan, out, bOut]
lossIteration = []
iter = []

#Creation of batch in order to speed up the gradient descent processing
batch_size = 64

#Separation of dataset in two : train and validate
trainingSize = int(len(words)*0.9)
trainingSet = words[:trainingSize]
TestSet = words[trainingSize:]
testSetSize = len(TestSet)

#Training time !!
for i in range(20000):
    idx = torch.randint(0, len(trainingSet), (batch_size,), generator=g)
    embeddingTens = C[contextTens[idx]]
    finalEmbedding = torch.cat(torch.unbind(embeddingTens, 1), 1)
    tanLayer = torch.tanh(finalEmbedding @ tan + bTan)
    outputLayer = (tanLayer @ out + bOut).exp()
    normalizedOutput = outputLayer / torch.sum(outputLayer, 1, keepdim=True)
    loss = -(normalizedOutput[torch.arange(batch_size), yEncoding[idx]].log().mean())
    loss.backward()
    iter.append(i)
    lossIteration.append(loss.item())
    for p in allParams:
        p.data += - (0.1 * p.grad)
        p.grad = None


plt.plot(iter,  lossIteration)
plt.savefig('sumpUp_pt2_LossBatch.png', dpi=300, bbox_inches='tight')

#Verification of loss
embeddingTens = C[contextTens[:testSetSize]]
finalEmbedding = torch.cat(torch.unbind(embeddingTens, 1), 1)
tanLayer = torch.tanh(finalEmbedding @ tan + bTan)
outputLayer = (tanLayer @ out + bOut).exp()
normalizedOutput = outputLayer / torch.sum(outputLayer, 1, keepdim=True)
loss = -(normalizedOutput[torch.arange(testSetSize), yEncoding[:testSetSize]].log().mean())

print("Loss on new dataset provided to MLP", loss.item(), ": last loss on seen data", lossIteration[1:].pop())