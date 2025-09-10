#Interlude : hyperparameters optimizations to compute the best solution
import torch
import matplotlib.pyplot as plt

g = torch.Generator().manual_seed(2147483647)

class ModelHolder:

    def __init__(self, name, embSize, batchSize, tanhLayerSize, numberOfTraining, contextualBlocSize):
        #architecture
        self._name = name
        self._embeddingDimension = embSize
        self._batchSize = batchSize
        self._tanhLayerSize = tanhLayerSize
        self._numberOfTraining = numberOfTraining
        self._contextLen = contextualBlocSize
        self._sizeOfTrainingSet = 0.9
        self._tr = 0
        self._tt = 0
        self.sc = 0
        #variable from dataset
        self._trainingSet = []
        self._testSet = []
        self._truthTraining = []
        self._truthTesting = []
        self._inputTraining = []
        self._inputTesting = []
        #results
        self.finalArchivedLoss = 77.7
        self.finalTrainingLoss = 77.7

    def dataSetConstructor(self, dataSet):
        stoi = {i: s for s, i in enumerate(sorted(set("".join(dataSet))))}
        itos = {s: i for s, i in enumerate(sorted(set("".join(dataSet))))}
        stoi["."] = 26
        itos[26] = "."

        size_bloc = self._contextLen
        contextConverted = []
        output = []
        wordToContextIndex = []
        for i, word in enumerate(dataSet):
            context = size_bloc * ["."]
            wordF = word + "."
            for w in wordF:
                contextConverted.append([stoi[c] for c in context])
                output.append(stoi[w])
                context = context[1:] + [w]
            wordToContextIndex.append([i, len(contextConverted)])

        trainingSize = int(len(words) * self._sizeOfTrainingSet)
        self._tr = trainingSize
        self._tt = int(len(words)) - self._tr
        self._trainingSet = dataSet[:trainingSize]
        self._testSet  = dataSet[trainingSize:]
        self.sc = wordToContextIndex[trainingSize][1]
        self.totalTest = len(contextConverted) - self.sc
        self._inputTraining =  contextConverted[:self.sc]
        self._inputTesting = contextConverted[self.sc:]
        self._truthTraining = output[:self.sc]
        self._truthTesting = output[self.sc:]

    def train(self, dataSet):
        self.dataSetConstructor(dataSet)

        C = torch.randn((27,self._embeddingDimension), generator=g, requires_grad=True)
        tan = torch.randn((self._embeddingDimension * self._contextLen,self._tanhLayerSize), generator=g, requires_grad=True, dtype=torch.float)
        bTan = torch.randn(self._tanhLayerSize, generator=g, requires_grad=True, dtype=torch.float)
        out = torch.rand((self._tanhLayerSize, 27), generator=g, requires_grad=True, dtype=torch.float)
        bOut = torch.randn(27, generator=g, requires_grad=True, dtype=torch.float)
        yEncoding = torch.tensor(self._truthTraining)
        contextTens = torch.tensor(self._inputTraining, dtype=torch.long)
        allParams = [C, tan, bTan, out, bOut]
        lossIteration = []
        iter = []

        #Training time !!
        for i in range( self._numberOfTraining):

            idx = torch.randint(0,self._tr, (self._batchSize,), generator=g)
            embeddingTens = C[contextTens[idx]]
            finalEmbedding = torch.cat(torch.unbind(embeddingTens, 1), 1)
            tanLayer = torch.tanh(finalEmbedding @ tan + bTan)
            outputLayer = (tanLayer @ out + bOut).exp()
            normalizedOutput = outputLayer / torch.sum(outputLayer, 1, keepdim=True)
            loss = -(normalizedOutput[torch.arange(self._batchSize), yEncoding[idx]].log().mean())
            loss.backward()
            iter.append(i)
            lossIteration.append(loss.item())
            for p in allParams:
                p.data += - (0.1 * p.grad)
                p.grad = None

        self.finalTrainingLoss = loss.item()

        yEncodingTest = torch.tensor(self._truthTesting)
        contextTensTest = torch.tensor(self._inputTesting, dtype=torch.long)
        embeddingTens = C[contextTensTest]
        finalEmbedding = torch.cat(torch.unbind(embeddingTens, 1), 1)
        tanLayer = torch.tanh(finalEmbedding @ tan + bTan)
        outputLayer = (tanLayer @ out + bOut).exp()
        normalizedOutput = outputLayer / torch.sum(outputLayer, 1, keepdim=True)
        self.finalArchivedLoss = (-(normalizedOutput[torch.arange(self.totalTest), yEncodingTest].log().mean())).item()

        plt.plot(iter,  lossIteration)
        plt.text(0.02, 0.98, f'Loss finale: {self.finalArchivedLoss:.4f}',
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.savefig(f'interlude_{self._name}.png', dpi=300, bbox_inches='tight')

words = open(file="names.txt").read().splitlines()
sumUp2 = ModelHolder(name="sumupSimilar", embSize=2, batchSize=64, tanhLayerSize=100, numberOfTraining=20000, contextualBlocSize=3)
sumUp2.train(words)
print("sumUp2", sumUp2.finalArchivedLoss)
testA = ModelHolder(name="testA", embSize=15, batchSize=64, tanhLayerSize=100, numberOfTraining=20000, contextualBlocSize=3)
testA.train(words)
print("testA",testA.finalArchivedLoss)
testB = ModelHolder(name="testB", embSize=2, batchSize=64, tanhLayerSize=3000, numberOfTraining=20000, contextualBlocSize=3)
testB.train(words)
print("testB",testB.finalArchivedLoss)
testC = ModelHolder(name="testC", embSize=2, batchSize=64, tanhLayerSize=100, numberOfTraining=20000, contextualBlocSize=6)
testC.train(words)
print("testC",testC.finalArchivedLoss)
testD = ModelHolder(name="everything", embSize=4, batchSize=64, tanhLayerSize=200, numberOfTraining=40000, contextualBlocSize=6)
testD.train(words)
print("testD",testD.finalArchivedLoss)


