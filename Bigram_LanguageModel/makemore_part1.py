######################################################################################
#                           Based on Andrej Karpathy Video                           #
#                The spelled-out intro to language modeling: building makemore       #
#                     https://www.youtube.com/watch?v=PaCmpygFfXo&t=2179s            #
######################################################################################
import torch
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F

words = open('names.txt').read().splitlines()
alphabet = sorted(set("".join(words)))
alphabet.append(".")
alphaNumber = {i:s for i, s in enumerate(alphabet)}
numberAlpha = {s:i for i, s in enumerate(alphabet)}

#bulding the dict with occurrences of paire of words
pairWords = {}
for word in words:
    updatedWord = "." + word + "."
    for charA, charB in zip(updatedWord, updatedWord[1:]):
        bigram = charA + "|" + charB
        pairWords[bigram] = pairWords.get(bigram, 0) +1

#bulding the tensor with pair of words occurrences
tens = torch.zeros((27,27), dtype=torch.int32)
for entry in pairWords:
    charA = entry[0]
    charB = entry[2]
    indexA = numberAlpha[charA]
    indexB = numberAlpha[charB]
    tens[indexA, indexB] = pairWords[entry]

#creation of generator for easier understanding through iteration
    g = torch.Generator().manual_seed(77)

def GenerationWithCounting():

    #bulding graphical representation of current names
    plt.figure(figsize=(20, 20))
    plt.imshow(tens, cmap='Blues')

    for i in range(27):
        for j in range(27):
            charPrint = alphaNumber[i] + alphaNumber[j]
            plt.text(j, i, charPrint, ha="center", va="bottom", fontsize="6", weight="bold" ,color="gray")
            plt.text(j, i, str(tens[i, j].item()), ha="center", va="top", fontsize="8", color="black")

    plt.axis('off')
    plt.title('Occurrences of bigram in dataset', fontsize=20, weight="bold")
    plt.show()

       # === Word generator Based on Occurrences ===
    def recursiveWordBulding(id):

        if id == 26:
            return alphaNumber[id]
        else:
            newRow = tens[id, :].float()
            newRow = newRow / newRow.sum()
            newId = torch.multinomial(newRow, num_samples=1, replacement=True, generator=g)
            res = recursiveWordBulding(newId.item())
            return alphaNumber[id] + res


    for i in range(50):
        aRow = tens[26, :].float()
        aRow = aRow / aRow.sum()  # matrice computation with : M x 1/sum
        res = torch.multinomial(aRow, num_samples=1, replacement=True, generator=g)
        print(recursiveWordBulding(res.item()))

def ComputeModelEfficiency():

    #Applying the mean negative log loss function to compute the efficiency of the model
    entries = ["emma", "alexandre", "jesus", "qzw"]
    for word in entries:
        negativeLossFunction = 0
        n = 0
        wordA = "." + word + "."
        for ch1, ch2 in zip(wordA, wordA[1:]):
            prob = tens[numberAlpha[ch1], numberAlpha[ch2]] / torch.sum(tens[numberAlpha[ch1], :], 0 , keepdim=True)
            logProb = torch.log(prob)
            n += 1
            negativeLossFunction -=  logProb
        print(f"Proba for name \"{word}\" \t\t\t=>{negativeLossFunction.item()/n}")
    print()
    print("Proba for untrained model \t=>", - math.log(1/27))
    print()
    negativeLossFunction = 0
    n = 0
    for word in words:
        negativeLossFunction = 0
        n = 0
        wordA = "." + word + "."
        for ch1, ch2 in zip(wordA, wordA[1:]):
            prob = tens[numberAlpha[ch1], numberAlpha[ch2]] / torch.sum(tens[numberAlpha[ch1], :], 0 , keepdim=True)
            logProb = torch.log(prob)
            n += 1
            negativeLossFunction -=  logProb

    print(f"Proba correct correspondance of untrained model is:{negativeLossFunction.item()/n}")

def ComputeWordsWithLayerOfNeurons():
    N = torch.randn((27,27), generator=g, dtype=torch.float,requires_grad=True)
    #27 entry per neuron, with 27 neurons
    tensEntry = []
    tensOutput = []

    entries = [ "aleksander"]
    for word in entries:
        wordA = "." + word + "."
        for i , (ch1, ch2) in enumerate(zip(wordA, wordA[1:])):
            tensEntry.append(numberAlpha[ch1])
            tensOutput.append(numberAlpha[ch2])
            #Bulding the dataset to work upon

    entryTens = torch.Tensor(tensEntry)
    print("++")
    print(entryTens)
    outputTens = torch.Tensor(tensOutput)
    xEncoding = F.one_hot(entryTens.long(), num_classes= 27).float()
    print(xEncoding)

    logits = xEncoding @ N # only positive numbers
    counts = logits.exp()
    probs = counts / torch.sum(counts, 1, keepdim=True)
    print(probs[0].sum())

def TrainModelOnWords():
    #bulding the inputs and desired output dataset
    neuralNet = torch.randn((27,27), generator=g, dtype=torch.float, requires_grad=True)

    entry = []
    expectedOutput = []
    for word in words:
        wordF = "." + word + "."
        for ch1, ch2 in zip(wordF, wordF[1:]):
            entry.append(numberAlpha[ch1])
            expectedOutput.append(numberAlpha[ch2])

    tensX = torch.tensor(entry, dtype=torch.float32)


    xEnc = F.one_hot(tensX.long(), num_classes=27).float() #Input that can now be given to a Layer
    matrixOuttput = (xEnc @ neuralNet)
    loggitsMatrix = matrixOuttput.exp()
    normalizedMatrix = loggitsMatrix / torch.sum(loggitsMatrix,1,keepdim=True)

    # Has to be done with a vectorized form : otherwise the tensor object isn't able to backpropagate
    ReferenceLoss =- normalizedMatrix[torch.arange(len(entry)), expectedOutput].log().mean()
    ReferenceLoss.backward()


    #Time for training !!
    for i in range(150):
        neuralNet.data += - (10 * neuralNet.grad)
        neuralNet.grad = None
        matrixOuttput = (xEnc @ neuralNet)
        loggitsMatrix = matrixOuttput.exp()
        normalizedMatrix = loggitsMatrix / torch.sum(loggitsMatrix,1,keepdim=True)

        loss =- normalizedMatrix[torch.arange(len(entry)), expectedOutput].log().mean()
        loss.backward()

    print("Gradient descent on single layer neuron with torch framework object manipulation :")
    print("\t => NeuralLayer's loss with no training :", ReferenceLoss.item())
    print("\t => NeuralLayer's loss trained on words :", loss.item())

ComputeWordsWithLayerOfNeurons()