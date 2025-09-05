#Section that duplicates the exemple of bigram.
#It's purpose is to be a synthesis and a summary as clear as possible
#of all the manipulated objects

import torch
import torch.nn.functional as F

#Bulding dictionary
words = open("names.txt").read().splitlines()
stringToNum = {i:s for i,s in enumerate(sorted(set("".join(words))))}
numToString = {s:i for i,s in enumerate(sorted(set("".join(words))))}
stringToNum[26] = "."
numToString["."] = 26

#Bulding dataset of bigram
entry = []
expectedOut = []
for word in words:
    wordF = "." + word + "."
    for ch1, ch2 in zip(wordF, wordF[1:]):
        entry.append(numToString[ch1])
        expectedOut.append(numToString[ch2])

#Creating NL + encoding
g = torch.Generator().manual_seed(77)
Layer = torch.randn((27,27), dtype=torch.float, generator=g, requires_grad=True)
xTens = torch.tensor(entry, dtype=torch.long)
xTensEncoding = F.one_hot(xTens, num_classes= 27).float()

#Computing loss
MatrixOutput = (xTensEncoding @ Layer)
MatrixPositive = MatrixOutput.exp()
MatrixNormalized = MatrixPositive / torch.sum(MatrixPositive, 1 ,keepdim=True)
relevantData = MatrixNormalized[torch.arange(len(expectedOut)) , expectedOut]
loss =- relevantData.log().mean()
print(f"Current loss with random weight => {loss.item()}")

#Training time !!
loss.backward()

for i in range(1500):
    #Adjust weight in the opposite direction of gradient in order to reduce the loss
    Layer.data += - (0.77 * Layer.grad)

    #Reseting the gradient
    Layer.grad = None

    #Computing the loss
    MatrixOutput = (xTensEncoding @ Layer)
    MatrixPositive = MatrixOutput.exp()
    MatrixNormalized = MatrixPositive / torch.sum(MatrixPositive, 1, keepdim=True)
    relevantData = MatrixNormalized[torch.arange(len(expectedOut)), expectedOut]
    loss = - relevantData.log().mean()

    #Backpropagate
    loss.backward()
print()
print("Final iteration loss", loss.item())
print("Generation of names with newly trained model")

for i in range(50):
    idW = [26]
    word=""
    while(True):
        characterPrediction = (F.one_hot(torch.tensor(idW, dtype=torch.long), num_classes=27).float() @ Layer)
        characterPositive = characterPrediction.exp()
        characterNormalized = characterPositive / torch.sum(characterPositive, 1, keepdim=True)
        nextCharacterIndex = torch.multinomial(characterNormalized, generator=g, num_samples=1, replacement=True)
        word += stringToNum[nextCharacterIndex.item()]
        idW = [nextCharacterIndex]
        if nextCharacterIndex.item() == 26:
            print(word)
            break