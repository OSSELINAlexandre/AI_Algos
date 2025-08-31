######################################################################################
#                           Based on Andrej Karpathy Video                           #
#                The spelled-out intro to language modeling: building makemore       #
#                     https://www.youtube.com/watch?v=PaCmpygFfXo&t=2179s            #
######################################################################################
import torch
import matplotlib.pyplot as plt

words = open('names.txt').read().splitlines()
alphabet = sorted(set("".join(words)))
alphabet.append(".")
alphaNumber = {i:s for i, s in enumerate(alphabet)}
numberAlpha = {s:i for i, s in enumerate(alphabet)}

#bulding the dict with occurrences of pair of words
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


