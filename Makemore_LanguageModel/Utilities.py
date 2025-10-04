
import torch
class Utilies:

    def __init__(self, contextLen, sz_train):
        self.stoi =  []
        self.itos =  []
        self._contextLen = contextLen
        self._sr_training = sz_train
        self.inputTraining = []
        self.truthTraining = []
        self.inputTesting = []
        self.truthTesting = []
        self.trainingSize = 0

    def dataSetConstructor(self):
        words = open(file="names.txt").read().splitlines()
        self.stoi = {i: s for s, i in enumerate(sorted(set("".join(words))))}
        self.itos = {s: i for s, i in enumerate(sorted(set("".join(words))))}
        self.stoi["."] = 26
        self.itos[26] = "."

        size_bloc = self._contextLen
        contextConverted = []
        output = []
        wordToContextIndex = []
        for i, word in enumerate(words):
            context = size_bloc * ["."]
            wordF = word + "."
            for w in wordF:
                contextConverted.append([self.stoi[c] for c in context])
                output.append(self.stoi[w])
                context = context[1:] + [w]
            wordToContextIndex.append([i, len(contextConverted)])

        self.trainingSize = int(len(words) * self._sr_training)
        sc = wordToContextIndex[self.trainingSize][1]
        self.inputTraining = torch.tensor(contextConverted[:sc])
        self.inputTesting = torch.tensor(contextConverted[sc:])
        self.truthTraining = torch.tensor(output[:sc])
        self.truthTesting = torch.tensor(output[sc:])