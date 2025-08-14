import math

import numpy as np

myData = open("restaurant.csv")
array = np.array(myData.read().split("\n"))
attributes = np.array(array[0].split(","))
myData.close()
array = np.delete(array, 0, 0)

def returnBooleanFromString(string):
    string = np.strings.lower(string)
    string = np.strings.strip(string)
    if np.equal(string, "yes"):
        return True
    elif np.equal(string, "no"):
        return False
    else:
        return string

def returnColumn(column, dataSet):
    mySet = []
    for row in dataSet:
        cell = np.array(row.split(","))
        if cell.size > 1:
            mySet.append(returnBooleanFromString(cell[column]))
    return mySet


def calculateGeneralEntropy(scores):
    yes = 0
    no = 0
    for score in scores:
        if returnBooleanFromString(score) == True:
            yes +=1
        else:
            no +=1
    yesProbability = yes / len(scores)
    noProbability = no / len(scores)
    return -yesProbability * math.log(yesProbability, 2) - noProbability * math.log(noProbability, 2)

def SubGroupEnthropy (indexToCompute, score):
    yes = 0
    no = 0
    finalComputation = 0
    for index in indexToCompute:
        if score[index] == True:
            yes += 1
        else:
            no +=1

    if yes > 0:
        yesProbability = yes / len(indexToCompute)
        finalComputation =+ - (yesProbability * math.log(yesProbability, 2))
    if no > 0:
        noProbability = no / len(indexToCompute)
        finalComputation =+ - (noProbability * math.log(noProbability, 2))


    return finalComputation

def generalEntropyForGroup (listOfResult, totalSamples):
    generalEntropy = 0
    for attributes in listOfResult:
        generalEntropy += attributes[0]/totalSamples * attributes[0][0]

    return generalEntropy


def calculateEntropy(indexOfAttr, attributes, scores, attributeColumn):
    computedAttributes = []
    computedScoreForAttributes = [[[]]]
    for attribute in attributes:
        continueProcessing = True
        for elem in computedAttributes:
            if elem == attribute:
                continueProcessing = False

        if continueProcessing == True:
            computedAttributes.append(attribute)
            indexToCompute = []
            for attr in attributeColumn:
                if (attr == attribute):
                    indexToCompute.append(attributes.index(attr))

            print("WhereAreWe => ", computedScoreForAttributes.append(attribute))
            print("WhereAreWe => ", indexToCompute)
            print("WhereAreWe => ", SubGroupEnthropy(indexToCompute, scores))
            computedScoreForAttributes.append(attribute)
            computedScoreForAttributes[indexOfAttr] = indexToCompute
            computedScoreForAttributes[indexOfAttr][0] = SubGroupEnthropy(indexToCompute, scores)

    return generalEntropyForGroup (computedScoreForAttributes ,len (scores))


print("Tree decision processing !")
attributesScore = returnColumn(len(attributes ) - 1, array)

print("== Global Data == ")
print("=> Array")
print(array)
print("=> attributes")
print(attributes)
print("=> attributesScore")
print(attributesScore)
print("== End Global Data == ")

i = 1
for attributes in attributes:
    result = calculateEntropy (i, attributes, attributesScore, returnColumn(i, array))
    i +=1