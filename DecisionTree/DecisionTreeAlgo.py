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
            mySet.append(cell[column])
    return mySet

def numberOfGivenAttribute(attribute, dataSet):
    mySet = []
    for row in dataSet:
        if row == attribute:
            mySet.append(1)
        else:
            mySet.append(0)
    return mySet

def differentAttribute (dataSet):
    seenGroup = []
    for row in dataSet:
        if row not in seenGroup:
            seenGroup.append(row)
    return seenGroup

def numberOfGivenAttribute (aGroup, dataSet):
    number = 0
    for row in dataSet:
        if row == aGroup:
            number += 1
    return number

def EntropyForSubGroup (dataSet, decision, aGroup):
    sample = numberOfGivenAttribute(aGroup, dataSet)
    positivReac = 0
    negativReac = 0
    result = 0
    for id, row in enumerate (dataSet):
        if row == aGroup:
            if returnBooleanFromString (decision[id]):
                positivReac += 1
            else:
                negativReac += 1

    if positivReac > 0:
        result -= positivReac/sample * math.log(positivReac/sample, 2)

    if negativReac > 0:
        result -= negativReac / sample * math.log(negativReac/sample, 2)

    return result

def computeAttributeTotalEntropy (allResultsForGroup, totalSamples):
    totalGroupEntropy = 0
    for aGroup in allResultsForGroup:
        totalGroupEntropy =+ totalGroupEntropy + ((aGroup[1][0] / totalSamples) * aGroup[1][1])

    return totalGroupEntropy



def calculateEntropy(attribute, scores, attributeColumn):

    allResults = []
    for aGroupOfAttribute in differentAttribute (attributeColumn):
        infoPerGroup = [ numberOfGivenAttribute(aGroupOfAttribute, attributeColumn),EntropyForSubGroup (attributeColumn, scores, aGroupOfAttribute)]
        resultForColum = [aGroupOfAttribute, infoPerGroup]
        allResults.append(resultForColum)

    return [attribute, computeAttributeTotalEntropy (allResults, len (attributeColumn ))]


print("Tree decision processing !")
attributesScore = returnColumn(len(attributes) - 1, array)

attributes = attributes[:-1]
allResult = []
for idx, attribute in enumerate (attributes):
    result = calculateEntropy ( attribute, attributesScore, returnColumn(idx, array))
    allResult.append(result)

allResult.sort (key=lambda a: a[1])
print("List organized by the most influencial criteria to the least.")
print(allResult)