correct_result = [[2,4], [3,6], [7,14], [6,12], [5,10], [30,60], [70,140]]

class simpleNeurone:

    def __init__(self, ):
        self.weight = 0.5
        self.speed_learning = 0.1
        self.history_weight = []
        self.error_on_prediction=[]

    def predict(self, input):
        return self.weight * input

    def adjustWeight(self, result, wantedResult):
        error = result - wantedResult
        self.error_on_prediction.append(error)
        theWeight = self.weight
        if error > 0:
            self.weight = theWeight - self.speed_learning * abs(error)
        else:
            self.weight = theWeight + self.speed_learning * abs(error)
        self.history_weight.append(self.weight)

    def train(self, testingSample):
        for i in range(270):
            result = self.predict(testingSample[i%len(testingSample)][0])
            self.adjustWeight(result, testingSample[i%len(testingSample)][1])
            print(f"Round {i} ;Input => {testingSample[i%len(testingSample)][0]}; Prediction => {result}; Error => {self.error_on_prediction[-1]}")

if __name__ == "__main__":
    print(" === [AI_Algos] === ")
    print("Basic Neurone adaptation")
    print("Adjust a neurone so it can predict the f(x) = 2x function.")
    print()
    simpleNeurone = simpleNeurone()
    simpleNeurone.train(correct_result)