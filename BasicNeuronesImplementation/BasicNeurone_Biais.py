
correct_result_highLearningRate= [[2,7], [3.5,10], [3,9], [2.5,8], [2.2,7.4], [3.2,9.4], [2.7,8.4]]
correct_result_highLearningRate_Int= [[2,7], [3,9], [4,11], [5,13], [6,15], [7,17], [8,19]]
correct_result_lowLearningRate = [[2,7], [3,9], [7,17], [6,15], [5,13], [30,63], [70,143]]

class simpleNeurone_biais:

    def __init__(self):
        self.weight = 0.5
        self.bias = 0.1
        self.speed_learning = 0.1
        self.history_weight = []
        self.history_bias = []
        self.error_on_prediction=[]
        self.tolerable_error = 10**-6

    def predict(self, input):
        return self.weight * input + self.bias

    def adjustWeight(self, input, result, wantedResult):
        error = result - wantedResult
        self.error_on_prediction.append(error)

        gradient_weight = error * input
        gradient_biais = error

        theWeight = self.weight
        theBias = self.bias

        self.weight = theWeight - gradient_weight * self.speed_learning
        self.bias = theBias - gradient_biais * self.speed_learning

        if abs (round(self.bias) - self.bias) < self.tolerable_error:
            self.bias = round(self.bias)

        if abs (round(self.weight) - self.weight) < self.tolerable_error:
            self.weight = round(self.weight)

        self.history_weight.append(theWeight)
        self.history_bias.append(theBias)


    def train(self, testingSample, numberOfIteration):
        for i in range(numberOfIteration):
            result = self.predict(testingSample[i%len(testingSample)][0])
            self.adjustWeight(testingSample[i%len(testingSample)][0], result, testingSample[i%len(testingSample)][1])
            print(f"\tRound {i} ;Input => {testingSample[i%len(testingSample)][0]}; Prediction => {result}; Error => {self.error_on_prediction[-1]}")
        try:
            self.error_on_prediction.index(0)
            print(f"Iteration needed before obtained => {self.error_on_prediction.index(0)}")
        except ValueError:
            print(f"Iteration done and no result obtained after => {numberOfIteration}")

if __name__ == "__main__":
    print(" === [AI_Algos] === ")
    print("Basic Neurone adaptation")
    print("Adjust a neurone so it can predict the f(x) = 2x + 3 function.")
    print()
    print("Value are close one to the other : learning rate hyperparameter can be higher")
    simpleNeurone = simpleNeurone_biais()
    simpleNeurone.train(correct_result_highLearningRate_Int, 250)

    print("Value are far from the other : learning rate hyperparameter has to be if not cannot succes.")
    simpleNeuroneLowLearning_CannotSucceed = simpleNeurone_biais()
    simpleNeuroneLowLearning_CannotSucceed.train(correct_result_lowLearningRate, 450)

    print("Value are far from the other : learning rate hyperparameter has to be lower and higher training needed.")
    simpleNeuroneLowLearning_CanSucceed = simpleNeurone_biais()
    simpleNeuroneLowLearning_CanSucceed.speed_learning = 0.001
    simpleNeuroneLowLearning_CanSucceed.train(correct_result_lowLearningRate, 18000)


