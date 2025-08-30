from BasicNeuronesImplementation.BasicNeurone_Biais import simpleNeurone_biais

correct_result_highLearningRate= [[2,7], [3.5,10], [3,9], [2.5,8], [2.2,7.4], [3.2,9.4], [2.7,8.4]]
correct_result_highLearningRate_Int= [[2,7], [3,9], [4,11], [5,13], [6,15], [7,17], [8,19]]
correct_result_lowLearningRate = [[2,7], [3,9], [7,17], [6,15], [5,13], [30,63], [70,143]]


class simpleNeurone_Momentum:

    def __init__(self):
        self.weight = 0.5
        self.bias = 0.1
        self.momentum = 0.9
        self.current_weight_momentum = 0
        self.current_bias_momentum = 0
        self.history_momentum_weight = []
        self.history_momentum_bias = []
        self.speed_learning = 0.1
        self.history_weight = []
        self.history_bias = []
        self.error_on_prediction=[]
        self.tolerable_error = 10**-6

    def predict(self, input):
        return self.weight * input + self.bias

    def adjustWeight(self, input, result, wantedResult):
        error = wantedResult - result

        gradient_Weight = error * self.speed_learning * input
        gradient_Bias = error * self.speed_learning

        cur_Momentum_Weigh = self.current_weight_momentum
        cur_Momentum_Bias = self.current_bias_momentum

        current_momentum_weigh = self.momentum * cur_Momentum_Weigh + gradient_Weight
        current_momentum_bias = self.momentum * cur_Momentum_Bias + gradient_Bias

        theWeight = self.weight
        theBias = self.bias
        self.weight = theWeight + current_momentum_weigh
        self.bias = theBias + current_momentum_bias

        self.history_momentum_weight.append(self.current_weight_momentum)
        self.history_momentum_bias.append(self.current_bias_momentum)

        self.error_on_prediction.append(error)
        self.history_weight.append(self.weight)
        self.history_bias.append(self.bias)

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
    #print("Basic Neurone adaptation")
    #print("Adjust a neurone so it can predict the f(x) = 2x + 3 function.")
    #print("=> Adding momentum to the neurone.")
    #testa_simpleNeurone_Momentum = simpleNeurone_Momentum()
    #testa_simpleNeurone_Momentum.train(correct_result_highLearningRate_Int, 1000)

    #print ("=> Is it better than average formula ? ")
    #SimpleNeurone = simpleNeurone_biais()
    #SimpleNeurone.train(correct_result_highLearningRate_Int, 1000)

    print("Answer is no : Basic implenetation is better for high learning rate.")
    print("Is it the same for needing low learning rate ? ")
    testo_simpleNeurone_Momentum_Second = simpleNeurone_Momentum()
    testo_simpleNeurone_Momentum_Second.train(correct_result_lowLearningRate, 1000)

    SimpleNeurone_b = simpleNeurone_biais()
    SimpleNeurone_b.train(correct_result_highLearningRate_Int, 1000)



