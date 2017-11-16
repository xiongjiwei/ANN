import numpy as np
import json

class OCRNeuralNetwork:

    theta1 = 0
    theta2 = 0

    input_layer_bias = 0
    hidden_layer_bias = 0

    LEARNING_RATE = 0.01 

    NN_FILE_PATH = 'parameter'# parameter save path 

    def __init__(self,num_hidden_nodes = 100):
        self.theta1 = self._rand_initialize_weights(784,num_hidden_nodes)
        self.theta2 = self._rand_initialize_weights(num_hidden_nodes,10)
        self.input_layer_bias = self._rand_initialize_weights(1,num_hidden_nodes)
        self.hidden_layer_bias = self._rand_initialize_weights(1,10)

    def _rand_initialize_weights(self, size_in, size_out):
        return [((x * 0.12) - 0.06) for x in np.random.rand(size_out,size_in)]

    def _sigmoid_scalar(self,z):
        return 1 / (1+np.exp(-z))

    def _sigmoid_prime(self,z):
        return np.multiply(1 / (1+np.exp(-z)), 1 - (1 / (1+np.exp(-z))))

    def _train(self,data,tra):
        actual_vals = [0] * 10
        actual_vals[tra] = 1  

        y1 = np.dot(np.mat(self.theta1), np.mat(data[0:]).T)
        sum1 = y1 + np.mat(self.input_layer_bias)
        y1 = self._sigmoid_scalar(sum1)

        y2 = np.dot(np.array(self.theta2),y1)
        y2 = np.add(y2,self.hidden_layer_bias)
        y2 = self._sigmoid_scalar(y2)

        output_errors = np.mat(actual_vals).T - np.mat(y2)
        hidden_errors = np.multiply(np.dot(np.mat(self.theta2).T, output_errors), self._sigmoid_prime(sum1))

        self.theta1 += self.LEARNING_RATE * np.dot(np.mat(hidden_errors), np.mat(data[0:]))
        self.theta2 += self.LEARNING_RATE * np.dot(np.mat(output_errors), np.mat(y1).T)
        self.hidden_layer_bias += self.LEARNING_RATE * output_errors
        self.input_layer_bias += self.LEARNING_RATE * hidden_errors

    def predict(self, test):
        y1 = np.dot(np.mat(self.theta1), np.mat(test).T)
        y1 += np.mat(self.input_layer_bias)
        y1 = self._sigmoid_scalar(y1)

        y2 = np.dot(np.array(self.theta2), y1)
        y2 += np.mat(self.hidden_layer_bias)
        y2 = self._sigmoid_scalar(y2)

        res = y2.T.tolist()[0]
        return res.index(max(res))

    def save(self):

        json_NN = {
            "theta1": [np_mat.tolist()[0] for np_mat in self.theta1],
            "theta2": [np_mat.tolist()[0] for np_mat in self.theta2],
            "b1": self.input_layer_bias[0].tolist()[0],
            "b2": self.hidden_layer_bias[0].tolist()[0]
        };

        with open(OCRNeuralNetwork.NN_FILE_PATH,'w') as nnFile:
            json.dump(json_NN,nnFile)

    def load(self):
        with open(OCRNeuralNetwork.NN_FILE_PATH) as nnFIle:
            nn = json.load(nnFIle)

        self.theta1 = [np.array(li) for li in nn['theta1']]
        self.theta2 = [np.array(li) for li in nn['theta2']]
        self.input_layer_bias = [np.array(nn['b1'][0])]
        self.hidden_layer_bias = [np.array(nn['b2'][0])]