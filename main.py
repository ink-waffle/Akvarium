import numpy as np
import math
import random
import pandas as pd
import plotly.express as px
import dask
def sigmoid(z):
    if -z > 100:
        return 0
    elif -z < -100:
        return 1
    return 1/(1+math.exp(-z))
class Model:
    def __init__(self):
        self.w = np.array([[0.0]], dtype=np.float64)
        self.b = 0
        self.trainingData = pd.DataFrame([], columns=['VarName', 'Value', 'Iteration'])
        for i in range(self.w.shape[0]):
            self.trainingData.loc[len(self.trainingData)] = ('w' + str(i), self.w[i, 0], 0)
        self.trainingData.loc[len(self.trainingData)] = ('b', self.b, 0)
    def predict(self, x: np.array):
        # print((self.w*x), self.b)
        return sigmoid((np.transpose(self.w)*x).item() + self.b)
    def train(self, m: np.array, y: np.array, trainingStep, trainingIterations):
        for tr_i in range(trainingIterations):
            dw = np.zeros((self.w.shape[0], 1))
            db = 0
            def backProppagateForCase(case):
                nonlocal dw, db
                dz = (-y[:,case] + self.predict(m[:,case])).item()
                dw += dz * m[:, case]
                db += dz
            # for i in range(m.shape[1]):
            #     dz = (-y[:,i] + self.predict(m[:,i])).item()
            #     dw += dz * m[:, i]
            #     db += dz

            delayed_local_function = dask.delayed(backProppagateForCase)
            delayed_tasks = [delayed_local_function(argument) for argument in range(m.shape[1])]
            dask.compute(*delayed_tasks, scheduler=dask.multiprocessing.get)
            dw /= m.shape[1]
            db /= m.shape[1]
            self.w -= trainingStep * dw
            self.b -= trainingStep * db
            for j in range(self.w.shape[0]):
                self.trainingData.loc[len(self.trainingData)] = ('w' + str(j), self.w[j, 0], tr_i + 1)
            self.trainingData.loc[len(self.trainingData)] = ('b', self.b, tr_i + 1)



#Preparing test cases
testCasesAmount = 10000
m = np.zeros((1, testCasesAmount))
y = np.zeros((1, testCasesAmount))
for i in range(testCasesAmount):
    testCase = -0.1 + random.random()*0.2
    m[:, i] = np.array([testCase])
    y[:, i] = np.array([int(testCase > 0)])


if __name__ == '__main__':
    model = Model()
    model.train(m, y, 0.00001, 10)
    px.line(model.trainingData, x='Iteration', y = 'Value', color='VarName')