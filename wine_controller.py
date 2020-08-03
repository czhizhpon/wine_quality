# Autores: Sarmiento Bryan, Zhizhpon Eduardo

import numpy as np
import pandas as pd
import matplotlib.pyplot as pp

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class CBR:

    def __init__(self):
        super().__init__()
        self.k_repeat = 0.0
        self.k_mean = 0.0
        self.k_first = 0.0
        self.distances = []
        self.similarities = []
        self.first_neighbor = 0

    def loadDataBase(self, file_csv='winequality-red.csv'):
        dataBase = pd.read_csv(file_csv, delimiter=';')
        return dataBase

    def preprocess(self, dataBase, label='quality'):
        X = dataBase.drop(label, 1)
        y = dataBase[[label]]
        return X, y

    def getTrainTest(self, inputs, outputs, test_size=0.3, random_state=15):
        X_train, X_test, y_train, y_test = train_test_split(inputs, 
                                            outputs, test_size = 0.3, random_state = 15)
        return (X_train, X_test, y_train, y_test)

    def calc(self, X_train, X_test, weight = 0):
        self.euclideanDistance(X_train_v=X_train, X_test_v=X_test)
        self.calcSimilarities(new_wine=X_train, originals=X_test, weight=weight)

    def euclideanDistance(self, X_train_v, X_test_v):
        distancesList = []
        for i in X_test_v:
            distances = []
            for j in X_train_v:
                distance = np.sqrt(np.sum(np.square(j - i)))
                distance = np.around(distance, decimals=4)
                distances.extend([distance])
            distancesList.append(np.array(distances))
        self.distances = np.array(distancesList)
        return self.distances

    def k_neighbor(self, y_train, X_test, reference, method='min'):
        neighbor = []
        for i in range(X_test.shape[0]):
            if method == 'min':
                ref_values = reference[i].argsort()[:1]
            elif method == 'max':
                ref_values = reference[i].argsort()[::-1][:1]
            neighbor.extend([y_train[ref_values[0]]])
        self.first_neighbor = neighbor[0]
        return self.first_neighbor
        

    def k_neighbors(self, y_train, X_test, reference, k=2, method='min'):
        
        k_mean = []
        k_repeat = []
        for i in range(X_test.shape[0]):
            if method == 'min':
                ref_values = reference[i].argsort()[:k]
            elif method == 'max':
                ref_values = reference[i].argsort()[::-1][:k]
            neighbors = []
            for d in range(len(ref_values)):
                neighbors.extend([y_train[ref_values[d]]])

            k_mean.extend([self.meanK(neighbors)])

            k_repeat.extend([self.repeatK(neighbors)])
        return np.array(k_mean), np.array(k_repeat)

    def meanK(self, k):
        return np.sum(k)/len(k)
    
    def repeatK(self, k):
        (unique, counts) = np.unique(k, return_counts=True)
        if np.max(counts) == 1:
            return k[0][0]
        else:
            i = np.where(counts == np.max(counts))
        i = i[0][0]
        return unique[i]

    def getRelativeError(self, original, calculated):
        return np.around(np.abs(original - calculated)/original, 4)

        return (error_mean_y, error_repeat_y)
    def getPrecision(self, original, calculated, totalData, tolerancy=0.0):
        TP = 0
        for i in range(len(calculated)):
            if ((original[i] <= (calculated[i]+tolerancy)) and (original[i] >= (calculated[i]-tolerancy))):
                TP = TP + 1
        percentage = (TP/totalData)*100
        return np.around(percentage, 4)

    def plotErrors(self, wines, mean_error, repeat_error):
        fig1 = pp.figure("Comparación de Errores")
        fig1.subplots_adjust(hspace=0.5, wspace=0.5)
        self.plot(fig=fig1, wines=wines, error=mean_error, fig_r=2, i=1, titulo='Promedio')
        self.plot(fig=fig1, wines=wines, error=repeat_error, fig_r=2, i=2, titulo='Repetición', color_p='green')
        pp.show()

    def plotError(self, wines, error):
        fig1 = pp.figure("Error relativo")
        fig1.subplots_adjust(hspace=0.5, wspace=0.5)
        self.plot(fig=fig1, wines=wines, error=error, i=1, titulo='K más cercano')
        pp.show()

    def plot(self, wines, error, i, titulo, fig, fig_r=1, fig_c=1, color_p='red'):
        ax = fig.add_subplot(fig_r, fig_c, i)
        ax.plot(wines, error, 'o', color=color_p)
        ax.set_title(titulo)
        ax.set_xlabel("Vino")
        ax.set_ylabel("Error Relativo")
        ax.grid(True)

    def plotBar(self, bars, colors=['blue'], legend=['K más cercano'], tolerancy=0.0, title='Precisión de los algoritmos'):
        fig2 = pp.figure(title)
        for i in range(len(bars)):
            pp.bar(i, bars[i], color=colors[i],  align='center')
        # pp.bar(2, repeat, color='green', align='center')
        pp.title(str(title) + ' con: '+ str(tolerancy) +' de tolerancia')
        pp.legend(legend)
        pp.xlabel("Algoritmos")
        pp.ylabel("Porcentaje de Precisión")
        pp.grid(True)
        pp.show()

    def calcSimilarities(self, new_wine, originals, weight):
        
        for wine in new_wine:
            similarity = []
            for original in originals:
                nearestNeighboorCalc = 0.0
                nearestNeighboorCalc += weight[0] * self.calcAttributeSimilarity(original[0], wine[0], 4.6, 15.9)
                nearestNeighboorCalc += weight[1] * self.calcAttributeSimilarity(original[1], wine[1], 0.12, 1.58)
                nearestNeighboorCalc += weight[2] * self.calcAttributeSimilarity(original[2], wine[2], 0.0, 1.0)
                nearestNeighboorCalc += weight[3] * self.calcAttributeSimilarity(original[3], wine[3], 0.9, 13.9)
                nearestNeighboorCalc += weight[4] * self.calcAttributeSimilarity(original[4], wine[4], 0.012, 0.611)
                nearestNeighboorCalc += weight[5] * self.calcAttributeSimilarity(original[5], wine[5], 1.0, 72.0)
                nearestNeighboorCalc += weight[6] * self.calcAttributeSimilarity(original[6], wine[6], 6.0, 289.0)
                nearestNeighboorCalc += weight[7] * self.calcAttributeSimilarity(original[7], wine[7], 0.99, 1.0)
                nearestNeighboorCalc += weight[8] * self.calcAttributeSimilarity(original[8], wine[8], 2.74, 4.01)
                nearestNeighboorCalc += weight[9] * self.calcAttributeSimilarity(original[9], wine[9], 0.33, 2.0)
                nearestNeighboorCalc += weight[10] * self.calcAttributeSimilarity(original[10], wine[10], 8.4, 14.9)

                
                s = np.around(nearestNeighboorCalc / np.sum(weight), 3)
                similarity.extend(np.array([s]))
        self.similarities = np.array(similarity)
        return self.similarities

    def calcAttributeSimilarity(self, baseCaseAttributeValue, newCaseAttributeValue, 
                                    minValue, maxValue):
        r = (1.0 - np.abs(baseCaseAttributeValue - newCaseAttributeValue) / (maxValue - minValue))
        return r

    def closePlot(self):
        pp.close('all')
        