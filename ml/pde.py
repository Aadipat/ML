# Implementing a LDA classifier (Generative and Parameteric).

# We must model the class distributions as gaussians with the:
#       Same covariance matrix as average variance * I
#       Same covariance matrix if LDA (Average)
#       Different covariance matrix if QDA

import numpy as np
import matplotlib.pyplot as plt


def MLE_Mean(tuples):
    # we compute the unbiased sample mean
    sum = np.zeros(len(tuples[0]))
    for tuple in tuples:
        sum = np.add(sum, tuple)
    return sum/len(tuples)

def average_Sample_Variance(tuples):
    avgvar = 0
    # we take variance of each feature and then average it. 
    arr = np.array(tuples).T
    vars = []
    for variable in arr:
        vars.append(np.var(variable))
    avgvar = np.average(vars)
    return avgvar

def MLE_Covariance_Matrix(feature_data):
    mean = MLE_Mean(feature_data)
    sum = np.zeros((len(feature_data[0]),len(feature_data[0])))
    for tuple in feature_data:
        outer_product = np.outer(np.subtract(tuple,mean), 
                               np.subtract(tuple,mean))
        sum = np.add(sum, outer_product)
    return sum/len(feature_data)

def class_conditional_pdf(x, mean, covariance_matrix):
    power = -0.5*np.dot((x-mean),
                        np.dot(np.linalg.inv(covariance_matrix),np.transpose(x-mean)))
    return np.exp(power)/np.sqrt(np.power(2*np.pi, len(covariance_matrix))*np.linalg.det(covariance_matrix))

class LDA:

    def __init__(self, features, targets):

        self.c1 = features[np.where(targets == 0)]
        self.c2 = features[np.where(targets == 1)]

        self.mean1 = MLE_Mean(self.c1)
        self.cov_m1 = MLE_Covariance_Matrix(self.c1)

        self.mean2 = MLE_Mean(self.c2)
        self.cov_m2 = MLE_Covariance_Matrix(self.c2)

        # need avg Cov matrix for linear decision boundary
        self.cov_m = np.add(self.cov_m1, self.cov_m2)/2

        self.discriminant = lambda x: (len(self.c1)/(len(self.c1)+len(self.c2)))*class_conditional_pdf(x, self.mean1, self.cov_m) - (len(self.c2)/(len(self.c1)+len(self.c2)))*class_conditional_pdf(x, self.mean2, self.cov_m)
    
    def classify(self, x):
        # Classify based on discriminant ln(p(y1|x)) - ln(p(y2|x))
        # print(self.discriminant(x))
        if(self.discriminant(x) > 0):
            return 0
        return 1
    
class QDA:

    def __init__(self, features, targets):

        self.c1 = features[np.where(targets == 0)]
        self.c2 = features[np.where(targets == 1)]

        self.mean1 = MLE_Mean(self.c1)
        self.cov_m1 = MLE_Covariance_Matrix(self.c1)

        self.mean2 = MLE_Mean(self.c2)
        self.cov_m2 = MLE_Covariance_Matrix(self.c2)

        self.discriminant = lambda x: (len(self.c1)/(len(self.c1)+len(self.c2)))*class_conditional_pdf(x, self.mean1, self.cov_m1) - (len(self.c2)/(len(self.c1)+len(self.c2)))*class_conditional_pdf(x, self.mean2, self.cov_m2)

    def classify(self, x):
        # Classify based on discriminant ln(p(y1|x)) - ln(p(y2|x))
        if(self.discriminant(x) > 0):
            return 0
        return 1

class Nearest_Mean:

    def __init__(self, features, targets):

        self.c1 = features[np.where(targets == 0)]
        self.c2 = features[np.where(targets == 1)]

        self.mean1 = MLE_Mean(self.c1)
        self.mean2 = MLE_Mean(self.c2)

        # Taking var * I as cov matrix for nearest mean.
        # self.cov_m1 = average_Sample_Variance(self.c1)*np.identity(len(features[0]))
        # self.cov_m2 = average_Sample_Variance(self.c2)*np.identity(len(features[0]))
        # self.cov_m = np.add(self.cov_m1, self.cov_m2)/2
        # print(self.cov_m)
        # self.discriminant = lambda x: np.log(class_conditional_pdf(x, self.mean1, self.cov_m)) - np.log(class_conditional_pdf(x, self.mean2, self.cov_m)) 
        
        # Better discriminant: Take closer mean
        self.discriminant = lambda x: np.log(np.linalg.norm(np.array(x) - np.array(self.mean1))) - np.log(np.linalg.norm(np.array(x) - np.array(self.mean2)))
    
    def classify(self, x):
        # Classify based on discriminant ln(p(y1|x)) - ln(p(y2|x))
        if(self.discriminant(x) > 0):
            return 0
        return 1



def draw(model, tolerance):
    plt.scatter([p[0] for p in model.c1],[p[1] for p in model.c1], color='blue')
    plt.scatter([p[0] for p in model.c2],[p[1] for p in model.c2], color='red')

    # draw boundary
            # draw boundary
    x1 = np.linspace(-10, 10, 200)
    x2 = np.linspace(-10, 10, 200)

    d_boundaryx1 = []
    d_boundaryx2 = []

    for i in range(len(x1)):
        for j in range(len(x2)):
            if(abs(model.discriminant([x1[i], x2[j]])) < tolerance):
                d_boundaryx1.append(x1[i])
                d_boundaryx2.append(x2[j])

    plt.scatter(d_boundaryx1, d_boundaryx2 , color='black')

    plt.show()


features = np.array([[0,1], [1,1], [1, 0], [2, 1], [8, 8], [7, 6], [9, 10], [6, 9]])
targets = np.array([0,0,0,1,1,1,1,1])

# model = Nearest_Mean(features, targets)

# print(model.classify([5,4]))

# draw(model, 0.0001)


# model = LDA(features, targets)

# print(model.classify([5,4]))

# draw(model, 0.00005)


# model = QDA(features, targets)

# print(model.classify([5,4]))

# draw(model, 0.0001)
