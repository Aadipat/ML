import numpy as np
import matplotlib.pyplot as plt

def MLE_Mean(tuples):
    # we compute the unbiased sample mean
    sum = np.zeros(len(tuples[0]))
    for tuple in tuples:
        sum = np.add(sum, tuple)
    return sum/len(tuples)

def Gaussian_Kernel(x, mean, h): 
    x = np.divide(np.subtract(x,mean),h)
    covariance_matrix = np.identity(len(x))
    power = -0.5*np.dot((x-mean),
                        np.dot(np.linalg.inv(covariance_matrix),np.transpose(x-mean)))
    return np.exp(power)/np.sqrt(np.power(2*np.pi, len(covariance_matrix))*np.linalg.det(covariance_matrix))

def Box_kernel(x, mean, h):
    if(np.sqrt(np.inner(np.subtract(mean,x),np.subtract(mean,x))) <= h): 
        return 1
    return 0.01

class ParzenKDE():
    def __init__(self, kernel, features, targets, h):
        # kernal function (x,mean) => p(x|mean)
        self.h = h
        self.kernel = kernel
        self.features = features
        self.targets = targets
        self.c1 = features[np.where(targets == 0)]
        self.c2 = features[np.where(targets == 1)]
        self.mean1 = MLE_Mean(self.c1)
        self.mean2 = MLE_Mean(self.c2)
        self.discriminant = lambda x: (len(self.c1)/(len(self.c1)+len(self.c2)))*(self.p(x, self.kernel,self.c1,self.mean1,self.h)) - (len(self.c2)/(len(self.c1)+len(self.c2)))*self.p(x, self.kernel,self.c2,self.mean2,self.h)
    def p(self,x, kernel, c, mean, h):
            p = 0
            for f in c:
                p += kernel(x, mean, h)
            return p/(len(c))
    def classify(self, x):
        # Classify based on discriminant ln(p(y1|x)) - ln(p(y2|x))
        if(self.discriminant(x) > 0):
            return 0
        return 1

def euclidean_distance(x,y):
    return np.linalg.norm(np.array(x) - np.array(y))

class kNN():
    # Assume local same model for k nearest neighbours.  
    def __init__(self, k, features, targets, distance_f):
        self.k = k
        self.features = features
        self.targets = targets
        self.c1 = features[np.where(targets == 0)]
        self.c2 = features[np.where(targets == 1)]
        self.distance_f = distance_f
        self.data = np.array([np.append(self.features[f],self.targets[f]) for f in range(len(self.features))])
        return
    def discriminant(self,x):
        data = [np.append(f,np.array([self.distance_f(x,f)])) for f in self.features]
        data = [np.append(data[d],self.targets[d]) for d in range(len(self.data))]
        data.sort(key=lambda x: x[-2])
        count0 = 0
        count1 = 0
        for i in range(self.k):
            if(np.isclose(data[i][-1],0.0)):
                count0 += 1
            if(np.isclose(data[i][-1],1.0)):
                count1 += 1
        return count0/(count0 + count1) - 0.5
    def p(self, x, c):
        data = [np.append(f,np.array([self.distance_f(x,f)])) for f in self.features]
        data = [np.append(data[d],self.targets[d]) for d in range(len(self.data))]
        data.sort(key=lambda x: x[-2])
        ki = 0
        for i in range(self.k):
            if(np.isclose(data[i][-1],c)):
                ki += 1
        if(c == 1):
            return ki/(len(self.c1)*np.pi*np.power(np.linalg.norm(euclidean_distance(data[self.k-1][0:-2],x)),3))
        return ki/(len(self.c2)*np.pi*np.power(np.linalg.norm(euclidean_distance(data[self.k-1][0:-2],x)),3))
    def posterior(self, x, c):
        data = [np.append(f,np.array([self.distance_f(x,f)])) for f in self.features]
        data = [np.append(data[d],self.targets[d]) for d in range(len(self.data))]
        data.sort(key=lambda x: x[-2])
        ki = 0
        for i in range(self.k):
            if(np.isclose(data[i][-1],c)):
                ki += 1
        if(c == 1):
            return ki/((len(self.c1) + len(self.c2))*np.pi*np.power(np.linalg.norm(euclidean_distance(data[self.k-1][0:-2],x)),3))
        return ki/((len(self.c1) + len(self.c2))*np.pi*np.power(np.linalg.norm(euclidean_distance(data[self.k-1][0:-2],x)),3))
    def classify(self, x):
        # Sort data by distance to x
        # look at nearest k and give majority vote.
        if(self.discriminant(x) > 0):
            return 0
        return 1

class Naive_Bayes():
    # We assume conditional independence between features .
    def __init__(self, model_cc):
        self.features = features
        self.targets = targets
        self.model_cc = model_cc
        self.c1 = features[np.where(targets == 0)]
        self.c2 = features[np.where(targets == 1)]
        return
    def p(self, x):
        
        p = 1
        for i in range(len(self.features[0])):
            p *= 1

        return p
    def classify(self):
        return
    

def draw(model, tolerance):
    plt.scatter([p[0] for p in model.c1],[p[1] for p in model.c1], color='blue')
    plt.scatter([p[0] for p in model.c2],[p[1] for p in model.c2], color='red')

    # draw boundary
            # draw boundary
    x1 = np.linspace(-10, 10, 10)
    x2 = np.linspace(-10, 10, 10)

    d_boundaryx1 = []
    d_boundaryx2 = []

    for i in range(len(x1)):
        for j in range(len(x2)):
            # print(model.discriminant([x1[i], x2[j]]))
            if(abs(model.discriminant([x1[i], x2[j]])) < tolerance):
                d_boundaryx1.append(x1[i])
                d_boundaryx2.append(x2[j])

    plt.scatter(d_boundaryx1, d_boundaryx2 , color='black')

    plt.show()



features = np.array([[0,1], [1,1], [1, 0], [2, 1], [8, 8], [7, 6], [9, 10], [6, 9]])
targets = np.array([0,0,0,1,1,1,1,1])

# model = ParzenKDE(Gaussian_Kernel,features, targets, 1)
# print(model.classify([0,0]))
# draw(model, 0.000000000001)

model = kNN(3,features, targets, euclidean_distance)
print(model.classify([0,0]))

# draw(model, 0.2)