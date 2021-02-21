import numpy as np

###################################################
# Modeling: what we want to compute

# points = [(np.array([2]), 4), (np.array([4]), 2)]
# d = 1

# Generate artificial data
true_w = np.array([1, 2, 3, 4, 5])
d = len(true_w)
points = []

for i in range(500000):
    x = np.random.randn(d)
    y = true_w.dot(x) + np.random.randn()
    # print(x, y)
    points.append((x, y))


def sF(w, i):
    x, y = points[i]
    return (w.dot(x) - y) ** 2

def sdF(w, i):
    x, y = points[i]
    return 2 * (w.dot(x) - y) * x  

#####################################################
# Algorithms: how we compute it

def stochasticGradientDescent(sF, sdF, d, n):
    #Stochastic gradient descent

    numUpdates = 0
    w = np.zeros(d) 
    eta = 0.01
   
    for t in range(1000):
        for i in range(n):
            value = sF(w, i)
            gradient = sdF(w, i)
            numUpdates += 1
            eta = 1.0 / numUpdates 
            w = w - eta * gradient    
        
        print("Iteration {}: w = {}, F(w) = {}".format(t, w, value))

stochasticGradientDescent(sF, sdF, d, len(points))
