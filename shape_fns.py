import numpy as np
from numpy.linalg import inv, det

# functions that return shape functions, their derivatives, and the global coordinate

def line2(xi, X):
    
    N = 0.5*np.array([1 - xi, 1 + xi])
    dN = np.array([[-1/2],[1/2]])
    return N, dN

def line3(xi, X):
    N = np.array([xi*(xi-1)/2, 1-xi**2, xi*(xi+1)/2])
    dN = np.array([[xi-0.5],[-2*xi],[xi+0.5]])
    return N, dN

def tri3(xi, X):

    N = np.array([1 - xi[0] - xi[1], xi[0], xi[1]])
    dN = np.array([[-1, -1], [1, 0], [0, 1]])
    return N, dN
    

def quad4(xi, X):

    N = 0.25*np.array([(1 + xi[0])*(1 + xi[1]),
                       (1 - xi[0])*(1 + xi[1]),
                       (1 - xi[0])*(1 - xi[1]),
                       (1 + xi[0])*(1 - xi[1])])
    dN = 0.25*np.array([[ (1 + xi[1]), (1 + xi[0])],
                        [-(1 + xi[1]), (1 - xi[0])],
                        [-(1 - xi[1]),-(1 - xi[0])],
                        [ (1 - xi[1]),-(1 + xi[0])]])
    return N, dN


SHAPE_FNS = {
    'line2': line2,
    'line3': line3,
    'tri3': tri3,
    'quad4': quad4
}

# compute Jacobian matrix
def jacobian(X, dN):

    # compute Jacobian matrix
    J = np.dot(np.transpose(X),dN)
    return inv(J), det(J)

