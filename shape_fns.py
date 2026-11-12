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

def legendre(xi, p):
    """Evaluate the p-th Legendre polynomial at master coordinate xi in [-1, 1]."""
    if p == 0:
        return 1.0
    elif p == 1:
        return float(xi)
    else:
        return ((2 * p - 1) * xi * legendre(xi, p - 1) + (1 - p) * legendre(xi, p - 2)) / p


def hierarchical(xi, X, p=1):
    """1D hierarchical p-FEM shape functions of polynomial order p on [-1, 1]."""
    assert p >= 1, "Polynomial order p must be at least 1."

    phi = [0.5 * (1.0 - xi), 0.5 * (1.0 + xi)]
    dphi = [-0.5, 0.5]

    if p >= 2:
        for j in range(2, p + 1):
            phi_j = (legendre(xi, j) - legendre(xi, j - 2)) / np.sqrt(4 * j - 2)
            dphi_j = np.sqrt((2 * j - 1) / 2.0) * legendre(xi, j - 1)
            phi.append(phi_j)
            dphi.append(dphi_j)

    N = np.array(phi, dtype=float)
    dN = np.array(dphi, dtype=float).reshape(-1, 1)
    return N, dN


SHAPE_FNS = {
    'line2': line2,
    'line3': line3,
    'tri3': tri3,
    'quad4': quad4,
    'linep': hierarchical
}

# compute Jacobian matrix
def jacobian(X, dN):

    # compute Jacobian matrix
    J = np.dot(np.transpose(X),dN)
    return inv(J), det(J)

