import sys, os, json, inspect
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from gauss import quadrature
from elastic import constitutive
from shape_fns import SHAPE_FNS, jacobian

DEFAULT_ELEM_TYPE = {
    (1, 2): 'line2',
    (1, 3): 'line3',
    (2, 3): 'tri3',
    (2, 4): 'quad4',
}

# read data from properly formatted json-formatted file
try:
    filename = sys.argv[1]
    with open(filename) as f:

        print("-- Reading file '{}'".format(filename))
        data = json.load(f)                     # load file
        nodes = np.array(data['nodes'])         # nodes array to numpy array
        elems = np.array(data['elements'])      # elements array to numpy array
        material = tuple(data['material'])      # material data
        bcs = data['boundary']                  # prescribed boundary conditions
        load = data['load']                     # prescribed loads
        nnodes, dpn = nodes.shape               # node count and dofs per node
        nelems, etype = elems.shape             # element count and element type
        elem_type = data['element']
        shape_fn = SHAPE_FNS[elem_type]
        dofs = dpn*nnodes                       # total number of dofs
        gauss = quadrature(2,data['gauss']) # quadrature data structure
        a_param, b_param = data.get('params', [1.0, 0.5]) # optional parameters for body force

except IndexError:
    print('Usage: python efem.py <filename>')
    quit()

except KeyError as err:
    frame = inspect.currentframe()
    print('{}:{}: error: Key {} not found in input data file'.format(__file__,frame.f_lineno,err))
    quit()


def body_force(x, a, xb):
    b = x - xb
    return 2 * (a + a**3 * b * (b - x +1))/(a**2*b**2 + 1)**2

# create data structures
K = sp.lil_matrix((dofs, dofs))
F = np.zeros(dofs)

# assemble stiffness matrix
print("-- Assembling stiffness matrix...")

# loop over elements to assemble matrices
for e, conn in enumerate(elems):

    # coordinate array for the element
    X = nodes[conn]     # we get the global coordinates of the nodes for the current element
    ldofs = dpn*len(conn)  # number of local degrees of freedom for the element
    k = np.zeros((ldofs, ldofs)) # local stiffness matrix 
    f = np.zeros(ldofs)          # local force vector 

    # get element degree of freedom array
    eft = np.array([dpn * n + i for n in conn for i in range(dpn)]) 

    # loop over gauss points
    for i, xi in enumerate(gauss.xi):

        # compute global coordinate, shape functions and their derivatives
        N, dN = shape_fn(xi, X)

        # compute jacobian
        Jinv, j = jacobian(X, dN)

        # shape functions derivatives with respect to global coordinates
        B = np.dot(dN, Jinv)

        # expand matrix to take into account multiple dofs per node
        BB = np.kron(B.T, np.identity(dpn))

        # compute material matrix
        matDT = constitutive(material, dpn)

        k += gauss.wgt[i] * j * np.dot(np.dot(BB.T, matDT), BB) 

        x = np.dot(N,X)[0] # isoparametric mapping - we get the location of the gauss point 
        f += gauss.wgt[i] * j * N * body_force(x, a_param, b_param) # local force vector contribution from the body force

    # add contribution to global stiffness matrix
    K[eft[:, np.newaxis], eft] += k
    F[eft] += f

K_orig = K.copy() # original stiffness matrix for computing U

F[load[0]] += load[1]

# apply boundary conditions
print("-- Applying boundary conditions...")

zero = bcs[0]                    # array of rows/columns which are to be zeroed out
F -= K[:, zero] * bcs[1]         # modify right hand side with prescribed values
K[:,zero] = 0; K[zero,:] = 0;    # zero-out rows/columns
K[zero,zero] = 1                 # add 1 in the diagional
F[zero] = bcs[1]                 # prescribed values

# apply loads



# solve system of equations
print("-- Solving system of equations...")
U = spsolve(K.tocsr(), F)

# output file to vtk legacy (easier) format
filepath,_ = os.path.splitext(filename)
Uh = 0.5 * (U @ (K_orig @ U)) # compute strain energy

print(f"-- Strain energy: {Uh}")
print("Exact strain energy: 0.1397797556255038")

vtk = {'line2': 3, 'line3': 21}  # map etype to vtk element type

# modify data structures since vtk requires 3 components
nodes = np.hstack([nodes, np.zeros([len(nodes), 3 - dpn])]) if dpn != 3 else nodes
U = np.reshape(U, (nnodes, dpn))
U = np.hstack([U, np.zeros([len(U), 3 - dpn])]) if dpn != 3 else U

outfile = filepath + ".vtk"
with open(outfile, 'w') as f:

    print("-- Writing file '{}'".format(outfile))
    print('# vtk DataFile Version 2.0', file=f)
    print('Results for', filename, file=f)
    print('ASCII', file=f)
    print('DATASET UNSTRUCTURED_GRID', file=f)
    print('\nPOINTS {} float'.format(nnodes), file=f)
    print('{}'.format('\n'.join(str(i).strip('[],') for i in nodes)), file=f)
    print('\nCELLS {} {}'.format(nelems, nelems + np.prod(elems.shape)), file=f)
    print('{}'.format('\n'.join(str(etype) + " " + str(i).strip('[],') for i in elems)), file=f)
    print('\nCELL_TYPES', nelems, file=f)
    print('{}'.format('\n'.join(str(vtk[elem_type]) for i in range(nelems))), file=f)
    print('\nPOINT_DATA', nnodes, file=f)
    print('VECTORS displacement float', file=f)
    print('{}'.format('\n'.join(str(i).strip('[],') for i in U)), file=f)
