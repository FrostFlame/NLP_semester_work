import scipy
from scipy import sparse
from numpy import hstack

x = [[1, 2], [3, 4], [5, 6]]
y = [[9, 8, 7], [11, 12, 13], [20, 21, 22]]

un = hstack((x, y))
print(un)