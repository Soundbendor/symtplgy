import numpy as np


b1 = np.array([[6,9],[7,11]])
b2 = np.array([[0,12],[1,12],[2,14],[3,12],[4,11],[6,4],[7,7]])
b3 = np.array([[0,9],[1,9],[2,7],[3,5],[4,7],[7,7]])
b4 = np.array([[0,7],[1,5],[2,5],[3,3],[4,5],[6,0],[7,5]])
b5 = np.array([[0,2],[0,7],[0,11]])

#cexp = lambda x,t,p: np.exp(np.multiply(-2j*np.pi, x/[t,p]))
# expects arrays of shape (num_pairs, 2) so x[:,0] gets all time steps
# returns (num_pairs, 2)

b_arr = [cexp(x,8,24) for x in [b1,b2,b3,b4,b5]]
