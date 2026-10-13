import numpy as np
import gudhi
import matplotlib.pyplot as plt
from persim import plot_diagrams
import ph_dft

b1 = np.array([[6,9],[7,11]])
b2 = np.array([[0,12],[1,12],[2,14],[3,12],[4,11],[6,4],[7,7]])
b3 = np.array([[0,9],[1,9],[2,7],[3,5],[4,7],[7,7]])
b4 = np.array([[0,7],[1,5],[2,5],[3,3],[4,5],[6,0],[7,5]])
b5 = np.array([[0,2],[0,7],[0,11]])
bars = [b1, b2, b3, b4, b5]

rips = ph_dft.calc_rips(bars,t=8,p=24,maxdim=1)
#plot_diagrams(rips['dgms'], show = True)
gudhi.plot_persistence_diagram(rips['dgms'])
plt.show()
gudhi.plot_persistence_barcode(rips['dgms'])
plt.show()
