import numpy as np
import gudhi
import matplotlib.pyplot as plt
from persim import plot_diagrams
import ph_dft

subdiv = 132
p0 = 0
p1 = 0

def make_polyrhythms(low, hi):
    bars = []
    for x in range(low, hi+1):
        for y in range(x+1, hi+1):
            if np.gcd(x,y) == 1:
                times_x = np.arange(x) * int(subdiv/x)
                times_y = np.arange(y) * int(subdiv/y)
                pairs_x = np.vstack((times_x, np.repeat(p0, x))).T
                pairs_y = np.vstack((times_y, np.repeat(p1, y))).T
                bar_xy = np.vstack((pairs_x, pairs_y))
                bars.append(bar_xy)
    return bars

rips = ph_dft.calc_rips(make_polyrhythms(2,12), t=subdiv, p=2)
#plot_diagrams(rips['dgms'], show = True)
gudhi.plot_persistence_diagram(rips['dgms'])
plt.show()
gudhi.plot_persistence_barcode(rips['dgms'])
plt.show()
