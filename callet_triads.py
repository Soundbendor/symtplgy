import musicnoteconv as mnc
import numpy as np
import gudhi
import matplotlib.pyplot as plt
from persim import plot_diagrams
import ph_dft

subdiv = 1
p = 12

b = [
        ['c4', 'e4', 'g4'],
        ['c4', 'ef4', 'g4'],
        ['ef4', 'g4', 'bf4'],
        ['ef4', 'gf4', 'bf4'],
        ['cs4', 'fs4', 'bf4'],
        ['cs4', 'fs4', 'a4'],
        ['cs4', 'e4', 'a4'],
        ['c4', 'e4', 'a4'],
        ['e4', 'gs4', 'b4'],
        ['e4', 'g4', 'b4'],
        ['d4', 'g4', 'b4'],
        ['d4', 'g4', 'bf4'],
        ['d4', 'f4', 'bf4'],
        ['cs4', 'f4', 'bf4'],
        ['cs4', 'f4', 'af4'],
        ['cs4', 'e4', 'gs4'],
        ['c4', 'ef4', 'gs4'],
        ['ef4', 'gs4', 'b4'],
        ['ds4', 'fs4', 'b4'],
        ['d4', 'fs4', 'b4'],
        ['d4', 'fs4', 'a4'],
        ['d4', 'f4', 'a4'],
        ['c4', 'f4', 'a4'],
        ['c4', 'f4', 'af4']
        ]


b_pc = [[np.array(mnc.note_to_midi(x) % 12) for x in y] for y in b]
b_pairs = [np.vstack((np.zeros(3), x)).T for x in b_pc]

rips = ph_dft.calc_rips(b_pairs, t=subdiv, p=p)
#plot_diagrams(rips['dgms'], show = True)
gudhi.plot_persistence_diagram(rips['dgms'])
plt.show()
gudhi.plot_persistence_barcode(rips['dgms'])
plt.show()
