__author__ = 'markov'

import numpy as np
import sys
import PoissonSystem as ps
reload(ps)
import cProfile
import Numerical_test as Numt
import Numerical_test_Poi as NumtP

# rxn1 0 -> M
# rxn2 M -> M + P
# rxn3 M -> 0
# rxn4 M+ P -> M
# rxn5 P  -> 0
# rxn6 2P -> D
# rxn7 D  -> 0
# rxns [2, 10, 0.6, 0.2, 0, 0.1, 1]

small_number = 0.1
verysmall_number = 1e-10
Nx = 500
Ny = 2000
n_iter = 1000
delta0 = 0.1
observed = np.array([[0,1],[0,2]])
alpha = 1
rxn_kinetics=[[[1., 0., 0.],  [[0.], [0.]]      ],
          [ [0., 1., 0.],  [[0.], [1.]]    ],
          [ [-1., 0., 0.], [[0.], [1.]]      ],
          [ [0., -1., 0.], [[0., 1.],[1., 1.]]   ],
          [ [0., -1., 0],  [[1.], [1.]]     ],
          [ [0., -2., 1.], [[1.], [2.]]       ],
          [ [0., 0., -1.], [[2.], [1.]]]    ]


theta0  = np.array([2, 10, 0.6, 0.2, verysmall_number, 0.1, 1.])
sigma0  = np.array([0.1, 0.2, 0.2])
snaptimes =np.array([0,3, 6, 9] )
initsnap = np.array([[0,0,0]] * Nx)

psystem = ps.PoissonSystem(kinetics = rxn_kinetics, theta = theta0, sigma = sigma0)
ntp= NumtP.Numerical_test_Poi(0.5, psystem.theta)

#theta_init = np.array([small_number, small_number ,small_number,small_number,small_number, small_number, small_number])
theta_init = np.abs(np.random.normal(0, size = 7))

snapshots = psystem.make_snapshots(snaptimes, initsnap, observed,  delta = delta0,  nx = Ny)
ntp.em_algorithm(n_iter, Nx, snapshots, delta0, theta_init, observed, initsnap, psystem, write = False, myalpha = alpha)