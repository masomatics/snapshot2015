
import execute_variance_test_normal as ex
import numpy as np
import util_Snap as util

#dimdats = np.array([2,5,10,15,20, 50,100])

#dimdats = np.array([2,5,10,15,20])
dimdats = np.array([10])
deltamus = np.array([])
fixed_deltamu = 10.
myN = 50
myM = 100
my_numexp = 1000
my_numticks = 100
my_datalocation = '../records'


for dim in dimdats:

    mydeltamu = util.vector_norm(np.ones([dim, 1]))
    print "deltamu is" + str(mydeltamu)
    print str(dim) + "dimension in progress"
    ex.run_variance_test(dim, N = myN,M = myM, numexp = my_numexp,  deltamu=mydeltamu, numticks = my_numticks, myattention_index = [], histbins = 20
    )


for dim in dimdats:

    mydeltamu = fixed_deltamu
    print "deltamu is" + str(mydeltamu)
    print str(dim) + "dimension in progress"
    ex.run_variance_test(dim, N = myN,M = myM, numexp = my_numexp,  deltamu=mydeltamu, numticks = my_numticks, myattention_index = [], histbins = 20
    )
