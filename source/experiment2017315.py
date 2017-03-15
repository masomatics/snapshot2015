
import execute_variance_test_normal as ex
import numpy as np

dimdats = np.array([2,5,10,15,20, 50,100])
for dim in dimdats:
    print str(dim) + "dimension in progress"
    ex.run_variance_test(dim, N = 50,M = 100, numexp = 1000,  loc1 = 0., loc2 = 0., numticks = 100, myattention_index = [], histbins = 20
    )

print "session1 complete"

for dim in dimdats:
    print str(dim) + "dimension in progress"
    ex.run_variance_test(dim, N = 50,M = 100, numexp = 1000,  loc1 = 1, loc2 = 0., numticks = 100, myattention_index = [], histbins = 20
    )

print "session2 complete"


for dim in dimdats:
    print str(dim) + "dimension in progress"
    ex.run_variance_test(dim, N = 50,M = 100, numexp = 1000,  loc1 = 1, loc2 = 0., numticks = 100, myattention_index = [0,1], histbins = 20
    )
