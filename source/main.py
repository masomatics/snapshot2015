alpha = 0.5
theta_approx = np.array([-1.5, 0, 0, 0.1, 0.2])
for iter in range(0,100):
        print theta_approx
        dsystem_old = Discrete_Doucet_system.Discrete_Doucet_system(theta = theta_approx)
        xdat_test, pdat_test= simul.simulate(dsystem_old, Nx_test,  seed = iter*2)
        px = Nx_test * dsystem.compare(xdat_test, xobs)
        xdat_new, pdat_new, A_new, B_new = simul.simulate(dsystem_old, Nx_test,  seed = iter*2, Px = px, stat= True)
        xdat_old, pdat_old, A_old, B_old = simul.simulate(dsystem_old, Nx_test,  seed = iter*2+1, stat= True)
        theta_approx = np.array(np.linalg.inv(A_new + alpha*A_old) * np.matrix(B_new + alpha*B_old).transpose())
        theta_approx = np.array(theta_approx.transpose().tolist()[0] + [0.2])
