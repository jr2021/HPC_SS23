import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
from tqdm import tqdm

class Simulation:
    '''
    Simulation class for Lattice Boltzmann solver
    '''
    
    def __init__(self, nx=101, ny=101, scenario='explosion', epsilon=0.5, omega=0.05, T=500):
        '''
        Takes in simulation parameters and assigns to class attributes
        '''

        # 9x2 grid of velocity channels
        self.c_92 = np.array([
            [0,  0], 
            [0,  1], 
            [1,  0], 
            [0, -1], 
            [-1, 0], 
            [1,  1], 
            [-1, -1], 
            [1,  -1], 
            [-1, 1]] 
        )
        
        # 1x9 grid of velocity channel weights
        self.w_9 = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
        self.nx, self.ny = nx, ny # grid dimensions
        self.scenario = scenario # scenario keyword (explosion, shear_wave_density, ...)
        self.omega = omega # collision factor
        self.epsilon = epsilon # perturbation factor
        self.T = T # time steps
    

    def init_parallel(self):
        '''
        Set the initial conditions for the parallelized simulation
        '''

        
        self.comm = MPI.COMM_WORLD
        self.n_sub = self.comm.Get_size() # number of subdomains
        self.rank = self.comm.Get_rank() # rank of subprocess

        if self.nx < self.ny: # the grid is rectangular
            self.nsubx = int(np.floor(np.sqrt(self.n_sub*self.nx/self.nx)))
            self.nsuby = int(np.floor(self.n_sub/self.nsubx))
        elif self.nx > self.ny: # the grid is rectangular
            self.nsuby = int(np.floor(np.sqrt(self.n_sub*self.ny/self.nx)))
            self.nsubx = int(np.floor(self.n_sub/self.nsuby))
        else: # the grid is square
            self.nsuby = int(np.floor(np.sqrt(self.n_sub)))
            self.nsubx = int(self.n_sub/self.nsuby)

        # create cartesian communicator 
        self.cartcomm = self.comm.Create_cart(dims=[self.nsubx, self.nsuby], periods=[True,True], reorder=False)
        self.rcoords = self.cartcomm.Get_coords(self.rank) # get coordinate points of subdomain

        # define send and recieve directions
        self.sR, self.dR = self.cartcomm.Shift(1, 1)
        self.sL, self.dL = self.cartcomm.Shift(1, -1)
        self.sU, self.dU = self.cartcomm.Shift(0, -1)
        self.sD, self.dD = self.cartcomm.Shift(0, 1)
        self.sd = np.array([self.sR, self.dR, self.sL, self.dL, self.sU, self.dU, self.sD, self.dD], dtype=int)


    def initial_conditions(self):
        '''
        Returns the initial conditions on velocity and density for a simulation scenario
        '''
        
        if self.scenario == 'explosion':
            # uniform density
            rho_yx = np.ones((self.nx, self.ny), dtype=float) / (self.nx * self.ny)
            
            # zero velocity
            u_2yx = np.zeros((2, self.nx, self.ny), dtype=float)
            
            # add small amount of density to center of grid
            rho_yx[self.nx//2, self.ny//2] *= 1 + self.epsilon

        elif self.scenario == 'shear_wave_density':
            # uniform density
            rho_yx = np.ones((self.nx, self.ny), dtype=float)
            
            # zero velocity
            u_2yx = np.zeros((2, self.nx, self.ny), dtype=float)
            
            # create meshgrid
            x, y = np.arange(self.nx), np.arange(self.ny)
            X, Y = np.meshgrid(x, y)
            
            # add sinusoidal density in x-direction
            rho_yx += self.epsilon * np.sin(2*np.pi*X.T/self.nx)

        elif self.scenario == 'shear_wave_velocity':
            # uniform density
            rho_yx = np.ones((self.ny, self.nx), dtype=float) / (self.nx * self.ny)
            
            # zero velocity
            u_2yx = np.zeros((2, self.ny, self.nx), dtype=float)
            
            # create meshgrid
            x, y = np.arange(self.nx), np.arange(self.ny)
            X, Y = np.meshgrid(x, y)
            
            # add sinusoidal velocity in x direction
            u_2yx[1] += self.epsilon * np.sin(2*np.pi*Y/self.ny)

        elif self.scenario in ('couette_flow', 'poiseuille_flow', 'sliding_lid', 'sliding_lid_parallel'):
            # zero velocity
            u_2yx = np.zeros((2, self.ny, self.nx), dtype=float)

            # uniform density
            rho_yx = np.ones((self.ny, self.nx), dtype=float)

        return u_2yx, rho_yx
    
    ################ Simulation Code ################        
    
    def stream_inplace(self, f_9yx):
        '''
        Stream the probability density function inplace
        '''
        
        # iterate over channels
        for i in range(0, 9):
            
            # roll probability density function in direction of ith channel
            f_9yx[i] = np.roll(f_9yx[i], shift=self.c_92[i], axis=(1, 0))

    
    def stream(self, f_9yx) -> np.array:
        '''
        Return the streamed probability density function
        '''

        f_streamed_9yx = np.zeros(f_9yx.shape) # to be returned
        for i in range(0, 9): # iterate over the channels
            
            # roll probability density function in direction of ith channel
            f_streamed_9yx[i] = np.roll(f_9yx[i], shift=self.c_92[i], axis=(0, 1))
            
        return f_streamed_9yx
    
            
    def density(self, f_9yx) -> np.array:
        '''
        Returns the overall density over all velocity channels
        '''
        
        # sums probability density function over channels
        return np.sum(f_9yx, axis=0)


    def velocity(self, f_9yx) -> np.array:
        '''
        Returns average velocity over all velocity channels
        '''
        
        # sums probability density function over channels and weighs by discretized velocity
        sum_2yx = np.einsum('ia,ijk->ajk', self.c_92, f_9yx)

        rho_yx = self.density(f_9yx) # gets density 

        return sum_2yx / rho_yx


    def equilibrium_function(self, u_2yx, rho_yx) -> np.array:
        '''
        Returns the equilibrium function
        '''
        
        # sums average velocities over x and y directions
        squ_yx = np.einsum('ajk,ajk->jk', u_2yx, u_2yx)
        # squ_yx = (u_2yx[0] + u_2yx[1]) ** 2
        
        # initializes equilibrium probability density function
        f_eq_9yx = np.zeros((9, rho_yx.shape[0], rho_yx.shape[1]))
        
        # cu_yx = np.einsum('ia,ajk->jk', self.c_92, u_2yx)
        
        # iterate over channels
        for i in range(0, 9):
            
            # sums average velocities over x and y directions and weighs by ith velocity channel
            cu_yx = np.einsum('a,ajk->jk', self.c_92[i], u_2yx)
            # cu_yx = self.c_92[i][0] * u_2yx[0] + self.c_92[i][1] * u_2yx[1]
            
            # equation X
            f_eq_9yx[i] = self.w_9[i] * rho_yx * (1.0 + 
                                                  3.0 * cu_yx + 
                                                  9.0/2.0 * cu_yx**2 -
                                                  3.0/2.0 * squ_yx)

        return f_eq_9yx


    def moving_wall(self, f_9yx, rho_yx, c_idx, wall_idx=0, s=1/np.sqrt(3), u_wall_2=[0, 0.05]) -> np.array:
        '''
        Performs the moving wall boundary condition at a given wall
        '''

        return f_9yx[c_idx, wall_idx, :] - 2 * self.w_9[c_idx] * rho_yx[0] * np.dot(self.c_92[c_idx], u_wall_2) / s**2
    

    def pressure_variation(self, f_9yx, u_2yx, rho_yx, c_idx, p_in=0.1, p_out=0.01, s=1/np.sqrt(3)) -> np.array:
        '''
        Performs the pressure variation boundary condition at the side walls
        '''

        # change in density
        p_delta = p_out - p_in
        
        # density coming in the left wall
        rho_in_yx = np.ones((self.ny, self.nx)) * (p_out+p_delta)/s**2

        # density going out the right wall
        rho_out_yx = np.ones((self.ny, self.nx)) * p_out/s**2

        # velocity coming in the left wall
        u_in_2yx = np.repeat(u_2yx[:, :, -2], self.nx).reshape((2, self.ny, self.nx))

        # velocity going out the right wall
        u_out_2yx = np.repeat(u_2yx[:, :, 1], self.nx).reshape((2, self.ny, self.nx))

        # gets the equilibrium probability density function
        f_eq_9yx = self.equilibrium_function(u_2yx, rho_yx) 

        # gets the eq. prob. dens. func. coming in the left wall 
        f_eq_in_9yx = self.equilibrium_function(u_in_2yx, rho_in_yx)

        # gets the eq. prob. dens. func. going out the right wall
        f_eq_out_9yx = self.equilibrium_function(u_out_2yx, rho_out_yx)

        return f_eq_in_9yx[c_idx, :, -2] + (f_9yx[c_idx, :, -2] - f_eq_9yx[c_idx, :, -2]), f_eq_out_9yx[c_idx, :, 1] + (f_9yx[c_idx, :, 1] - f_eq_9yx[c_idx, :, 1])


    def bounce_back_upper(self, f_9yx, wall_idx=0) -> np.array:
        '''
        Performs the bounce-back boundary condition at the top wall
        '''

        f_9yx[8, wall_idx, :] = f_9yx[6, wall_idx, :]
        f_9yx[1, wall_idx, :] = f_9yx[3, wall_idx, :]
        f_9yx[5, wall_idx, :] = f_9yx[7, wall_idx, :]

        return f_9yx

    
    def bounce_back_right(self, f_9yx, wall_idx=-1) -> np.array:
        '''
        Performs the bounce-back boundary condition at the right wall
        '''

        f_9yx[5, :, wall_idx] = f_9yx[7, :, wall_idx]
        f_9yx[2, :, wall_idx] = f_9yx[4, :, wall_idx]
        f_9yx[6, :, wall_idx] = f_9yx[8, :, wall_idx]

        return f_9yx


    def bounce_back_left(self, f_9yx, wall_idx=0) -> np.array:
        '''
        Performs the bounce-back boundary condition at the left wall
        '''

        f_9yx[7, :, wall_idx] = f_9yx[5, :, wall_idx]
        f_9yx[4, :, wall_idx] = f_9yx[2, :, wall_idx]
        f_9yx[8, :, wall_idx] = f_9yx[6, :, wall_idx]

        return f_9yx


    def bounce_back_lower(self, f_9yx, wall_idx=-1) -> np.array:
        '''
        Performs the bounce-back boundary condition at the bottom wall
        '''

        f_9yx[6, wall_idx, :] = f_9yx[8, wall_idx, :]
        f_9yx[3, wall_idx, :] = f_9yx[1, wall_idx, :]
        f_9yx[7, wall_idx, :] = f_9yx[5, wall_idx, :]

        return f_9yx


    def moving_wall_top(self, f_9yx, rho_yx, wall_idx=0) -> np.array:
        '''
        Performs the moving-wall boundary conditino at the top wall
        '''

        f_9yx[8, wall_idx, :] = self.moving_wall(f_9yx, rho_yx, c_idx=6, wall_idx=wall_idx)
        f_9yx[1, wall_idx, :] = self.moving_wall(f_9yx, rho_yx, c_idx=3, wall_idx=wall_idx)
        f_9yx[5, wall_idx, :] = self.moving_wall(f_9yx, rho_yx, c_idx=7, wall_idx=wall_idx)

        return f_9yx
    

    def pressure_variation_sides(self, f_9yx, u_2yx, rho_yx) -> np.array:
        '''
        Performs the pressure variation boundary-condition at the side walls
        '''

        f_9yx[7, :, 0], f_9yx[7, :, -1] = self.pressure_variation(f_9yx, u_2yx, rho_yx, c_idx=7)
        f_9yx[4, :, 0], f_9yx[4, :, -1] = self.pressure_variation(f_9yx, u_2yx, rho_yx, c_idx=4)
        f_9yx[8, :, 0], f_9yx[8, :, -1] = self.pressure_variation(f_9yx, u_2yx, rho_yx, c_idx=8)

        return f_9yx


    def boundary_condition(self, f_9yx, u_2yx, rho_yx) -> np.array:
        '''
        Performs a certain boundary condition based on the given scenario
        '''

        if self.scenario == 'couette_flow':
            
            # bounce-back boundary condition on lower wall
            f_9yx = self.bounce_back_lower(f_9yx)

            # moving wall boundary condition on upper wall
            f_9yx = self.moving_wall_top(f_9yx, rho_yx)
    
            return f_9yx
        
        elif self.scenario == 'poiseuille_flow':

            # bounce-back boundary condition on upper and lower wall
            f_9yx = self.bounce_back_lower(f_9yx)
            f_9yx = self.bounce_back_upper(f_9yx)

            # pressure variation boundary condition on side walls
            f_9yx = self.pressure_variation_sides(f_9yx, u_2yx, rho_yx)

            return f_9yx
        
        elif self.scenario == 'sliding_lid':

            # bounce-back boundary condition on lower and side walls
            f_9yx = self.bounce_back_lower(f_9yx)
            f_9yx = self.bounce_back_right(f_9yx)
            f_9yx = self.bounce_back_left(f_9yx)

            # moving wall boundary condition on upper wall
            f_9yx = self.moving_wall_top(f_9yx, rho_yx)

            return f_9yx
        
        elif self.scenario == 'sliding_lid_parallel':
            rcoords = np.empty((self.n_sub, 2), dtype=int)
            self.comm.Allgather(np.array(self.rcoords, dtype=int), rcoords)

            if self.rcoords[1] == np.max(rcoords[:, 1]):
                f_9yx = self.bounce_back_right(f_9yx, wall_idx=-2)
            if self.rcoords[1] == 0:
                f_9yx = self.bounce_back_left(f_9yx, wall_idx=1)
            if self.rcoords[0] == np.max(rcoords[:, 0]):
                f_9yx = self.bounce_back_lower(f_9yx, wall_idx=-2)
            if self.rcoords[0] == 0:
                f_9yx = self.moving_wall_top(f_9yx, rho_yx, wall_idx=1)

            return f_9yx

        else:
            return f_9yx


    def run(self):
        '''
        Main loop for Lattice Boltzmann solver
        '''

        u_2yx, rho_yx = self.initial_conditions() # gets initial conditions on average velocity and overall density
        f_9yx = self.equilibrium_function(u_2yx, rho_yx) # gets equilbrium probability density function
        self.fs = [] # records for plotting
        for t in tqdm(range(self.T), desc='t='): # iterate over time steps
            f_str_9yx = self.stream(f_9yx.copy()) # streams probability density function
            u_str_2yx, rho_str_yx = self.velocity(f_str_9yx), self.density(f_str_9yx) # calcualates average velocity and overall density
            f_eq_9yx = self.equilibrium_function(u_str_2yx, rho_str_yx) # calculates equilibrium function
            f_col_9yx = f_str_9yx + self.omega * (f_eq_9yx - f_str_9yx) # calculates collision part of LBTE
            u_col_2yx, rho_col_yx = self.velocity(f_col_9yx), self.density(f_col_9yx) # calcualates average velocity and overall density
            f_9yx = self.boundary_condition(f_col_9yx, u_col_2yx, rho_col_yx) # calculates boundary conditions
            self.fs.append(f_9yx) # records for plotting


    def run_parallel(self):

        # gets the initial conditions for the experiment
        u_2yx, rho_yx = self.initial_conditions()

        # gets equilbrium probability density function
        f_9yx = self.equilibrium_function(u_2yx, rho_yx)

        # defines coordinate range for subdomain
        y_start, y_end = self.rcoords[0]*self.ny//self.nsuby, (self.rcoords[0]+1)*self.ny//self.nsuby
        x_start, x_end = self.rcoords[1]*self.nx//self.nsubx, (self.rcoords[1]+1)*self.nx//self.nsubx

        # divides equilibrium probability density function into subdomain
        f_9subyx = f_9yx[:, y_start:y_end, x_start:x_end]

        self.fs = [] # records for plotting

        # iterate over time steps
        for t in tqdm(range(self.T), desc='t='):
            
            # set up
            f_9subyx = self.communicate_many_to_many(f_9subyx)
            
            # streams probability density function
            f_str_9subyx = self.stream(f_9subyx.copy())

            # calcualates average velocity and overall density
            u_str_2subyx, rho_str_subyx = self.velocity(f_str_9subyx), self.density(f_str_9subyx)

            # calculates equilibrium function
            f_eq_9subyx = self.equilibrium_function(u_str_2subyx, rho_str_subyx)
            
            # calculates collision part of LBTE
            f_col_9subyx = f_str_9subyx + self.omega * (f_eq_9subyx - f_str_9subyx)

            # calcualates average velocity and overall density
            u_col_2subyx, rho_col_subyx = self.velocity(f_col_9subyx), self.density(f_col_9subyx)

            # calculates boundary conditions
            f_9subyx = self.boundary_condition(f_col_9subyx, u_col_2subyx, rho_col_subyx)  

            self.fs.append(f_9subyx) # records for plotting


    def communicate_many_to_many(self, f_9yx):
        recvbuf = np.zeros(f_9yx[:, :, 1].shape)
        sR, dR, sL, dL, sU, dU, sD, dD = self.sd

        sendbuf = f_9yx[:, :, -2].copy()
        self.cartcomm.Sendrecv(sendbuf, dR, recvbuf=recvbuf, source=sR)
        f_9yx[:, :, 0] = recvbuf 

        sendbuf = f_9yx[:, :, 1].copy()
        self.cartcomm.Sendrecv(sendbuf, dL, recvbuf=recvbuf, source=sL)
        f_9yx[:, :, -1] = recvbuf

        sendbuf = f_9yx[:, 1, :].copy()
        self.cartcomm.Sendrecv(sendbuf, dU, recvbuf=recvbuf, source=sU)
        f_9yx[:, -1, :] = recvbuf

        sendbuf = f_9yx[:, -2, :].copy()
        self.cartcomm.Sendrecv(sendbuf, dD, recvbuf=recvbuf, source=sD)
        f_9yx[:, 0, :] = recvbuf
    
        return f_9yx

        
    ################ Helper Functions ################
            
    def check_mass_conservation(self):

        rhos = np.array([self.density(f) for f in self.fs])
        masses = [np.sum(rho) for rho in rhos]

        try:
            assert np.allclose(masses, masses[0])
            print('mass conserved')
        except AssertionError:
            print('mass not conserved: ' + str(masses))
            
            
    def plot_sin_over_time(self):
        fig, ax = plt.subplots(figsize=(6, 4))

        for i, f in enumerate(self.fs):
            if i % 100 == 0:
                u = self.velocity(self.fs[i])[1][:, 0]
                im = ax.plot(np.arange(self.ny), u)
                
        ax.set_xlabel('x-axis (nx)')
        ax.set_ylabel('velocity (u)')

        ax.legend()

        fig.tight_layout()
        
        fig.savefig(f'figures/shear_wave_decay.pdf', dpi=600)
        
        plt.show()


    def plot_couette_flow_over_time(self):
        fig, ax = plt.subplots(figsize=(6, 4))

        for i, f in enumerate(self.fs):
            if i % 10 == 0:
                ax.plot(np.arange(1, self.ny-1), self.velocity(self.fs[i])[1][:, 0][1:-1])

        ax.set_xlabel('y')
        ax.set_ylabel('velocity (u)')

        # cbar = fig.colorbar(sm, ax=ax)
        # cbar.ax.set_title('time (t x 10)')
        ax.legend()

        fig.tight_layout()
        
        fig.savefig(f'figures/couette_flow.pdf', dpi=600)
        
        plt.show()
        

    def animation(self):
        
        fig, axes = plt.subplots(figsize=(20, 4), ncols=4)
        im1 = axes[0].imshow(self.velocity(self.fs[0])[0].T, cmap='viridis',)
        im2 = axes[1].imshow(self.velocity(self.fs[0])[1].T, cmap='viridis')
        im3 = axes[2].imshow(self.density(self.fs[0]).T, cmap='viridis')
        
        X, Y = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        im4 = axes[3].streamplot(x=X, y=Y, u=self.velocity(self.fs[0])[1], v=self.velocity(self.fs[0])[0], density=2)
        axes[3].invert_yaxis()

        for j in range(4):
            axes[j].set_xlabel('x')
            axes[j].set_ylabel('y')
        
        def animate(i):
            
            im1.set_array(self.velocity(self.fs[i])[0])
            im2.set_array(self.velocity(self.fs[i])[1])
            im3.set_array(self.density(self.fs[i]))
            axes[3].cla()
            axes[3].streamplot(x=X, y=Y, u=self.velocity(self.fs[i])[1], v=self.velocity(self.fs[i])[0], density=2)
            axes[3].invert_yaxis()

            axes[0].set_title(f'y-velocity t = {i}')
            axes[1].set_title(f'x-velocity t = {i}')
            axes[2].set_title(f'density t = {i}')
            axes[2].set_title(f'streamplot t = {i}')
    
            return im1, im2, im3,
            
        anim = animation.FuncAnimation(fig, animate, frames=self.T)
        
        try:
            anim.save(f"figures/animation_{self.scenario}_{self.rank}.gif", writer='pillow', fps=15)
        except:
            anim.save(f"figures/animation_{self.scenario}.gif", writer='pillow', fps=15)
    

    def plot_amplitude_over_time(self):
        fig, ax = plt.subplots(figsize=(5, 4))
        
        time = np.arange(self.T)
        nu_true = (1/3)*((1/self.omega)-(1/2))
        amp_true = self.epsilon * np.exp(-nu_true*(2*np.pi/self.ny)**2*time)
        amp_est = [np.max(self.velocity(f)[1][:,0]) for f in self.fs]
        
        ax.plot(time, amp_true, label='true', linestyle='solid', zorder=1)
        ax.plot(time, amp_est, label='est.', linestyle='dashed', zorder=2)
        
        ax.set_xlabel('time (t)')
        ax.set_ylabel('amplitude')
        ax.legend()
        
        fig.tight_layout()
        
        fig.savefig(f'figures/amplitude_decay.pdf', dpi=600)
        
        plt.show()


    def plot_viscosity_over_omega(self, min_omega=0.01, max_omega=1.7, num=10):
        
        fig, ax = plt.subplots(figsize=(5, 4))
        
        omegas = np.linspace(min_omega, max_omega, num)
        t_1, y_a = 1, self.ny // 4

        nu_est, nu_true = [], []
        for omega in omegas:
            sim = Simulation(
                nx=100,
                ny=50,
                scenario='shear_wave_velocity',
                epsilon=0.05,
                omega=omega,
                T=100,
            )
            sim.run_simulation()
            
            nu_true.append((1/3)*((1/omega)-(1/2)))
            
            a_0 = sim.velocity(sim.fs[0])[1][0][y_a]
            a_1 = sim.velocity(sim.fs[t_1])[1][0][y_a]
            k = 2* np.pi / self.ny
        
            nu_est.append((np.log(a_0) - np.log(a_1)) / (k**2 * t_1))
            
        ax.plot(omegas, nu_true)


    def plot_final_velocity_field(self):

        f_9yx = self.fs[-1]
        u_2yx = self.velocity(f_9yx)


        X, Y = np.meshgrid(np.arange(u_2yx.shape[2]), np.arange(u_2yx.shape[1]))
        fig, ax = plt.subplots(figsize=(4, 4))

        ax.streamplot(x=X, y=Y, u=u_2yx[1], v=u_2yx[0], density=1.5)
        ax.invert_yaxis()

        try:
            fig.savefig(f"figures/streamplot_{self.scenario}_{self.rank}.pdf", dpi=600)
        except:
            fig.savefig(f"figures/streamplot_{self.scenario}.pdf", dpi=600)
 
    
if __name__ == '__main__':
    '''
    General process flow for a parallel experiment. To see how to run a sequential experiment please refer to Simulation.ipynb
    '''

    from mpi4py import MPI


    # instantiates the Simulation class
    sim = Simulation(
        nx=300,
        ny=300,
        scenario='sliding_lid_parallel',
        omega=0.5,
        T=1000
    )
    # sets experiment parameters
    sim.init_parallel()

    # runs experiment in parallel
    sim.run_parallel()

    # saves animation of velocity and density
    # sim.plot_final_velocity_field()