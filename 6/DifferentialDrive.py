class DifferentialRobot:
    ''' Simulates kinematics behaviour of a 2D differential drive robot driven by linear and angular velocities. '''
    def __init__(self, x0=None, R=0.0381, LL=0.1, max_speed=5, max_acc=10):
        ''' Initialize differential drive robot. '''
        import numpy as np
        self.wheel_radius = R # wheel radius size [m]
        self.wheel_base = LL # wheel base distance [m]
        self.max_speed = max_speed # max. speed [m/s]        
        self.max_acceleration = max_acc # max. acceleration
        self.dt = 0.1 # sampling rate [s]
        if x0 is None:
            self.x = np.zeros(3)
        else:
            self.x = x0[:]
        self.x_hist = [self.x.copy()]
        self.v_hist = [0.0]
        self.w_hist = [0.0]
        self.v_prev = 0.0
        self.w_prev = 0.0
        
    def diff(self, x, v, w):
        ''' Ordinary differential equation for forward kinematics motion. '''
        import numpy as np
        acc = (v - self.v_prev)/self.dt
        if acc > self.max_acceleration:
            v = self.v_prev + self.max_acceleration*self.dt
        elif acc < -self.max_acceleration:
            v = self.v_prev - self.max_acceleration*self.dt
        self.v_prev = v
        # limit velocity
        v = min(self.max_speed, max(v, -self.max_speed))
        # compute derivative
        dx = np.array([v*np.cos(x[2]), v*np.sin(x[2]), w])
        return dx

    def forward(self, v, w):
        ''' Updates the vehicle state by applying the forward kinematics equation 
        for one timestep with control signal u = (v, w) for velocity v and steering w.
        Corresponds to Euler forward integration of the differential equation.
        '''
        # update state
        dx = self.dt * self.diff(self.x, v, w)
        self.x += dx
        # store state and control
        self.x_hist.append(self.x.copy())
        self.v_hist.append(v)
        self.w_hist.append(w)
        return self.x
        
    def plot(self, figsize=(7,7)):
        ''' Plot trajectory history. '''
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, axes = plt.subplots(2,2, figsize=figsize)

        axes[0,0].plot([x[0] for x in self.x_hist], [x[1] for x in self.x_hist], "r.")
        axes[0,0].set_aspect(1)
        axes[0,0].set_xlabel("X [m]")
        axes[0,0].set_ylabel("Y [m]")
        axes[0,0].set_title("Position")
        axes[0,1].plot(self.dt*np.arange(0, len(self.x_hist), 1), [x[2] for x in self.x_hist], "r.")
        asp = np.diff(axes[0,1].get_xlim())[0] / np.diff(axes[0,1].get_ylim())[0]
        axes[0,1].set_aspect(asp)                
        axes[0,1].set_xlabel("Time [s]")
        axes[0,1].set_ylabel("Heading [rad]")
        axes[0,1].set_title("Heading")
        axes[1,0].plot(self.dt*np.arange(0, len(self.v_hist), 1), self.v_hist)
        axes[1,0].set_xlabel('Time [s]')
        axes[1,0].set_ylabel('Speed [m/s]')
        asp = np.diff(axes[1,0].get_xlim())[0] / np.diff(axes[1,0].get_ylim())[0]
        axes[1,0].set_aspect(asp)        
        axes[1,1].plot(self.dt*np.arange(0, len(self.w_hist), 1), self.w_hist)
        axes[1,1].set_xlabel('Time [s]')
        axes[1,1].set_ylabel('Steering [rad/s]')
        asp = np.diff(axes[1,1].get_xlim())[0] / np.diff(axes[1,1].get_ylim())[0]
        axes[1,1].set_aspect(asp)        
        plt.tight_layout()        
        return fig, axes
            
        
class DifferentialRobot2(DifferentialRobot):
    ''' 2D Differential drive kinematics given wheel rotation speeds as control. '''        
    def diff(self, x, phi_l, phi_r):
        ''' Forward kinematics motion. '''
        v = self.wheel_radius*(phi_l+phi_r)/2
        w = self.wheel_radius*(phi_r-phi_l)*self.wheel_base
        return super(DifferentialRobot2, self).diff(x, v, w)
      