# MAC0318 Intro to Robotics
# Please fill-in the fields below with your info
#
# Name: Luis Vitor Pedreira Iten Zerkowski
# NUSP: 9837201
#
# ---
#
# Assignment 5 - Open-loop feedback control
#
# Task:
#  - Implement the following behavior agents:
#   1. SquareAgent   - moves in a fixed length square;
#   2. CircleAgent   - moves in a fixed radius circle;
#   3. OvertakeAgent - does a duck overtake.
#  - Use the constants from Assignment 4 to implement the power function and use it to move the
#  agents.
#
# Don't forget to run this file from the Duckievillage root directory path (example):
#   cd ~/MAC0318/duckievillage
#   conda activate duckietown
#   python3 assignments/control/openloop/agents.py
#
# Submission instructions:
#  0. Add your name and USP number to the file header above.
#  1. Make sure that any last change haven't broken your code. If the code crashes without running you'll get a 0.
#  2. Submit this file together with your trajectory plots via e-disciplinas.

import sys
import pyglet
import numpy as np
from matplotlib import pyplot as plt
import math
from pyglet.window import key
from duckievillage import create_env
import cv2

class Agent:
    # Agent initialization
    def __init__(self, environment):
        """ Initializes agent """
        self.env = environment
        # Time remaining.
        self.time = 0.0
        # If countdown is running.
        self.running = False

        # Wheel radius.
        self.R = 0.0318
        # Distance between wheels.
        self.L = environment.unwrapped.wheel_dist
        # K_m constant.
        self.K_m = 0.07543279656639355
        # K_t constant.
        self.K_t = 1.92196868297577e-05

        # Record trajectory.
        self.trajectory = [environment.get_position()]

        key_handler = key.KeyStateHandler()
        environment.unwrapped.window.push_handlers(key_handler)
        self.key_handler = key_handler

    def start(self):
        self.running = True

    def get_pwm_control(self, v: float, w: float)-> (float, float):
        ''' Takes velocity v and angle w and returns left and right power to motors.'''
        l = (self.K_m - self.K_t)*(v-w*self.L/2)/self.R
        r = (self.K_m + self.K_t)*(v+w*self.L/2)/self.R
        return l, r

    def send_commands(self, dt):
        ''' Agent control loop '''
        # Store pose so that we can plot the robot's trajectory.
        # These values should not be used by your agent, since we are running in open loop mode.
        p = self.env.get_position() 
        q = self.trajectory[-1]
        if (not math.isclose(p[0], q[0], abs_tol = 1e-4)) or (not math.isclose(p[1], q[1], abs_tol = 1e-4)):
            self.trajectory.append(p)

class SquareAgent(Agent):
    def __init__(self, env):
        '''Constructs a SquareAgent that walks a square.'''
        super().__init__(env)
        self.turning = False
        #self.env_frame = env.unwrapped.frame_rate

    def start(self):
        super().start()
        # Initial time we give for the robot when it starts moving. Change as needed.
        #Número de tarefas a serem cumpridas: reto, vira, reto, vira...
        self.time = 8

    def send_commands(self, dt):
        ''' Agent control loop '''
        super().send_commands(dt)

        # Here's a snippet of code for riding in a straight line. Adapt your code to ride in a
        # rectangle.
        pwm_left, pwm_right = 0, 0
        if self.time > 0:
            self.time -= dt
            #Reto ~ v*dt*execs = (0.2*0.03333...)*30
            #Curva ~ w*dt*execs = ((np.pi/2)*0.03333...)*30
            if self.time > 7:
                v, w = 0.2, 0.0
                pwm_left, pwm_right = self.get_pwm_control(v, w)
            elif self.time > 6:
                v, w = 0.0, (np.pi+0.3)/2
                pwm_left, pwm_right = self.get_pwm_control(v, w)
            elif self.time > 5:
                v, w = 0.2, 0.0
                pwm_left, pwm_right = self.get_pwm_control(v, w)
            elif self.time > 4:
                v, w = 0.0, (np.pi+0.3)/2
                pwm_left, pwm_right = self.get_pwm_control(v, w)
            elif self.time > 3:
                v, w = 0.2, 0.0
                pwm_left, pwm_right = self.get_pwm_control(v, w)
            elif self.time > 2:
                v, w = 0.0, (np.pi+0.3)/2
                pwm_left, pwm_right = self.get_pwm_control(v, w)
            elif self.time > 1:
                v, w = 0.2, 0.0
                pwm_left, pwm_right = self.get_pwm_control(v, w)

        elif self.running:
            self.running = False
        # End of snippet of code for line drawing.

        self.env.step(pwm_left, pwm_right)
        self.env.render()

class CircleAgent(Agent):
    def __init__(self, env):
        '''Constructs a CircleAgent that walks a circle.'''
        super().__init__(env)
        self.env_frame = env.unwrapped.frame_rate
        #print(self.env_frame)

    def start(self):
        super().start()
        # Initial time we give for the robot when it starts moving. Change as needed.
        #time ~ time = execs*0.03333...
        self.time = 6.8

    def send_commands(self, dt):
        ''' Agent control loop '''
        super().send_commands(dt)

        # Here's a snippet of code for riding in a straight line. Adapt your code to ride in a
        # rectangle.
        pwm_left, pwm_right = 0, 0
        if self.time > 0:
            self.time -= dt
            #v ~ execs*(v*dt) = 2*pi*r ~ execs*(0.1*0.03333...) = 2*pi*0.1
            #w ~ execs*(w*dt) = 2*pi = execs*(1*0.03333...) = 2*pi
            v, w = 0.1, 1
            pwm_left, pwm_right = self.get_pwm_control(v, w)
        elif self.running:
            self.running = False
        # End of snippet of code for line drawing.

        self.env.step(pwm_left, pwm_right)
        self.env.render()

class OvertakeAgent(Agent):
    def __init__(self, env):
        '''Constructs an OvertakeAgent that does an overtake.'''
        super().__init__(env)

    def start(self):
        super().start()
        # Initial time we give for the robot when it starts moving. Change as needed.
        #time ~ time = execs*0.03333...
        self.time = 10

    def send_commands(self, dt):
        ''' Agent control loop '''
        super().send_commands(dt)

        # Here's a snippet of code for riding in a straight line. Adapt your code to ride in a
        # rectangle.
        pwm_left, pwm_right = 0, 0
        if self.time > 0:
            self.time -= dt

            #Reto ~ v*dt*execs = (0.2*0.03333...)*54
            if self.time > (10 - 1.8):
                v, w = 0.2, 0.0


            #v ~ execs*(v*dt) = 2*pi*r/5 ~ execs*(0.1*0.03333...) = 2*pi*0.1/5
            #w ~ execs*(w*dt) = 2*pi/5 = execs*(1*0.03333...) = 2*pi/5
            elif self.time > (8.2 - 1.3):
                v, w = 0.1, 1
            elif self.time > (6.9 - 1.3):
                v, w = 0.1, -1

            #Reto ~ v*dt*execs = (0.2*0.03333...)*18
            elif self.time > (5.6 - 0.8):
                v, w = 0.2, 0.0

            #v ~ execs*(v*dt) = 2*pi*r/5 ~ execs*(0.1*0.03333...) = 2*pi*0.1/5
            #w ~ execs*(w*dt) = 2*pi/5 = execs*(1*0.03333...) = 2*pi/5
            elif self.time > (4.8 - 1.3):
                v, w = 0.1, -1
            elif self.time > (3.5 - 1.3):
                v, w = 0.1, 1.005

            #Reto ~ v*dt*execs = (0.2*0.03333...)*60
            else:
                v, w = 0.2, 0.0

            pwm_left, pwm_right = self.get_pwm_control(v, w)
        elif self.running:
            self.running = False
        # End of snippet of code for line drawing.

        self.env.step(pwm_left, pwm_right)
        self.env.render()


def plot_trajectory(T: np.ndarray, path: str):
    print("Saving trajectory...")
    plt.plot(*zip(*T))
    plt.savefig(path, bbox_inches='tight')
    print("  Saved to ", path)


def main():
    print("MAC0318 - Assignment 5")
    env = create_env(
        raw_motor_input = True,
        noisy = True,
        mu_l = 0.00007123895,
        mu_r = -0.00000523123,
        std_l = 1e-7,
        std_r = 1e-7,
        seed = 101,
        map_name = './maps/grassy_road.yaml',
        draw_curve = False,
        draw_bbox = False,
        domain_rand = False,
        #color_sky = [0, 0, 0],
        user_tile_start = (0, 0),
        distortion = False,
        top_down = False,
        cam_height = 10,
        is_external_map = True,
        randomize_maps_on_reset = False,
        enable_sun = True,
    )

    angle = env.unwrapped.cam_angle[0]

    env.start_pose = [[0.5, 0, 1.575], 0] # initial pose - position and heading
    env.reset()
    env.render('human') # show visualization

    agents = [SquareAgent(env), CircleAgent(env), OvertakeAgent(env)]
    which = 0

    agent = agents[which]

    @env.unwrapped.window.event
    def on_key_press(symbol, modifiers):
        nonlocal agent, which
        if symbol == key.ESCAPE: # exit simulation
            env.close()
            sys.exit(0)
        elif symbol == key.SPACE: # Tells agent to start performing an action.
            agent.start()
        elif symbol == key.RETURN:  # Reset pose.
            env.reset_pos()
        elif symbol == key.PERIOD: # Change to next agent.
            which = (which + 1) % 3
            pyglet.clock.unschedule(agent.send_commands)
            agent = agents[which]
            env.reset_pos()
            pyglet.clock.schedule_interval(agent.send_commands, 1.0 / env.unwrapped.frame_rate)
        elif symbol == key.COMMA: # Change to previous agent.
            which = (which + len(agents)-1) % 3
            pyglet.clock.unschedule(agent.send_commands)
            agent = agents[which]
            env.reset_pos()
            pyglet.clock.schedule_interval(agent.send_commands, 1.0 / env.unwrapped.frame_rate)
        elif symbol == key.P: # Save trajectory so far.
            print('Função desativada por ora')
            plot_trajectory(agent.trajectory, f"{agent.__class__.__name__}-{env.unwrapped.step_count}.png")

    pyglet.clock.schedule_interval(agent.send_commands, 1.0 / env.unwrapped.frame_rate)
    pyglet.app.run()
    env.close()

if __name__ == '__main__':
    main()