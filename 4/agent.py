# MAC0318 Intro to Robotics
# Please fill-in the fields below with your info
#
# Name: Luis Vitor Pedreira Iten Zerkowski
# NUSP: 9837201
#
# ---
#
# Assignment 4 - Modeling
#
# Task:
#  - Empirically estimate parameters K_m and K_t
#
# Don't forget to run this file from the Duckievillage root directory path (example):
#   cd ~/MAC0318/duckievillage
#   conda activate duckietown
#   python3 assignments/control/model/agent.py
#
# Submission instructions:
#  0. Add your name and USP number to the file header above.
#  1. Make sure that any last change haven't broken your code. If the code crashes without running you'll get a 0.
#  2. Submit this file via e-disciplinas.

import sys
import pyglet
import numpy as np
import math
from pyglet.window import key
from duckievillage import create_env
import cv2

v, w = 0, 0
pwm_left, pwm_right = 0, 0
d, t = 0, 0
pwm_left_aux, pwm_right_aux = 0, 0

class Agent:
    # Agent initialization
    def __init__(self, environment):
        ''' Initializes agent '''
        self.env = environment
        # Time remaining.
        self.time = 0.0
        # Position at start of countdown.
        self.p_start = [0, 0]
        # Position at end of countdown.
        self.p_end = [0, 0]
        # Angle at start of countdown.
        self.a_start = 0
        # Angle at end of countdown.
        self.a_end = 0
        # If countdown is running.
        self.running = False
        # Wheel radius.
        self.radius = 0.0318 # [m]
        # Distance between wheels
        self.baseline = 0.102 # [m]

        # Motor constants
        self.motor_gain = 0.074 # K_m -- you should modify this value
        self.motor_trim = 0.0000189  # K_t -- you should modify this value

        key_handler = key.KeyStateHandler()
        environment.unwrapped.window.push_handlers(key_handler)
        self.key_handler = key_handler

    def get_pwm_control(self, v: float, w: float)-> (float, float):
        ''' Takes velocity v and angle w and returns left and right power to motors.'''
        V_l = (self.motor_gain - self.motor_trim)*(v-w*self.baseline/2)/self.radius
        V_r = (self.motor_gain + self.motor_trim)*(v+w*self.baseline/2)/self.radius
        return V_l, V_r

    def send_commands(self, dt):
        ''' Agent control loop '''
        global v
        global w

        # Map keys to velocity and angle actions.
        if self.key_handler[key.W]:
            v = 0.15 # move forward at 0.2 [m/s]
        if self.key_handler[key.A]:
            w = 0.05 # rotate counter-clockwise 10 [rot/s]
        if self.key_handler[key.S]:
            v = -0.15 # move backward at 0.2 [m/s]
        if self.key_handler[key.D]:
            w = -0.05 # rotate clockwise at 10 [rot/s]

        pwm_left, pwm_right = self.get_pwm_control(v, w)

        '''
        global pwm_left
        global pwm_right

        global pwm_left_aux
        global pwm_right_aux

        global d
        global t

        pwm_left = 0
        pwm_right = 0

        # Here's a snippet of code for measuring and estimating constants.
        # Add power to motors as long as time hasn't run out yet.
        # You may remove this after you have already estimated constants.
        if self.time > 0:
            self.time -= dt
            if self.motor_trim == 0:
                pwm_left = 0.15
                pwm_right = 0.15

            elif self.motor_trim != 0:
                pwm_left, pwm_right = self.get_pwm_control(d, 0)
            
            pwm_left_aux = pwm_left
            pwm_right_aux = pwm_right

        elif self.running:
            self.running = False
            self.p_end = self.env.get_position()
            self.a_end = self.env.cur_angle
            print(f"Started at position {self.p_start}\n  Ended at {self.p_end}")
            print(f"Started at angle {self.a_start}\n  Ended at {self.a_end}\n")

            d = dist(self.p_start, self.p_end)
            t = self.a_end-self.a_start
            print(f"Distance: {d}")
            print(f"Angle difference: {t}")

            print()
            a = pwm_left_aux*self.radius/(d - t*self.baseline/2)
            b = pwm_right_aux*self.radius/(d + t*self.baseline/2)
            print(f" K_m - K_t = ", a)
            print(f" K_m + K_t = ", b)
            print(f" K_m = ", (a+b)/2)
            print(f" K_t = ", (b-a)/2)
            print()
            
            self.motor_gain = (a+b)/2
            self.motor_trim = (b-a)/2

            d_2 = (pwm_left_aux*self.radius)/(self.motor_gain - self.motor_trim)
            self.motor_gain = self.radius*pwm_right_aux/d_2 - self.motor_trim

            # Power to motors to correctly move distance d with no angular error.
            pwm_left, pwm_right = self.get_pwm_control(d, 0)
            print(f"  Estimate for K_m = {self.motor_gain}")
            print(f"  Estimate for K_t = {self.motor_trim}")
            print(f"    Verifying: for v = {d} and w = 0\n    pwm_l, pwm_r = ", pwm_left, pwm_right)
        # End of measurement snippet.

        #print((pwm_left, pwm_right))
        #print(self.motor_gain)
        #print(self.motor_trim)
        '''

        self.env.step(pwm_left, pwm_right)
        self.env.render()

def dist(p: np.ndarray, q: np.ndarray):
    '''Returns the distance between two points p and q.'''
    return math.sqrt(np.sum((p-q)**2))

def main():
    print("MAC0318 - Assignment 4")
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

    env.start_pose = [[0.5, 0, 1.575], 0]
    env.reset()
    env.render('human') # show visualization

    @env.unwrapped.window.event
    def on_key_press(symbol, modifiers):
        if symbol == key.ESCAPE: # exit simulation
            env.close()
            sys.exit(0)
        elif symbol == key.SPACE: # Tells agent to perform an action for 2 seconds.
            agent.time = 1
            agent.running = True
            agent.p_start = env.get_position()
            agent.a_start = env.cur_angle
        elif symbol == key.RETURN:  # Reset pose.
            env.reset_pos()

        env.render() # show image to user

    # Instantiate agent
    agent = Agent(env)
    # Call send_commands function from periodically (to simulate processing latency)
    pyglet.clock.schedule_interval(agent.send_commands, 1.0 / env.unwrapped.frame_rate)
    # Now run simulation forever (or until ESC is pressed)
    pyglet.app.run()
    # When it's done, close environment and exit program
    env.close()

if __name__ == '__main__':
    main()