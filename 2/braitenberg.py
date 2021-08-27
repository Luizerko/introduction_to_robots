# MAC0318 Intro to Robotics
# Please fill-in the fields below with your info
#
# Name: Luis Vitor Pedreira Iten Zerkowski
# NUSP: 9837201
#
# ---
#
# Assignment 2 - Braitenberg vehicles
#
# Task:
#  - Implement a reactive agent that implements the "lover" behaviour of Braitenberg's vehicle.
# Your agent should approach duckies without reaching too close (and without colliding with them!)
# This is achieved by modifying the left and right activation matrices and the motor connections
#
# Don't forget to run this from the Duckievillage root directory (example):
#   cd ~/MAC0318/duckievillage
#   python3 assignments/braitenberg/braitenberg.py
#
# Submission instructions:
#  0. Add your name and USP number to the file header above.
#  1. Make sure that any last change haven't broken your code. If the code chrases without running you'll get a 0.
#  2. Submit this file via e-disciplinas.
#  3. Push changes to your git fork.

import sys
import pyglet
import numpy as np
from pyglet.window import key
from duckievillage import create_env
import cv2

class Agent:
    # Agent initialization
    def __init__(self, environment):
        """ Initializes agent """
        self.env = environment
        # Color segmentation hyperspace
        self.lower_hsv = np.array([5, 70, 90])
        self.upper_hsv = np.array([40, 255, 255])
        # Acquire image for initializing activation matrices
        img = self.env.front()
        img_shape = img.shape[0], img.shape[1]
        self.left_motor_matrix = np.zeros(shape=img_shape, dtype="float32")
        self.right_motor_matrix = np.zeros(shape=img_shape, dtype="float32")
        # TODO! Replace with your code
        # Each motor activation matrix specifies how much power is given to the respective motor after the image processing routines are applied
        self.left_motor_matrix[:, :] = -1
        self.left_motor_matrix[225:375, 325:475] = 1
        self.right_motor_matrix[:, :] = -1
        self.right_motor_matrix[225:375, 325:475] = 1

    # Image processing routine - Color segmentation
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """ Returns a 2D array mask color segmentation of the image """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)//255
        #     masked = cv2.bitwise_and(image, image, mask=mask)
        return mask

    def send_commands(self, dt):
        ''' Agent control loop '''
        # acquire front camera image
        img = self.env.front()
        # run image processing routines
        P = self.preprocess(img)
        # build left and right signals
        L = float(np.sum(P * self.left_motor_matrix))
        R = float(np.sum(P * self.right_motor_matrix))
        limit = img.shape[0]*img.shape[1]
        # These are big numbers, thus rescale them to unit interval
        L = rescale(L, 0, limit)
        R = rescale(R, 0, limit)
        # Tweak with the constants below to get to change velocity or stabilize movements
        # Recall that pwm sets wheel torque, and is capped to be in [-1,1]
        gain = 5.0
        const = 0.2 # power under null activation - this ensures the robot does not halt
        pwm_left = const + R * gain
        pwm_right = const + L * gain
        # print('>', L, R, pwm_left, pwm_right) # uncomment for debugging
        # Now send command
        self.env.step(pwm_left, pwm_right)
        self.env.render('human')


def rescale(x: float, L: float, U: float):
    ''' Map scalar x in interval [L, U] to interval [0, 1]. '''
    return (x - L) / (U - L)


def main():
    print("MAC0318 - Assignment 1")
    env = create_env(
      raw_motor_input = True,
      seed = 101,
      map_name = './maps/nothing.yaml',
      draw_curve = False,
      draw_bbox = False,
      domain_rand = False,
      #color_sky = [0, 0, 0],
      user_tile_start = (0, 0),
      distortion = False,
      top_down = False,
      cam_height = 10,
      is_external_map = True,
      enable_lightsensor = False
    )
    for o in env.objects: o.scale = 0.085

    angle = env.unwrapped.cam_angle[0]

    env.start_pose = [[0.5, 0, 0.5], 150]
    env.reset()
    env.render('human')

    @env.unwrapped.window.event
    def on_key_press(symbol, modifiers):
        if symbol == key.ESCAPE: # exit simulation
            env.close()
            sys.exit(0)
        elif symbol == key.BACKSPACE or symbol == key.SLASH: # reset simulation
            print("RESET")
            env.reset()
        elif symbol == key.RETURN:  # Take a screenshot
            print('saving screenshot')
            img = env.render('rgb_array')
            cv2.imwrite(f'screenshot-{env.unwrapped.step_count}.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        env.render()

    # Instantiate agent
    agent = Agent(env)
    # Call send_commands function from periodically (to simulate processing latency)
    pyglet.clock.schedule_interval(agent.send_commands, 1.0 / env.unwrapped.frame_rate)
    # Now run simulation forever (or until ESC is pressed)
    pyglet.app.run()
    env.close()

if __name__ == '__main__':
    main()
