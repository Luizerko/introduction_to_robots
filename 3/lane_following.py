# MAC0318 Intro to Robotics
# Please fill-in the fields below with your info
#
# Name: Luis Vitor Pedreira Iten Zerkowski
# NUSP: 9837201
#
# ---
#
# Assignment 3 - Braitenberg vehicle for lane following
#
# Task:
#  - Implement a Braitenberg's vehicle that performs the task of lane following in the duckietown environment.
# Your agent should be able to follow a lane by reacting to the traffic markings on the road. Construct a
# color segmentation filter to identify road markings and adapt the "lover" behavior so that the robot
# moves forward while maintaining a short distance from either lane marking.
#
# Don't forget to run this file from the Duckievillage root directory path (example):
#   cd ~/MAC0318/duckievillage
#   conda activate duckietown
#   python3 assignments/braitenberg/lane_following.py
#
# Submission instructions:
#  0. Add your name and USP number to the file header above.
#  1. Make sure that any last change haven't broken your code. If the code crashes without running you'll get a 0.
#  2. Submit this file via e-disciplinas.
#  3. Push changes to your git fork.

import sys
import pyglet
import numpy as np
from pyglet.window import key
from duckievillage import create_env
import cv2

import matplotlib.pyplot as plt
import random

class Agent:
    # Agent initialization
    def __init__(self, environment):
        """ Initializes agent """
        self.env = environment
        # Color segmentation hyperspace - TODO: MODIFY THE VALUES BELOW
        self.inner_lower = np.array([10, 80, 140])
        self.inner_upper = np.array([40, 255, 255])
        self.outer_lower = np.array([0, 0, 190])
        self.outer_upper = np.array([179, 80, 255])

        self.pista_lower = np.array([0, 0, 25])
        self.pista_upper = np.array([179, 70, 170])

        # Acquire image for initializing activation matrices
        img = self.env.front()
        img_shape = img.shape[0], img.shape[1]
        self.inner_left_motor_matrix = np.zeros(shape=img_shape, dtype="float32")
        self.inner_right_motor_matrix = np.zeros(shape=img_shape, dtype="float32")
        self.outer_left_motor_matrix = np.zeros(shape=img_shape, dtype="float32")
        self.outer_right_motor_matrix = np.zeros(shape=img_shape, dtype="float32")

        self.pista_left_motor_matrix = np.zeros(shape=img_shape, dtype="float32")
        self.pista_right_motor_matrix = np.zeros(shape=img_shape, dtype="float32")

        # Connecition matrices - TODO: Replace with your code
        
        #Returning to the right direction
        self.inner_left_motor_matrix[350:, 7*img_shape[1]//8:] = 1
        self.inner_right_motor_matrix[350:, 7*img_shape[1]//8:] = -1

        #Turning right
        self.inner_left_motor_matrix[250:, img_shape[1]//8:7*img_shape[1]//8] = 1
        self.inner_right_motor_matrix[250:, img_shape[1]//8:7*img_shape[1]//8] = -1
        
        #Accelerating on the straight line
        self.inner_left_motor_matrix[350:, :img_shape[1]//8] = 1
        self.inner_right_motor_matrix[350:, :img_shape[1]//8] = 1

        #Returning to the right direction
        self.outer_left_motor_matrix[450:, :img_shape[1]//8] = -0.6
        self.outer_right_motor_matrix[450:, :img_shape[1]//8] = 0.8

        #Turning left
        self.outer_left_motor_matrix[340:, img_shape[1]//8:7*img_shape[1]//8] = -0.8
        self.outer_right_motor_matrix[340:, img_shape[1]//8:7*img_shape[1]//8] = 1
        
        #Accelerating on the straight line
        self.outer_left_motor_matrix[350:, 7*img_shape[1]//8:] = 1
        self.outer_right_motor_matrix[350:, 7*img_shape[1]//8:] = 1

        #Finding our way back to the road on the left
        self.pista_left_motor_matrix[:300, :img_shape[1]//3] = 0.8
        self.pista_right_motor_matrix[:300, :img_shape[1]//3] = 1

        #Finding our way back to the road in front of us
        self.pista_left_motor_matrix[:300, img_shape[1]//3:2*img_shape[1]//3] = 1
        self.pista_right_motor_matrix[:300, img_shape[1]//3:2*img_shape[1]//3] = 1

        #Finding our way back to the road on the left
        self.pista_left_motor_matrix[:300, 2*img_shape[1]//3:] = 1
        self.pista_right_motor_matrix[:300, 2*img_shape[1]//3:] = 0.8

        #Slow down the road finder matrix if you are already on the road
        self.pista_left_motor_matrix[560:, :] = -0.8
        self.pista_right_motor_matrix[560:, :] = -0.8

    # Image processing routine - Color segmentation
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """ Returns a 2D array mask color segmentation of the image """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) # obtain HSV representation of image
        # filter out dashed yellow "inner" line
        inner_mask = cv2.inRange(hsv, self.inner_lower, self.inner_upper)//255
        # filter out solid white "outer" line
        outer_mask = cv2.inRange(hsv, self.outer_lower, self.outer_upper)//255

        pista_mask = cv2.inRange(hsv, self.pista_lower, self.pista_upper)//255

        # Note: it is possible to filter out pixels in the RGB format
        #  by replacing `hsv` with `image` in the commands above
        # produces combined mask (might or might not be useful)
        mask = cv2.bitwise_or(inner_mask, outer_mask)
        self.masked = cv2.bitwise_and(image, image, mask=mask)

        '''
        inner_img = cv2.bitwise_and(image, image, mask=inner_mask)
        outer_img = cv2.bitwise_and(image, image, mask=outer_mask)
        pista_img = cv2.bitwise_and(image, image, mask=pista_mask)
        if random.randint(0, 100) == 0:
            fig, ax = plt.subplots(1, 3, figsize=(8, 8))
            ax[0].imshow(cv2.cvtColor(inner_img, cv2.COLOR_BGR2RGB))
            ax[1].imshow(cv2.cvtColor(outer_img, cv2.COLOR_BGR2RGB))
            ax[2].imshow(cv2.cvtColor(pista_img, cv2.COLOR_BGR2RGB))
            plt.show()
        '''

        return inner_mask, outer_mask, pista_mask, mask

    def send_commands(self, dt):
        ''' Agent control loop '''
        # acquire front camera image
        img = self.env.front()
        # run image processing routines
        P, Q, R, M = self.preprocess(img) # returns inner, outter and combined mask matrices
        # build left and right motor signals from connection matrices and masks (this is a suggestion, feel free to modify it)
        L = float(np.sum(P * self.inner_left_motor_matrix)) + float(np.sum(Q * self.outer_left_motor_matrix)) + \
            float(np.sum(R * self.pista_left_motor_matrix))
        R = float(np.sum(P * self.inner_right_motor_matrix)) + float(np.sum(Q * self.outer_right_motor_matrix)) + \
            float(np.sum(R * self.pista_right_motor_matrix))
        # Upper bound on the values above (very loose bound)
        limit = img.shape[0]*img.shape[1]*2
        # These are big numbers, better to rescale them to the unit interval
        L = rescale(L, 0, limit)
        R = rescale(R, 0, limit)
        # Tweak with the constants below to get to change velocity or to stabilize the behavior
        # Recall that the pwm signal sets the wheel torque, and is capped to be in [-1,1]
        gain = 3.0   # increasing this will increasing responsitivity and reduce stability
        const = 0.15 # power under null activation - this affects the base velocity
        pwm_left = const + L * gain
        pwm_right = const + R * gain
        # print('>', L, R, pwm_left, pwm_right) # uncomment for debugging
        # Now send command to motors
        self.env.step(pwm_left, pwm_right)
        #  for visualization
        self.env.render('human')


def rescale(x: float, L: float, U: float):
    ''' Map scalar x in interval [L, U] to interval [0, 1]. '''
    return (x - L) / (U - L)


def main():
    print("MAC0318 - Assignment 3")
    env = create_env(
      raw_motor_input = True,
      seed = 101,
      map_name = './maps/loop_empty.yaml',
      draw_curve = False,
      draw_bbox = False,
      domain_rand = False,
      #color_sky = [0, 0, 0],
      user_tile_start = (0, 0),
      distortion = False,
      top_down = False,
      cam_height = 10,
      is_external_map = True,
    )

    angle = env.unwrapped.cam_angle[0]

    env.start_pose = [[0.8, 0, 0.8], 4.5] # initial pose - position and heading
    env.reset()
    env.render('human') # show visualization

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
            cv2.imwrite(f'screenshot-masked-{env.unwrapped.step_count}.png', cv2.cvtColor(agent.masked, cv2.COLOR_RGB2BGR))
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
