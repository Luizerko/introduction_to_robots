# MAC0318 Intro to Robotics
# Please fill-in the fields below with your info
#
# Name: Luis Vitor Zerkowski
# NUSP: 9837201
#
# ---
#
# Link para modelo: https://drive.google.com/file/d/1XFYKJcqDRAWM-N2MZHesAxiWRxzstO_f/view?usp=sharing
#
# Assignment 12 - Route planning
#
# Don't forget to run this file from the Duckievillage root directory path (example):
#   cd ~/MAC0318/duckievillage
#   conda activate duckietown
#   python3 assignments/route-planning/agent.py
#
# Submission instructions:
#  0. Add your name and USP number to the file header above.
#  1. Make sure that any last change hasn't broken your code. If the code crashes without running you'll get a 0.
#  2. Submit this file via e-disciplinas.

import sys
import pyglet
import numpy as np
import math
import random
from pyglet.window import key, mouse
from duckievillage import create_env, FRONT_VIEW_MODE
import cv2
import tensorflow

# Uncomment the following line if you get an error with the parallel library
# accusing multiple instantiations
#import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

COMMAND_LABELS = ["NOOP", "UP", "RIGHT", "LEFT"]
DIRECTIONS = {'n': 0, 's': 2, 'e': 1, 'w': 3}

# Evaluation agent.
class Agent:
    def __init__(self, env):
        ''' Initializes agent '''
        self.env = env
        self.radius = 0.0318 # R
        self.baseline = env.unwrapped.wheel_dist/2
        self.motor_gain = 0.68*0.0784739898632288
        self.motor_trim = 0.0007500911693361842

        self.command = -1

        # Color filters
        # Yellow dashed line
        self.inner_lower = np.array([20, 85, 20])
        self.inner_upper = np.array([30, 255, 255])
        # White solid line
        self.outer_lower = np.array([255, 180, 180])
        self.outer_upper = np.array([255, 255, 255])
        # Red markings on intersections
        self.inter_lower1 = np.array([0, 100, 20])
        self.inter_lower2 = np.array([140, 100, 20])
        self.inter_upper1 = np.array([10, 255, 255])
        self.inter_upper2 = np.array([255, 255, 255])

        key_handler = key.KeyStateHandler()
        env.unwrapped.window.push_handlers(key_handler)
        self.key_handler = key_handler

        # Load neural network that takes an image and returns linear and angular velocity.
        self.predictor = Agent.load_regression_model("./assignments/route-planning/mlp_atividade12.h5")
        self.score = 0

        # Auxiliary variables for deciding when (and how) to turn at an intersection.
        self.near_threshold = env.road_tile_size*0.9 # How near is it from an intersection?
        self.is_inside = False # Is the agent in an intersection?
        self.next_move = 0 # The index of the next move to take in route planning.
        self.plan = None # The (route) plan(ning).
        self.goal = None # Goal state.
        self.dt = 0 # Time it takes to go through intersection.

    def filter_image(self) -> np.ndarray:
        ''' Applies color filters to camera feed, returning the resulting image. '''
        # Remove horizon and resize to 80x42.
        I = cv2.resize(self.env.front()[180:,:,:], (80, 42))
        hsv = cv2.cvtColor(I, cv2.COLOR_RGB2HSV)
        # Apply white and yellow line filters.
        lane_mask = cv2.bitwise_or(cv2.inRange(hsv, self.inner_lower, self.inner_upper),
                                   cv2.inRange(I, self.outer_lower, self.outer_upper))//255
        # Apply red markings in intersections.
        inter_mask = cv2.bitwise_or(cv2.inRange(hsv, self.inter_lower1, self.inter_upper1),
                                    cv2.inRange(hsv, self.inter_lower2, self.inter_upper2))//255
        # Join both masks.
        mask = cv2.bitwise_or(lane_mask, inter_mask)
        # Mask image.
        I = cv2.bitwise_and(I, I, mask=mask)/255
        return I

    def get_pwm_control(self, v: float, w: float)-> (float, float):
        ''' Takes velocity v and angle w and returns left and right power to motors.'''
        V_l = (self.motor_gain - self.motor_trim)*(v-w*self.baseline)/self.radius
        V_r = (self.motor_gain + self.motor_trim)*(v+w*self.baseline)/self.radius
        return V_l, V_r

    @staticmethod
    def load_regression_model(filepath: str) -> tensorflow.keras.Model:
        ''' Loads a Tensorflow model. '''
        model = tensorflow.keras.models.load_model(filepath)
        # If you get an error (most likely due to loading a model saved in a newer version of TensorFlow, try using the line below instead)
        # model = tensorflow.keras.models.load_model(filepath, compile=False)
        model.summary()
        return model

    def next_intersection(self, p: tuple) -> (tuple, str):
        ''' Returns the next intersection as a tuple position. Also returns from which direction
        and the distance the agent will reach the intersection from. '''
        n = self.plan[self.next_move][0]
        dx, dy = p[0]-n[0], p[1]-n[1]
        d = dx*dx+dy*dy
        direction = None
        # What's the direction the agent is arriving the intersection from?
        if abs(dx) > abs(dy): direction = 'e' if dx > 0 else 'w'
        else: direction = 's' if dy > 0 else 'n'
        return n, direction, math.sqrt(d)

    def intersection_command(self, coming_from: str, going_to: str):
        ''' Returns the command (go straight onto the intersection, or turn left or right) that
        should be given to the agent when it arrives at the intersection. '''
        if (DIRECTIONS[coming_from] % 2) == (DIRECTIONS[going_to] % 2):
            # If they're going through the same axis (north to south, east to west or vice-versa).
            # Go straight.
            return 0
        elif (coming_from == 'w' and going_to == 's') or (coming_from == 'e' and going_to == 'n') \
            or (coming_from == 'n' and going_to == 'w') or (coming_from == 's' and going_to == 'e'):
            # Go right.
            return 1
        # Go left.
        return 2

    def go_to(self, q: tuple):
        ''' Tell agent to go to the closest drivable tile to p. '''
        self.plan = self.env.junction_graph.path(self.env.gps.track(), q)
        self.goal = self.env.nearest_drivable(q)
        self.next_move = 0

    def dist(self, p: tuple, q: tuple):
        ''' Euclidean distance between two points. '''
        return math.sqrt((p[0]-q[0])**2+(p[1]-q[1])**2)

    def preprocess(self, dt) -> (float, float):
        '''
        Compute velocities based on whether it is on an intersection or another drivable tile.
        '''

        # If there's no assigned goal, stay put.
        if self.goal is None: return 0, 0
        # Get an imprecise measurement of our position.
        p = self.env.gps.track()
        # If we're close enough to our goal state.
        if self.dist(p, self.goal) < self.env.road_tile_size*0.5:
            # Reached goal state. Reset everything.
            self.plan = None
            self.goal = None
            return 0, 0
        # Otherwise, check if plan has been reset.
        elif self.plan is None:
            # If so, then we've already passed the last intersection and are on our way to the goal state.
            I = self.filter_image()
            img = np.concatenate((I.reshape((-1, 42, 80, 3)), np.full((1, 42, 80, 1), -1)), axis=-1)
            v, w = self.predictor.predict(img)[0]
            #print('-1!')
            return v, w

        # Get the position of the next intersection, which directon we'll be coming from and
        # distance from it.
        q, direction, d = self.next_intersection(p)

        # If we're close enough and not already inside the intersection.
        if d < self.near_threshold and not self.is_inside:
            # Then set us as inside an intersection and set the direction we should go to take on
            # the intersection.
            self.is_inside = True
            self.command = self.intersection_command(direction, self.plan[self.next_move][1])

        # If we're inside the intersection.
        if self.is_inside:
            # Make an open-loop trajectory depending on where we should go from the intersection.
            if self.dt > 13.0:
                # Time's up. Get next move and set ourselves as outside the intersection.
                self.dt = 0
                self.next_move += 1
                self.is_inside = False
                self.command = -1
                # If that was our last move, then set our plan to None so that we can arrive safely
                # to our destination.
                if self.next_move >= len(self.plan):
                    self.plan = None
            else:
                # Here we give out the velocities for each command.
                # TODO: Change these open-loop commands to closed-loop versions.
                self.dt += dt
                I = self.filter_image()
                # Go straight.
                if self.command == 0:
                    img = np.concatenate((I.reshape((-1, 42, 80, 3)), np.full((1, 42, 80, 1), 0)), axis=-1)
                    v, w = self.predictor.predict(img)[0]
                    #print('0!')
                    return v, w
                # Turn right.
                elif self.command == 1:
                    img = np.concatenate((I.reshape((-1, 42, 80, 3)), np.full((1, 42, 80, 1), 1)), axis=-1)
                    v, w = self.predictor.predict(img)[0]
                    #print('1!')
                    return v, w
                # Turn left.
                elif self.command == 2:
                    img = np.concatenate((I.reshape((-1, 42, 80, 3)), np.full((1, 42, 80, 1), 2)), axis=-1)
                    v, w = self.predictor.predict(img)[0]
                    #print('2!')
                    return v, w
        else:
            # Otherwise we should keep lane-following as usual.
            self.command = -1

        # Get velocities from image.
        I = self.filter_image()
        img = np.concatenate((I.reshape((-1, 42, 80, 3)), np.full((1, 42, 80, 1), -1)), axis=-1)
        v, w = self.predictor.predict(img)[0]
        #print('-1!')
        return v, w

    def send_commands(self, dt):
        ''' Agent control loop '''
        pwm_left, pwm_right = 0, 0

        velocity, rotation = self.preprocess(dt)
        pwm_left, pwm_right = self.get_pwm_control(velocity, rotation)

        _, r, _, _ = self.env.step(pwm_left, pwm_right)
        self.env.render(text = f", command: {COMMAND_LABELS[self.command+1]}")

def main():
    print("MAC0318 - Assignment 12")
    env = create_env(
        raw_motor_input = True,
        noisy = True,
        mu_l = 0.007123895,
        mu_r = -0.000523123,
        std_l = 1e-7,
        std_r = 1e-7,
        seed = 101,
        map_name = './maps/udem1.yaml',
        draw_curve = False,
        draw_bbox = False,
        domain_rand = False,
        distortion = False,
        top_down = False,
        cam_height = 10,
        is_external_map = True,
        randomize_maps_on_reset = False,
        enable_junction = True,
        enable_gps = True,
    )

    env.reset()
    env.render() # show visualization

    @env.unwrapped.window.event
    def on_key_press(symbol, modifiers):
        nonlocal agent
        if symbol == key.ESCAPE: # exit simulation
            # Saves dataset to /tmp.
            env.close()
            sys.exit(0)

    @env.unwrapped.window.event
    def on_mouse_motion(x, y, _, __):
        # Gives us some visual indicator of where we're pointing to.
        pointer.pos[0], pointer.pos[2] = env.convert_coords(x, y)

    @env.unwrapped.window.event
    def on_mouse_release(x, y, button, _):
        if button == mouse.LEFT:
            # Go!
            agent.go_to(env.convert_coords(x, y))

    pointer = env.add_cone(0, 0, scale = 5.0)

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
