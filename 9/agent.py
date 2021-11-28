# MAC0318 Intro to Robotics
# Please fill-in the fields below with your info
#
# Name: Luis Vitor Pedreira Iten Zerkowski
# NUSP: 9837201
#
# https://drive.google.com/file/d/1JM4v5gt42t5QfFS5AqXTO6nwNJxhBZja/view?usp=sharing
#
# ---
#
# Assignment 9 - End-to-end lane following with convolutional neural networks
#
# Don't forget to run this file from the Duckievillage root directory path (example):
#   cd ~/MAC0318/duckievillage
#   conda activate duckietown
#   python3 assignments/regression-cnn/agent.py
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
from pyglet.window import key
from duckievillage import create_env, FRONT_VIEW_MODE
import cv2
import tensorflow

# Uncomment the following line if you get an error with the parallel library
# accusing multiple instantiations
#import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Basic agent.
class Agent:
    def __init__(self, env, randomize: bool = False):
        self.env = env
        self.radius = 0.0318 # R
        self.baseline = env.unwrapped.wheel_dist/2
        self.motor_gain = 0.68*0.0784739898632288
        self.motor_trim = 0.0007500911693361842

        if randomize:
            f = [self.env.add_static_duckie, self.env.add_static_big_duckie, self.env.add_cone]
            k = self.env.road_tile_size/2
            s = (-k/2, k/2)
            r = lambda: random.random()*random.choice(s)
            for i, t in enumerate(self.env.grid):
                if (t["kind"] == "floor") and (random.random() > 0.5):
                    x, y = t["coords"]
                    random.choice(f)(x*self.env.road_tile_size+k+r(), y*self.env.road_tile_size+k+r())

            self.env.domain_rand = True
            self.env.color_sky = [random.random(), random.random(), random.random()]
            self.env.color_ground = (random.random(), random.random(), random.random())
            self.env.start_pose = self.env.random_road_pose()
            self.env.random_reset()

        key_handler = key.KeyStateHandler()
        env.unwrapped.window.push_handlers(key_handler)
        self.key_handler = key_handler

        self.velocity = 0.2 # robot's logitudinal velocity
        self.rotation = 0.0 # robot's angular velocity
        self.horizon = 180
        self.dt = (1.0/self.env.unwrapped.frame_rate)

        self.kp = 7
        
        self.ki = 0.001
        self.accum_err = 0

        self.kd = 0.7
        self.last_err = 0

    def get_pwm_control(self, v: float, w: float)-> (float, float):
        ''' Takes velocity v and angle w and returns left and right power to motors.'''
        V_l = (self.motor_gain - self.motor_trim)*(v-w*self.baseline)/self.radius
        V_r = (self.motor_gain + self.motor_trim)*(v+w*self.baseline)/self.radius
        return V_l, V_r

    def preprocess(self) -> float:
        pass

    def send_commands(self, dt):
        pass

# Agent for data collection.
class DataAgent(Agent):
    def __init__(self, environment):
        ''' Initializes agent '''
        super().__init__(environment, randomize = False)

        self.C = 6

        self.images = []
        self.labels = []

    def preprocess(self) -> float:
        d, alpha = self.env.lf_target()
        return self.C*d+alpha

    def send_commands(self, dt):
        ''' Agent control loop '''
        pwm_left, pwm_right = 0, 0

        y = self.preprocess()
        # print(y) # uncomment this for debugging

        self.accum_err -= y

        self.rotation = self.kp*(-y) + self.ki*self.accum_err + self.kd*((-y) - self.last_err)/self.dt
        pwm_left, pwm_right = self.get_pwm_control(self.velocity, self.rotation)

        self.last_err = -y

        I = self.env.front() # capture image from the robot's front camera        
        
        I = cv2.resize(I[self.horizon:,:,:], (80, 60)) # crop and resize it to reduce processing cost

        '''
        if self.key_handler[key.K] or self.env.unwrapped.step_count%10 == 0 or abs(y) > 0.05: # Capture image -- good for debuging and further improving model
            import matplotlib.pyplot as plt
            #print(self.env.unwrapped.step_count)
            #plt.imshow(I)
            filename=f'./assignments/regression-cnn/data/6-image-capture-{self.env.unwrapped.step_count}.png'
            np.save(filename, I)
            f = open("./assignments/regression-cnn/data/values.txt", "a+")
            f.write(f'6-image-capture-{self.env.unwrapped.step_count}.png,{y}\n')
            #plt.savefig(filename)
            #print("  Saved image to ", filename)
        '''

        self.env.step(pwm_left, pwm_right) # send commands to motor
        self.env.render() # simulate environment

# Evaluation agent.
class EvaluationAgent(Agent):
    def __init__(self, environment):
        ''' Initializes agent '''
        super().__init__(environment, randomize = False)
        self.pose_estimator = EvaluationAgent.load_regression_model("assignments/regression-cnn/mlp_atividade9.h5")
        self.score = 0
        self.velocity = 0.2

    @staticmethod
    def load_regression_model(filepath: str) -> tensorflow.keras.Model:
        ''' Loads a Tensorflow model. '''
        model = tensorflow.keras.models.load_model(filepath)
        # If you get an error (most likely due to loading a model saved in a newer version of TensorFlow, try using the line below instead)
        # model = tensorflow.keras.models.load_model(filepath, compile=False)
        model.summary()
        return model

    def preprocess(self) -> (float, float):
        '''Returns the metric to be used as signal for the PID controller.'''
        I = cv2.resize(self.env.front(), (80, 60))/255
        #y = self.pose_estimator.predict(I.reshape((-1, 60, 80, 3)))[0]
        y = self.pose_estimator.predict(I.reshape((-1, 80, 60, 3)))[0]
        return y

    def send_commands(self, dt):
        ''' Agent control loop '''
        pwm_left, pwm_right = 0, 0

        y = self.preprocess()

        self.accum_err -= y

        self.rotation = self.kp*(-y) + self.ki*self.accum_err + self.kd*((-y) - self.last_err)/self.dt
        pwm_left, pwm_right = self.get_pwm_control(self.velocity, self.rotation)

        self.last_err = -y

        #pwm_left, pwm_right = self.get_pwm_control(velocity, rotation)
        _, r, _, _ = self.env.step(pwm_left, pwm_right)
        self.score += (r-self.score)/self.env.step_count
        self.env.render(text = f", score: {self.score:.3f}")

def main():
    print("MAC0318 - Assignment 9")
    env = create_env(
        raw_motor_input = True,
        noisy = True,
        mu_l = 0.007123895,
        mu_r = -0.000523123,
        std_l = 1e-7,
        std_r = 1e-7,
        seed = 101,
        map_name = './maps/loop_empty.yaml',
        draw_curve = False,
        draw_bbox = False,
        domain_rand = False,
        user_tile_start = (1, 3),
        distortion = False,
        top_down = False,
        cam_height = 10,
        is_external_map = True,
        randomize_maps_on_reset = False,
        # video_path = "/tmp/regression-cnn.mp4",
    )

    env.set_view(FRONT_VIEW_MODE)
    env.reset()
    env.render() # show visualization

    @env.unwrapped.window.event
    def on_key_press(symbol, modifiers):
        nonlocal agent
        if symbol == key.ESCAPE: # exit simulation
            # Saves dataset to /tmp.
            if isinstance(agent, DataAgent):
                np.save("/tmp/images.npy", agent.images, allow_pickle = True)
                np.save("/tmp/labels.npy", agent.labels, allow_pickle = True)
            env.close()
            sys.exit(0)
        elif (symbol == key.E) or (symbol == key.D):
            # Change to evaluation or data collecting agent.
            pyglet.clock.unschedule(agent.send_commands)
            if (symbol == key.E) and (key.E not in agents): agents[key.E] = EvaluationAgent(env)
            agent = agents[symbol]
            pyglet.clock.schedule_interval(agent.send_commands, 1.0 / env.unwrapped.frame_rate)

        env.render() # show image to user

    # Instantiate agent
    agents = {key.D: DataAgent(env)}
    initial = key.E if (len(sys.argv) > 1) and (sys.argv[1] == 'eval') else key.D
    if initial == key.E: agents[key.E] = EvaluationAgent(env)
    agent = agents[initial]
    # Call send_commands function from periodically (to simulate processing latency)
    pyglet.clock.schedule_interval(agent.send_commands, 1.0 / env.unwrapped.frame_rate)
    # Now run simulation forever (or until ESC is pressed)
    pyglet.app.run()
    # When it's done, close environment and exit program
    env.close()

if __name__ == '__main__':
    main()
