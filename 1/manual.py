# MAC0318 Intro to Robotics
# Please fill-in the fields below with your info
#
# Name: Luis Vitor Pedreira Iten Zerkowski
# NUSP: 9837201
#
# ---
#
# Assignment 1 - Manual steering
#
# Task:
#  - Write a remote control Duckiebot
#
# Don't forget to run this from the Duckievillage root directory (example):
#   cd ~/MAC0318/duckievillage
#   conda activate duckietown
#   python3 assignments/manual/manual.py
#
# Submission instructions:
#  0. Add your name and USP number to the file header above.
#  1. Make sure that any last change haven't broken your code. If the code chrases without running you'll get a 0.
#  2. Submit this file via e-disciplinas.
#  3. Push changes to your git fork.import sys

import sys
import pyglet
from pyglet.window import key
from duckievillage import create_env

pwm_left, pwm_right = 0, 0

class Agent:
    # Agent initialization
    def __init__(self, environment):
        """ Initializes agent """
        # KeyStateHandler handles key states.
        key_handler = key.KeyStateHandler()
        # Let's register our key handler to the environment's key listener.
        environment.unwrapped.window.push_handlers(key_handler)
        self.env = environment
        self.key_handler = key_handler

    def send_commands(self, dt):
        ''' Agent control loop '''
        # This function updates the robot behaviour. Parameter dt is the elapsed time, in
        # milliseconds, since the last update call.
        # At each step, the agent produces an action in the form of two reals in [-1,1]:
        #   pwm_left, pwm_right = left motor power, right motor power
        # Play with these values and figure out how to make your own remote control duckiebot!
        
        #pwm_left, pwm_right = 0, 0

        global pwm_left
        global pwm_right

        # The key_handler object handles keyboard events. It's basically a map indexed by Pyglet
        # keys with values True if the key is being held, or False otherwise.
        if self.key_handler[key.D]:
            print('Acelera roda esquerda!')
            if abs(pwm_left) < 1:
            	pwm_left += 0.01
        if self.key_handler[key.U]:
            print('Acelera roda direita!')
            if abs(pwm_right) < 1:
            	pwm_right += 0.01
        if self.key_handler[key.C]:
            print('Desacelera roda esquerda!')
            if abs(pwm_left) < 1:
            	pwm_left -= 0.01
        if self.key_handler[key.K]:
            print('Desacelera roda direita!')
            if abs(pwm_right) < 1:
            	pwm_right -= 0.01

        # At each step, the environment may (or may not) change given your actions. Function step takes
        # as parameter the two motor powers as action and returns an observation (what the robot is
        # currently seeing), a reward (mostly used for reinforcement learning), whether the episode is
        # done (also used for reinforcement learning) and some info on the elapsed episode.  Let's ignore
        # return values for now.
        obs, reward, done, info = self.env.step(pwm_left, pwm_right)

        # Refresh at every update.
        self.env.render()

def main():
    print("MAC0318 - Assignment 1")
    # We'll use our version of Duckietown: Duckievillage. This environment will be where we'll run most
    # our tasks.
    env = create_env(
        raw_motor_input = True,
        seed = 101,
        map_name = 'loop_empty',
        draw_curve = False,
        draw_bbox = False,
        domain_rand = False,
        distortion = False,
        top_down = False
    )

    # Let's reset the environment to get our Duckiebot somewhere random.
    env.reset()
    # This function is used to draw the environment to a graphical user interface using Pyglet.
    env.render()

    # We use this function for on-press key events (not something we use for real-time feedback,
    # though). We'll register ESC as our way out of the Matrix.
    @env.unwrapped.window.event
    def on_key_press(symbol, modifiers):
        if symbol == key.ESCAPE:
            env.close()
            sys.exit(0)
        env.render()

    # Instantiante agent
    agent = Agent(env)
    # Call send_commands function from periodically (to simulate processing latency)
    pyglet.clock.schedule_interval(agent.send_commands, 1.0 / env.unwrapped.frame_rate)
    # Now run simulation forever (or until ESC is pressed)
    pyglet.app.run()
    env.close()

if __name__ == '__main__':
    main()
