'''
Script provided by Professor Lomuscio to generate the example used in the Reverify paper.
'''
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

'''
Code provided by Professor Lomuscio as described in https://arxiv.org/pdf/1706.07351.pdf
'''
def train_cartpole_nnet():
    #from test import CartPoleContEnv

    ENV_NAME = 'CartPole-v0'
    gym.undo_logger_setup()

    # Get the environment and extract the number of actions.
    env = gym.make(ENV_NAME)

    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n

    # Next, we build a very simple model.
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=60000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
          target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    dqn.fit(env, nb_steps=60000, visualize=False, verbose=2)

    # get model weights
    weights = model.get_weights()

    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=5, visualize=True)
    return weights


def weights_to_txt(weights, f):
    shape = weights.shape
    n_rows = shape[0]
    n_cols = shape[1]
    weights = np.reshape(weights, (n_cols, n_rows))
    for r in range(0,n_cols):
            for c in range(0, n_rows - 1):
                    f.write(str(weights[r,c])+", ")
            f.write(str(weights[r,c])+"\n")

def bias_to_txt(bias, f):
    for i in range(0,len(bias)):
        f.write(str(bias[i])+"\n")

'''
Convert trained cartpole nnet into a txt file that is compatible with the read_nnet function in util.jl
'''
def cartpole_nnet_to_txt():
    layers = train_cartpole_nnet()
    f = open("cartpole_nnet.txt", "w+")
    f.write("4\n")
    f.write("4, 16, 16, 16, 2\n")
    for i in range(5):
        f.write("0\n")
    for i,layer in enumerate(layers):
        if i % 2 == 0:
            weights_to_txt(layer, f)
        else:
            bias_to_txt(layer, f)

		
def main():
	cartpole_nnet_to_txt()

if __name__ == "__main__":
	main()
