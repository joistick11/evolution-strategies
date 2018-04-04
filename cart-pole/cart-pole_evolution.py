import gym
from gym import wrappers
from keras.models import Sequential
import numpy as np
from copy import deepcopy
from keras.layers import Dense
import heapq

def randomize_weights(weights):
    new_weights = []

    for curr in weights:
        epsilon = np.random.normal(0, 1, curr.shape)
        new_weights.append(curr + epsilon)

    return new_weights

gym.envs.register(
    id='CartPole-v12',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=400
)
env = gym.make('CartPole-v12')
env = wrappers.Monitor(env, 'cart-pole-results', force=True)

model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='softmax'))

#alpha = 0.005
num_workers = 5

previous_reward = 0
try:
    while True:

        all_weights = []
        all_rewards = []

        # Рандомизируем модель 5 раз и получим результаты
        for i in range(0, num_workers):
            inner_model = deepcopy(model)
            inner_weights = randomize_weights(model.get_weights())
            inner_model.set_weights(inner_weights)

            all_weights.append(inner_weights)

            current_state = env.reset()
            overall_reward = 0
            while True:
                predicted = inner_model.predict(np.array([current_state]))
                # env.render()
                current_state, reward, done, inf = env.step(np.argmax(predicted))
                overall_reward += reward

                if done:
                    break
            all_rewards.append(overall_reward)

        print("Rewards on current cycle " + str(all_rewards))

        # Вычислим новые веса модели, где каждая рандомизировання модель сделает вклад, пропорциональный полученной награде
        maxTwoItemsIndexes = heapq.nlargest(2, range(len(all_rewards)), all_rewards.__getitem__)

        mean = np.mean([all_rewards[maxTwoItemsIndexes[0]], all_rewards[maxTwoItemsIndexes[1]]]) - 25
        if mean < previous_reward:
            continue

        previous_reward = np.mean([all_rewards[maxTwoItemsIndexes[0]], all_rewards[maxTwoItemsIndexes[1]]]) - 25

        new_weights = deepcopy(model.get_weights())
        for index, curr in enumerate(model.get_weights()):
            for i in range(0, len(curr)):
                if isinstance(curr[i], list):
                    for j in range(0, len(curr[i])):
                        dec = np.random.randint(0, 2)
                        new_weights[index][i][j] = all_weights[maxTwoItemsIndexes[dec]][index][i][j]
                else:
                    dec = np.random.randint(0, 2)
                    new_weights[index][i] = all_weights[maxTwoItemsIndexes[dec]][index][i]

        model.set_weights(new_weights)

        print("max reward " + str(np.max(all_rewards)))
        #print("Get only " + str(need_to_process))
        print("New prev reward " + str(previous_reward))
        print("//=======================================//")
        print("\n")
except KeyboardInterrupt:
    import pickle
    pickle.dump(model.get_weights(), open("out.pkl", "wb"))






