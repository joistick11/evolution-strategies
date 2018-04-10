import gym
from gym import wrappers
from keras.models import Sequential
import numpy as np
from copy import copy
from keras.layers import Dense
import heapq
import random


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

num_workers = 100

# Сгенерируем начальную популяцию
population = []
for i in range(0, num_workers):
    population.append(randomize_weights(model.get_weights()))

best_result = []
best_ind = -1
try:
    while True:

        all_weights = []
        all_rewards = []

        # Мутируем 25 случаных членов особи
        to_mutate = random.sample(range(num_workers), 25)
        for ind in to_mutate:
            population[ind] = randomize_weights(population[ind])

        for ind in population:
            model.set_weights(ind)

            current_state = env.reset()
            overall_reward = 0
            while True:
                predicted = model.predict(np.array([current_state]))
                # env.render()
                current_state, reward, done, inf = env.step(np.argmax(predicted))
                overall_reward += reward

                if done:
                    break
            all_rewards.append(overall_reward)

        # Возьмём 75 лучших агентов
        maxItemIndexes = heapq.nlargest(75, range(len(all_rewards)), all_rewards.__getitem__)
        best_result = population[maxItemIndexes[0]]
        best_ind = maxItemIndexes[0]

        new_population = []
        for ind in maxItemIndexes:
            new_population.append(population[ind])

        # От них размножим ещё 25
        for i in range(0, 25):
            # выберем два случаных родителя
            parents = random.sample(range(75), 2)

            # и сольём их случайным образом
            child = copy(new_population[0])
            for index, curr in enumerate(new_population[0]):
                for i in range(0, len(curr)):
                    if isinstance(curr[i], list):
                        for j in range(0, len(curr[i])):
                            dec = np.random.randint(0, 2)
                            child[index][i][j] = new_population[parents[dec]][index][i][j]
                    else:
                        dec = np.random.randint(0, 2)
                        child[index][i] = new_population[parents[dec]][index][i]
            new_population.append(child)

        population = new_population

        print("max reward " + str(np.max(all_rewards)))
        print("god damn good model " + str(best_ind))
        print("average reward " + str(np.mean(all_rewards)))
        print("//=======================================//")
        print("\n")
except KeyboardInterrupt:
    import pickle
    print(best_result[0])
    print(best_result[2])
    print(best_result[5])
    print("FALLING. Best: " + str(best_ind))
    pickle.dump(best_result, open("best_weights.pkl", "wb"))






