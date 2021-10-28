import numpy as np
from pypokerengine.api.game import start_poker
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from dl_config import my_setup_config
from dl_player import DLPlayer
from honest_player import MyHonestPlayer

PLAYER_ONE_NAME = 'HONEST_PLAYER'
DL_PLAYER_NAME = 'MODEL_PLAYER'


class DLPokerModel:
    def __init__(self, verbose=False):
        self.num_inputs = 15
        self.num_actions = 3
        self.num_hidden = 128
        self.max_rounds = 500
        self.gamma = 0.99
        self.eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

        self.inputs = layers.Input(shape=(self.num_inputs,))
        self.common = layers.Dense(self.num_hidden, activation="relu")(self.inputs)
        self.action = layers.Dense(self.num_actions, activation="softmax")(self.common)
        self.critic = layers.Dense(1)(self.common)

        self.model = keras.Model(inputs=self.inputs, outputs=[self.action, self.critic])
        self.config = my_setup_config(max_round=1, initial_stack=100, small_blind_amount=1)
        self.config.register_player(name=PLAYER_ONE_NAME, algorithm=MyHonestPlayer(name=PLAYER_ONE_NAME))

        self.critic_history = []
        self.rewards_history = []
        self.action_probs_history = []
        self.verbose = verbose

    def train(self, epochs=1000):
        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        huber_loss = keras.losses.Huber()
        running_reward = 0
        for epoch in range(epochs):
            print('STARTING EPOCH: {epoch}'.format(epoch=epoch))
            episode_reward = 0
            rounds_won = 0
            with tf.GradientTape() as tape:
                for round in range(1, self.max_rounds):
                    self.config.register_player(name=DL_PLAYER_NAME,
                                                algorithm=DLPlayer(model=self.model, critic_history=self.critic_history,
                                                                   rewards_history=self.rewards_history,
                                                                   action_probs_history=self.action_probs_history,
                                                                   verbose=self.verbose))
                    game_result = start_poker(self.config, verbose=0)
                    reward = self._get_reward(game_result['players'])
                    if reward > 0:
                        rounds_won += 1
                    episode_reward += reward
                    self.rewards_history.append(reward)
                    self.config.unregister_player(name=DL_PLAYER_NAME)

                # Update running reward to check condition for solving
                running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

                returns = []
                discounted_sum = 0
                for r in self.rewards_history[::-1]:
                    discounted_sum = r + self.gamma * discounted_sum
                    returns.insert(0, discounted_sum)

                # Normalize
                returns = np.array(returns)
                returns = (returns - np.mean(returns)) / (np.std(returns) + self.eps)
                returns = returns.tolist()

                # Calculating loss values to update our network
                history = zip(self.action_probs_history, self.critic_history, returns)
                actor_losses = []
                critic_losses = []
                for log_prob, value, ret in history:
                    # At this point in history, the critic estimated that we would get a
                    # total reward = `value` in the future. We took an action with log probability
                    # of `log_prob` and ended up recieving a total reward = `ret`.
                    # The actor must be updated so that it predicts an action that leads to
                    # high rewards (compared to critic's estimate) with high probability.
                    diff = ret - value
                    actor_losses.append(-log_prob * diff)  # actor loss

                    # The critic must be updated so that it predicts a better estimate of
                    # the future rewards.
                    critic_losses.append(
                        huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                    )

                # Backpropagation
                loss_value = sum(actor_losses) + sum(critic_losses)
                grads = tape.gradient(loss_value, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                # Clear the loss and reward history
                self.action_probs_history.clear()
                self.critic_history.clear()
                self.rewards_history.clear()

                if epoch % 10 == 0:
                    template = "running reward: {:.2f} at epoch {}"
                    print(template.format(running_reward, epoch))
                    print('Rounds won: {rounds_won}'.format(rounds_won=rounds_won))
                    print('Amount won: {episode_reward}'.format(episode_reward=episode_reward))
    def _get_reward(self, players):
        first_or_default = next((player for player in players if player['name'] == DL_PLAYER_NAME), None)
        if first_or_default is None:
            print('ERROR: COULD NOT FIND PLAYER RESULT')
            return 0
        return first_or_default['stack'] - 100


DLPokerModel(verbose=False).train(1000)
