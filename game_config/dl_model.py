from pypokerengine.api.game import start_poker
from tensorflow import keras
from tensorflow.keras import layers

from dl_config import my_setup_config
from dl_player import DLPlayer
from honest_player import MyHonestPlayer

PLAYER_ONE_NAME = 'HONEST_PLAYER'
DL_PLAYER_NAME = 'MODEL_PLAYER'


class DLPokerModel:
    def __init__(self, verbose=False):
        self.num_inputs = 8
        self.num_actions = 3
        self.num_hidden = 128
        self.max_rounds = 10
        self.gamma = 0.99

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

    def train(self, epochs=10):
        for epoch in range(epochs):
            print('========== STARTING EPOCH: {epoch} =========='.format(epoch=epoch))
            episode_reward = 0
            for round in range(1, self.max_rounds):
                self.config.register_player(name=DL_PLAYER_NAME,
                                            algorithm=DLPlayer(model=self.model, critic_history=self.critic_history,
                                                               rewards_history=self.rewards_history,
                                                               action_probs_history=self.action_probs_history,
                                                               verbose=self.verbose))
                game_result = start_poker(self.config, verbose=0)
                print('AFTER PLAY, CRITIC HISTORY LENGTH: {}'.format(str(len(self.critic_history))))
                print('AFTER PLAY, ACTIONS PROBS HISTORY LENGTH: {}'.format(str(len(self.action_probs_history))))
                reward = self._get_reward(game_result['players'])
                self.rewards_history.append(reward)
                self.config.unregister_player(name=DL_PLAYER_NAME)
            returns = []
            discounted_sum = 0
            for r in self.rewards_history[::-1]:
                discounted_sum = r + self.gamma * discounted_sum
                returns.insert(0, discounted_sum)
            # hacer cosas con returns y episode_reward, empezar a mejorar el modelico

    def _get_reward(self, players):
        first_or_default = next((player for player in players if player['name'] == DL_PLAYER_NAME), None)
        if first_or_default is None:
            print('ERROR: COULD NOT FIND PLAYER RESULT')
            return 0
        return first_or_default['stack'] - 100


DLPokerModel(verbose=True).train()
