import numpy as np
import pypokerengine.utils.visualize_utils as U
from pypokerengine.engine.card import Card
from pypokerengine.players import BasePokerPlayer
import torch
import tensorflow as tf


class DLPlayer(BasePokerPlayer):

    def __init__(self, model, critic_history, rewards_history, action_probs_history, verbose=False):
        self.model = model
        self.verbose = verbose
        self.critic_history = critic_history
        self.rewards_history = rewards_history
        self.action_probs_history = action_probs_history
        BasePokerPlayer.__init__(self)

    def declare_action(self, valid_actions, hole_card, round_state):
        if self.verbose:
            print(U.visualize_declare_action(valid_actions, hole_card, round_state, self.uuid))
        model_input = self._get_input_from_state(hole_card, round_state)
        action, amount = self._receive_action_from_model(valid_actions, model_input)
        return action, amount

    def receive_game_start_message(self, game_info):
        if self.verbose:
            print(U.visualize_game_start(game_info, self.uuid))
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        if self.verbose:
            print(U.visualize_round_start(round_count, hole_card, seats, self.uuid))
        pass

    def receive_street_start_message(self, street, round_state):
        if self.verbose:
            print(U.visualize_street_start(street, round_state, self.uuid))
        pass

    def receive_game_update_message(self, new_action, round_state):
        if self.verbose:
            print(U.visualize_game_update(new_action, round_state, self.uuid))
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        if self.verbose:
            print(U.visualize_round_result(winners, hand_info, round_state, self.uuid))
        pass

    def _receive_action_from_model(self, valid_actions, model_input):
        if self.verbose:
            print('INPUT: {input}'.format(input=str(model_input)))
        action_prob, critic = self.model(model_input)
        if self.verbose:
            print('ACTION_PROB: {action_prob}'.format(action_prob=str(action_prob)))
            print('CRITIC: {critic}'.format(critic=str(critic)))
        self.critic_history.append(critic)
        # todo if fail check what is action_prob
        action = np.random.choice(len(valid_actions), p=np.squeeze(action_prob))
        self.action_probs_history.append(tf.math.log(action_prob[0, action]))
        choice_action = valid_actions[action]
        amount = choice_action['amount'] if type(choice_action['amount']) is int else choice_action['amount']['min']
        if self.verbose:
            print('ACTION: {action}'.format(action=str(choice_action)))
            print('AMOUNT: {amount}'.format(amount=str(amount)))
        return choice_action['action'], amount

    @staticmethod
    def _get_card_id(card):
        return Card.from_str(card).to_id()

    def _get_input_from_state(self, hole_cards, state):
        result = []
        for card in hole_cards:
            result.append(DLPlayer._get_card_id(card))
        for card in state['community_card']:
            result.append(DLPlayer._get_card_id(card))

        while len(result) < 7:
            result.append(0)

        result.append(DLPlayer._get_aggression_value(self.uuid, state))
        return tf.expand_dims(tf.convert_to_tensor(result), 0)

    @staticmethod
    def _get_aggression_value(uuid, state):
        last_action_street = ''
        street = state['street']
        if street == 'showdown':
            last_action_street = 'river'
        if street == 'river':
            last_action_street = 'river' if 'river' in state['action_histories'] and state['action_histories'][
                'river'] is not None else 'turn'
        if street == 'turn':
            last_action_street = 'turn' if 'turn' in state['action_histories'] and state['action_histories'][
                'turn'] is not None else 'flop'
        if street == 'flop':
            last_action_street = 'flop' if 'flop' in state['action_histories'] and state['action_histories'][
                'flop'] is not None else 'preflop'
        if street == 'preflop':
            if 'flop' not in state['action_histories'] or state['action_histories']['preflop'] is None:
                # no hubo movimiento anterior
                return 0
            last_action_street = 'preflop'

        if len(state['action_histories'][last_action_street]) == 0:
            # no hubo movimiento anterior
            return 0

        last_action = state['action_histories'][last_action_street][-1]
        if last_action['uuid'] == uuid:
            last_action = state['action_histories'][last_action_street][-2]
        if last_action['amount'] == 0:
            # si ammount es 0 CHECKEO
            return 1
        if last_action['action'] == 'CALL':
            # rival PAGO
            return 2
        if last_action['action'] == 'RAISE':
            # rival SUBIO/APOSTO
            return 3
        print('COULD NOT GET OPP AGGRETION VALUE')
        return 0
