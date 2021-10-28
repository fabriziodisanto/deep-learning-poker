from examples.players.honest_player import HonestPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate


class MyHonestPlayer(HonestPlayer):

    def __init__(self, name):
        self.name = name
        self.simulation = 100
        HonestPlayer.__init__(self)

    def receive_round_start_message(self, round_count, hole_card, seats):
        # print('{name} hole cards are: {cards}'.format(name=self.name, cards=hole_card))
        HonestPlayer.receive_round_start_message(self, round_count, hole_card, seats)

    def declare_action(self, valid_actions, hole_card, round_state):
        community_card = round_state['community_card']
        win_rate = estimate_hole_card_win_rate(
            nb_simulation=self.simulation,
            nb_player=self.nb_player,
            hole_card=gen_cards(hole_card),
            community_card=gen_cards(community_card)
        )
        print('{name} win rate is: {win_rate}'.format(name=self.name, win_rate=win_rate))
        if win_rate >= 0.7:
            action = valid_actions[2]
            action['amount'] = action['amount']['min']
        elif win_rate >= 1.0 / self.nb_player:
            action = valid_actions[1]
        else:
            action = valid_actions[1]
        print('{name} action {action}, amount {amount}'.format(name=self.name, action=action['action'],
                                                               amount=action['amount']))
        return action['action'], action['amount']
