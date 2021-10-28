from pypokerengine.api.game import Config


class MyConfig(Config):

    def unregister_player(self, name):
        first_or_default = next((player for player in self.players_info if player['name'] == name), None)
        if first_or_default is not None:
            self.players_info.pop(self.players_info.index(first_or_default))


def my_setup_config(max_round, initial_stack, small_blind_amount, ante=0):
    return MyConfig(max_round, initial_stack, small_blind_amount, ante)
