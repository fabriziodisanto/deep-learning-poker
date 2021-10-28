from pypokerengine.api.game import setup_config, start_poker

from honest_player import MyHonestPlayer

config = setup_config(max_round=1, initial_stack=100, small_blind_amount=1)
player_one_name = 'HONEST_ONE'
player_two_name = 'HONEST_TWO'

config.register_player(name=player_one_name, algorithm=MyHonestPlayer(name=player_one_name))
config.register_player(name=player_two_name, algorithm=MyHonestPlayer(name=player_two_name))
for num in range(1):
    game_result = start_poker(config, verbose=1)
    print(game_result)


#         SB/D   BB
# PREFLOP  1      2
# POSTFLOP 2      1