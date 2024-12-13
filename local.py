from othello.OthelloGame import OthelloGame
from othello.bots.Random import BOT
from othello.bots.AlphaBeta import AlphaBetaBot

game = OthelloGame(n=6)
random_bot = BOT()
alpha_beta_bot = AlphaBetaBot()
game.play(black=random_bot, white=alpha_beta_bot, verbose=True)