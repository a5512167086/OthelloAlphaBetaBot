import numpy as np
from othello.OthelloUtil import getValidMoves

class BOT():
    def __init__(self, *args, **kargs):
        pass
    
    def getAction(self, game, color):
        valids=getValidMoves(game, color) # 取得合法步列表
        position=np.random.choice(range(len(valids)), size=1)[0] # 隨機選擇其中一合法步
        position=valids[position]
        return position
