import random 
import numpy as np

random.seed(1)

def create_board():
    return np.zeros((3,3), dtype = int)

def possibilities(board):
    available_positions = []
    for i in range(3):
        for j in range(3):
            if board[i,j] ==0:
                available_positions.append((i,j))
    return available_positions
    
def place(board, player, position):
    if board[position] == 0:
        board[position] = player
    return board
    
def random_place(board, player):
    selection = possibilities(board)
    position = random.choice(selection)
    board[position] = player
    return board

def row_win(board, player):
    if np.all(board == player,axis=1).any():
        return True
    else:
        return False

def col_win(board, player):
    if np.all(board == player,axis=0).any():
        return True
    else:
        return False


def diag_win(board, player):
    if np.diagonal(board == player).all() or np.fliplr(board==player).diagonal().all():
        return True
    else:
        return False

def evaluate(board):
    winner = 0
    for player in [1, 2]:
        if row_win(board,player) is True or col_win(board,player) is True or diag_win(board,player) is True:
            winner = player
            break
        else:
            pass
 
    return winner

def play_game():
    player = 2
    board = play_strategic_game()
    for _ in range(8):
        board = random_place(board, player)
        if row_win(board, player) or col_win(board, player) or diag_win(board, player):
            return player

        player = 1 if player == 2 else 2
    return 0
    
def play_strategic_game():
    board = create_board()
    board = place(board, 1, (1,1))
    return board
    
results = [play_game() for _ in range(1000)]
p1=results.count(1)
p2=results.count(2)
draw=results.count(0)
print(f" Player 1 won {p1} games\n Player 2 won {p2} games\n Drew {draw} games")
