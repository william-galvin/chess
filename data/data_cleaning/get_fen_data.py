#### Stores a list of fens and outcomes
import chess
import re
import random

board = chess.Board()
def getFENs (pgn: str): # -> list[str]
    board.reset()
    FENs = []
    moves = get_moves(pgn)
    for move in moves:
        board.push_san(move)
        FENs.append(board.fen())
    return FENs

def get_moves(pgn: str): # -> list[str]
    moves_string = pgn[pgn.index("1. "):]
    moves_string = moves_string[:moves_string.rindex("-")]
    moves_string = moves_string[: moves_string.rindex(" ")]
    moves_string = re.sub("[0-9]+\.", "", moves_string)
    return moves_string.split()

import time
import math

reader = open("lichess_elite_2021-11.pgn", "r")
writer = open("fen_data.csv", "w")

start = time.time()

games = reader.read()
games = games.split("\n\n")

result_key = {"0-1": 0, "1-0": 1, "1/2": .5}

for i in range(0, len(games) - 2, 2): 
  board.reset()

  meta = games[i]
  index = meta.index("Result \"") + len("Result \"")
  result = ""
  try:
    result = result_key[meta[index: index + 3]]
  except:
    print(f"key error on {meta[index: index + 3]}")
    continue

  if result == .5:
      continue
  
  pgn = games[i+1]
  try:
    moves = get_moves(pgn)
  except:
    print(f"get_moves() failed on {pgn}")
    continue
  eligible_positions = []

  for (j, move) in enumerate(moves[:-2]):
    board.push_san(move)
    if not 'x' in moves[j + 1] and j >= 8: 
      eligible_positions.append(board.fen())

  if len(eligible_positions) < 10:
    continue
  for position in random.sample(eligible_positions, 10):
    writer.write(f"{position}|{result}\n")

