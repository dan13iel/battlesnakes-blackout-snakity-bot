import numpy as np
from numba import jit
import game
from game import generate_init_data, game_state_types

batch_size = 850
game.batch_size = batch_size

game_state: game_state_types = {
    "data":     generate_init_data(batch_size),
    "flags":    np.full(batch_size // 2, np.uint8(0xFF), dtype=np.uint8),
    "food":     np.zeros((batch_size, 29), dtype=np.uint8),
    "length":   np.ones((batch_size, 4), dtype=np.uint8),
    "health": np.full((batch_size, 4), np.uint8(100), dtype=np.uint8),
    "timestep": np.uint32(0),
}

m = np.empty((batch_size, 4), dtype=np.int64)
moves = np.empty(batch_size, dtype=np.uint8)

@jit(nopython=True)
def random_moves(m, out):
    m[:] = np.random.randint(0, 4, (batch_size, 4))
    out[:] = (m[:, 0] | (m[:, 1] << 2) | (m[:, 2] << 4) | (m[:, 3] << 6)).astype(np.uint8)

for i in range(10):
    random_moves(m, moves)
    game.step_wrapper(game_state, moves)
for i in range(5_000):
    random_moves(m, moves)
    game.step_wrapper(game_state, moves)

m = np.empty((batch_size, 4), dtype=np.int64)
moves = np.empty(batch_size, dtype=np.uint8)
import time
dur = 100_000
start = time.time()
for game_num in range(int(dur/500)):
    game_state: game_state_types = {
        "data":     generate_init_data(batch_size),
        "flags":    np.full(batch_size // 2, np.uint8(0xFF), dtype=np.uint8),
        "food":     np.zeros((batch_size, 29), dtype=np.uint8),
        "length":   np.ones((batch_size, 4), dtype=np.uint8),
        "health": np.full((batch_size, 4), np.uint8(100), dtype=np.uint8),
        "timestep": np.uint32(0),
    }
    for i in range(500):
        random_moves(m, moves)
        game.step_wrapper(game_state, moves)
    print("\r", game_num, end='')

end = time.time()
print("nanoseconds per game:", (end-start)*1000/batch_size*1000*1000/dur)
print("total time (hours)", (end-start)/3600)

