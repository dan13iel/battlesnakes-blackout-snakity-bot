import numpy as np
from numba import jit
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import game
from game import generate_init_data, game_state_types

batch_sizes = np.unique(np.logspace(np.log10(500), np.log10(20_000), 50).astype(int))
repeats = 7
warmup_iters = 10
timed_iters = 100

results = []  # (batch_size, ns_per_game)

with tqdm(total=len(batch_sizes) * repeats) as pbar:
    for batch_size in batch_sizes:
        game.batch_size = batch_size

        for _ in range(repeats):
            game_state: game_state_types = {
                "data":     generate_init_data(batch_size),
                "flags":    np.full(batch_size // 2, np.uint8(0xFF), dtype=np.uint8),
                "food":     np.zeros((batch_size, 29), dtype=np.uint8),
                "length":   np.ones((batch_size, 4), dtype=np.uint8),
                "health":   np.full((batch_size, 4), np.uint8(100), dtype=np.uint8),
                "timestep": np.uint32(0),
            }

            m = np.empty((batch_size, 4), dtype=np.int64)
            moves = np.empty(batch_size, dtype=np.uint8)

            @jit(nopython=True)
            def random_moves(m, out):
                m[:] = np.random.randint(0, 4, (batch_size, 4))
                out[:] = (m[:, 0] | (m[:, 1] << 2) | (m[:, 2] << 4) | (m[:, 3] << 6)).astype(np.uint8)

            for _ in range(warmup_iters):
                random_moves(m, moves)
                game.step_wrapper(game_state, moves)

            start = time.time()
            for _ in range(timed_iters):
                random_moves(m, moves)
                game.step_wrapper(game_state, moves)
            end = time.time()

            ms_per_batch = (end - start) * 10
            ns_per_game = (ms_per_batch * 1_000_000) / batch_size
            results.append((batch_size, ns_per_game))

            pbar.update(1)

batch_arr = np.array([r[0] for r in results])
ns_arr = np.array([r[1] for r in results])

plt.scatter(batch_arr, ns_arr)
plt.xscale("log")
plt.xlabel("batch size")
plt.ylabel("ns per game")
plt.legend(["ns per game"])
plt.show()