import numpy as np
from numba import jit, njit, uint32
import numpy.typing as npt
from typing import TypedDict
import itertools

batch_size = None
food_spawn_chance = 25
min_food = 1

'''
ALGORITHM

conventions (apply everywhere):
    moves: 0,1,2,3 -> up,down,left,right. y+ is down, so "up" is y-1.
    ring slots: snake sk's head at time t lives at data[g][sk][t % 225].
    body segment i (0 = tail) is at (t - L + 1 + i) % 225. no head pointer needed.

    array shapes are NOT consistent, sorry:
        snakes_died, death_cause -> (4, game_count)
        food_to_add              -> (game_count, 4)

    numba adds module-level globals (food_spawn_chance, min_food) in at compile time, and cache=True persists that across runs. 
    changing them after the first compile might do nothing until you clear the cache. 
    batch_size is a module global that step_wrapper reads. set it before calling.

the data:
    for each of n parallel games:
        data[game][snake][ring_slot] -> packed uint32 cell
            bits: x(4) | y(4) | age(8) | timestep(10) | snake_id(2)
            x=0xF means that cell is empty so no snakity
            age and timestep are WRITE-ONLY in here - step never reads them back.
                they exist for whatever consumes the buffer downstream. the
                timestep field wraps at 1024, way past the 225 ring period, so
                don't trust it for ordering.
        flags[game] -> 4-bit alive mask per snake, 2 games per byte.
            game g -> byte g>>1, nibble shift (g&1)*4 (even game = low nibble)
        food[game] -> 225-bit packed grid, 29 bytes incl. 7 bits of padding.
            LSB-first: bit for pos is (food[g][pos>>3] >> (pos&7)) & 1.
            note pack_single_flags uses the OPPOSITE bit order (MSB-first). yes really.
        length[game][snake]
        health[game][snake]
        timestep -> global counter, shared ring-buffer index for all snakes that way
            u dont need a 50k long array of pointers to snake head locations bc
            thats hard to vectorise

step func
1. determine GAME_OVER mask:
    a game is over if its alive-mask has 0 or 1 bits set

2. age every occupied snake cell by +1 (skip game-over games)

3. decrement health for every alive snake:
    health already 0 -> die without decrementing
    else health-- and die if it just hit 0
    both paths set death_cause = 1 (starved). the snake is NOT removed here, it
    just carries snakes_died forward into step 7.

4. unpack alive flags into per-snake boolean array (flag_arr)

5. build occupancy board (15*15) for this timestep:
    for each alive snake, walk exactly length slots back from the head:
        board_snake_id[pos] = snake, board_ring_idx[pos] = ring_slot
    slots holding the 0xF sentinel are skipped - length and ring contents can
    disagree, and 0xF flattens to index 240 and corrupts the heap if you don't.
    at spawn all 3 segments sit on the same cell, so board_ring_idx ends up holding
    the HEAD's slot rather than the tail

6. calculate moves:
    for each snake with flag_arr set (starved snakes from step 3 included their moves, that gets filtered next step tho):
        read head cell at data[g][sk][timestep % 225]
        if head is the 0xF sentinel -> body already cleared while still flagged alive, so it gets marked dead and skipped.
        out of bounds -> set boundary_dead (NOT snakes_died - converted in step 7)
        else compute new_head_pos
        record collision bitmap.

7. collide physics, per snake, sequentially (snake 0 fully resolves before 1, etc):
    boundary_dead -> snakes_died, cause 2
    head-to-head, for snakes sharing the target cell (already-dead ones ignored):
        longest wins; anyone shorter dies; if you tie with the longest, both of
        you die. cause 3.
    otherwise check the occupant of the target cell in board_snake_id. because the
    board is mutated as we go, who is on that cell depends on processing order:
        occupant == self -> safe only if it's exactly your own vacating tail and
            you're not eating, else cause 4
        occupant j < sk  -> j already moved, so its tail is already cleared from
            the board. anything still sitting there is real body -> cause 4
        occupant j > sk  -> j hasn't moved yet, so its tail is still on the board
            and about to leave. safe if that cell is j's tail and j isn't eating,
            else cause 4
    this forces the same thing to happen regardless of snake update order btw
    then, in this order:
        write the new head into ring slot new_ring
        if not eating: clear the tail slot ((timestep - length + 1) % 225, using
            the OLD length - step 8 hasn't grown it yet) and free its board cell
        update board_snake_id / board_ring_idx for the new head position

8. eat, for snakes alive, NOT dead this tick, and landed on food:
    clear food bit, length++, health = 100, food_to_add[game][snake] = 1
    (dying on a food square gets you nothing)

9. remove dead snakes: overwrite all 225 slots with the empty sentinel (snake_id preserved) for any snake marked dead

10. rebuild each game's alive nibble from flag_arr & ~snakes_died

returns snakes_died (4, n), food_to_add (n, 4), death_cause (4, n)
    death_cause: 0 alive, 1 starved, 2 wall, 3 head-to-head, 4 body, 5 invalid state

numba hates half of numpy and i was tired so seperate func for this:
spawn_food, called AFTER step and AFTER timestep++ (so dead snakes are already
cleared and their cells count as free, which matches the real server):
    for each active (non-game-over) game:
        unpack occupied cells - union of ALL 225 slots per snake, not just the
            first `length`. different rule to step 5, same answer as long as the
            ring is consistent
        unpack current food bits, count existing food
        roll 1..100; skip this game if roll > food_spawn_chance AND food_count >= min_food. i.e. spawn on a successful roll, or always when below the minimum
        compute free cells (not occupied, not already food), pick one uniformly, set its food bit
    returns placed (n, 2) -> x,y of the food spawned this tick, (255,255) if none.
        step_wrapper stores it as game_state['new_apple'].
'''

# data in each snake cell:
# bits 0-3: X (0xF=empty), 4-7: Y, 8-15: age (0=head), 16-25: timestep, 26-27: snake_id
X_FIELD = (np.uint32(0),  np.uint32(0xF))
Y_FIELD = (np.uint32(4),  np.uint32(0xF))
AGE_FIELD = (np.uint32(8),  np.uint32(0xFF))
TIMESTEP_FIELD = (np.uint32(16), np.uint32(0x3FF))
SNAKE_ID_FIELD = (np.uint32(26), np.uint32(0x3))

SPAWN_POINTS = [(1,1),(1,13),(13,1),(13,13),(1,7),(7,1),(7,13),(13,7)]

@jit(nopython=True)
def get_field(data, shift, mask):
    return (data >> shift) & mask

@njit(uint32(uint32, uint32, uint32, uint32))
def set_field(data, shift, mask, value):
    return (data & ~(mask << shift)) | ((uint32(value) & mask) << shift)

@jit(nopython=True)
def is_multi_snake(val):
    return val != np.uint8(0xFF) and (val & (val - np.uint8(1))) != np.uint8(0)

@jit(nopython=True)
def get_flags(flags, index):
    byte_side = index & 1 # odd or even
    shift = np.uint8(4 * byte_side)
    return (flags[index >> 1] >> shift) & np.uint8(0xF) # mask to nibblle and bit shift

@jit(nopython=True)
def xy_to_idx(x, y):
    return y * np.uint32(15) + x

@jit(nopython=True)
def idx_to_xy(idx):
    return idx % np.uint32(15), idx // np.uint32(15)

class game_state_types(TypedDict):
    data:     npt.NDArray[np.uint32] # (batch, 4, 225)
    flags:    npt.NDArray[np.uint8] # (batch//2,) packed 4-bit alive masks
    food:     npt.NDArray[np.uint8] # (batch, 29) packed bits for 225 positions
    length:   npt.NDArray[np.uint8] # (batch, 4) snake lengths
    health:   npt.NDArray[np.uint8]
    timestep: np.uint32 # global step counter

def build_luts(): # look up table for rotations (x,y) - > (x,y) so trained snake only sees from one perspective (so parameters arent wasted on generalising that)
    idx = np.arange(225, dtype=np.uint16)
    x, y = idx % 15, idx // 15
    c14 = np.uint16(14)
    rotation_coord_lut = np.stack([
        y         | ((c14 - x) << 8),
        (c14 - x) | ((c14 - y) << 8),
        (c14 - y) | (x         << 8),
    ]).astype(np.uint16)
    nx = (rotation_coord_lut & 0xFF).astype(np.uint16)
    rotation_index_lut = ((rotation_coord_lut >> 8) * 15 + nx).astype(np.uint8)
    return rotation_coord_lut, rotation_index_lut

rotation_coord_lut, rotation_index_lut = build_luts()

def remap_canonical(data, food, timestep, out_data, out_food, apple_xy=None, out_apple_xy=None, out_rot=None, in_rot=None):
    # btw only roates everything for seat 0
    # out_data/out_food are caller-allocated output buffers, same shape/dtype as
    # data/food. We never write into data/food themselves - this leaves the real
    # game_state untouched and only produces a rotated *view* for the model.
    np.copyto(out_data, data)
    np.copyto(out_food, food)

    # head pos: ring slot is timestep-based (same rule the step function uses),
    # not a fixed slot - the head moves around the ring buffer every tick.
    head_ring = int(timestep % 225)
    hx = data[:, 0, head_ring] & np.uint32(0xF)
    hy = (data[:, 0, head_ring] >> np.uint32(4)) & np.uint32(0xF)

    # figure out rotation for each thing
    # canonical orientation has head at (7,1) which is looking up in center col
    # the other 3 cardinal mid-edge positions each need a cw rotation to reach canonical.
    if in_rot is not None:
        # caller decides the rotation (e.g. from the snake's current heading, so the
        # canonical view holds for the whole episode rather than only on tick 0).
        rot = in_rot.astype(np.int8, copy=False)
    else:
        rot = np.zeros(len(data), dtype=np.int8)
        rot[(hx == 13) & (hy == 7)] = 1 # right -> 90deg CW
        rot[(hx == 7)  & (hy == 13)] = 2 # down -> 180deg CW
        rot[(hx == 1)  & (hy == 7)] = 3 # left -> 270deg CW

    if out_rot is not None:
        np.copyto(out_rot, rot)

    if apple_xy is not None:
        # apple_xy comes in as raw game-frame (x, y), same frame as data/food before
        # rotation - it needs the same per-game rot applied so it lines up with
        # out_data/out_food/board_buf etc, which are all in canonical frame.
        np.copyto(out_apple_xy, apple_xy)
        sentinel = apple_xy[:, 0] == np.uint8(255) # (255,255) marks "no tracked apple"
        for r in range(1, 4):
            mask = (rot == r) & ~sentinel
            if not np.any(mask):
                continue
            ax = apple_xy[mask, 0].astype(np.uint16)
            ay = apple_xy[mask, 1].astype(np.uint16)
            old_idx = (ay * np.uint16(15) + ax).astype(np.int32)
            lv = rotation_coord_lut[r - 1][old_idx]
            out_apple_xy[mask, 0] = (lv & np.uint16(0xFF)).astype(np.uint8)
            out_apple_xy[mask, 1] = ((lv >> 8) & np.uint16(0xFF)).astype(np.uint8)

    for r in range(1, 4):
        mask = rot == r
        if not np.any(mask):
            continue
        d = data[mask]

        # get x from bits 0-3; sentinel 0xF marks an empty/unused segment slot
        x = (d & np.uint32(0xF)).astype(np.uint16)
        empty = x == np.uint16(0xF)

        # Convert packed (x, y) to a flat grid index for LUT lookup; clamp empty slots to 0
        old_idx = (((d >> np.uint32(4) & np.uint32(0xF)) * ~empty) * np.uint32(15) + x * ~empty).astype(np.int32)

        # Look up rotated coordinates: low byte = new x, high byte = new y
        lv = rotation_coord_lut[r - 1][old_idx]
        nx_ = lv.astype(np.uint32) & np.uint32(0xFF)
        ny_ = (lv.astype(np.uint32) >> 8) & np.uint32(0xFF)

        # write rotated (x, y) into the output buffer without changing anyhting else
        out_data[mask] = np.where(empty, d, (d & ~np.uint32(0xFF)) | nx_ | (ny_ << np.uint32(4)))

        # remap the food bit board as well (also its 29 bytes SINCE NUMPY LOVES TO STORE 1 BOOL BIT IN 1 BYTE)
        bits = np.unpackbits(food[mask], axis=1, bitorder='little')[:, :225] # unpack and strip 7 padding bits
        new_bits = np.zeros_like(bits)
        new_bits[:, rotation_index_lut[r - 1]] = bits # relocate bits to rotated positions
        padded = np.zeros((len(bits), 232), dtype=np.uint8) # re-pad to byte boundary (232 = 29*8)
        padded[:, :225] = new_bits
        out_food[mask] = np.packbits(padded, axis=1, bitorder='little')

def pack_cell(x=0xF, y=0xF, age=0, timestep=0): # store a bunch of data in a uint32 which is on a per cell basis
    val = np.uint32(0)
    val = set_field(val, *X_FIELD, x)
    val = set_field(val, *Y_FIELD, y)
    val = set_field(val, *AGE_FIELD, age)
    val = set_field(val, *TIMESTEP_FIELD, timestep)
    return val

@njit("void(uint8[::1], uint8[::1], uint8[::1])", cache=True)
def pack_single_flags(flags, packed, indices):
    packed[:] = np.uint8(0)
    for i in range(indices.shape[0]):
        n = int((flags[i >> 1] >> np.uint8(4 * (i & 1))) & np.uint8(0xF))
        if not (n & (n - 1)):
            packed[i >> 3] |= np.uint8(0x80) >> np.uint8(i & 7)
            if   n == 0:  indices[i] = 5
            elif n >= 8:  indices[i] = 0
            elif n >= 4:  indices[i] = 1
            elif n >= 2:  indices[i] = 2
            else:         indices[i] = 3

def generate_init_data(n):
    # all 24 permutations of the 4 cardinal mid-edge starting positions
    configs = np.array(list(itertools.permutations([(7,1),(7,13),(1,7),(13,7)])), dtype=np.uint32)
    # cycle through permutations deterministically across n games
    heads = configs[np.arange(n) % len(configs)]

    x_shift, x_mask = X_FIELD
    y_shift, y_mask = Y_FIELD
    s_shift, s_mask = SNAKE_ID_FIELD

    # shape (1, 4, 1) so it broadcasts over (n, 4, 225)
    snake_ids = (np.arange(4, dtype=np.uint32).reshape(1, 4, 1) & s_mask) << s_shift

    # all 225 segment slots start as empty sentinel, with snake ID embedded
    data = np.full((n, 4, 225), pack_cell(), dtype=np.uint32) | snake_ids

    # snakes start at length 3, all 3 segments stacked on the spawn cell.
    # at timestep=0 the ring slots occupied by a length-L snake are
    # (0 - L + 1 + i) % 225 for i in 0..L-1, so slots 223, 224, 0.
    start_len = 3
    ring_slots = [(-start_len + 1 + i) % 225 for i in range(start_len)]
    cell_vals = (
        ((heads[..., 0] & x_mask) << x_shift) |
        ((heads[..., 1] & y_mask) << y_shift) |
        snake_ids[:, :, 0]
    )
    for ri in ring_slots:
        data[:, :, ri] = cell_vals
    return data

@jit(nopython=True, cache=True)
def spawn_food(data, flags, food, n):
    placed = np.full((n, 2), np.uint8(255), dtype=np.uint8)

    for g in range(n):
        # flags packs two games per byte as two nibbles; extract this games nibble
        byte = flags[g >> 1]
        nibble = (byte >> np.uint8((g & 1) * 4)) & np.uint8(0xF)

        # 4 bits per nibble, each one is a flag for either snake alive or dead
        # dark magic checks if there is only one bit set to 1
        if (nibble & (nibble - np.uint8(1))) == np.uint8(0):
            continue

        # 1s and 0s saying if a snake is there
        occupied = np.zeros(225, dtype=np.bool_)
        for sk in range(4):
            for ri in range(225):
                cell = data[g, sk, ri]
                x = int(cell & np.uint32(0xF))
                y = int((cell >> np.uint32(4)) & np.uint32(0xF))
                if x != 0xF:
                    occupied[y * 15 + x] = True

        # 29 packed bytes -> 225 bool bits to preserve sanity
        food_bits = np.zeros(225, dtype=np.bool_)
        for pos in range(225):
            if (food[g, pos >> 3] >> np.uint8(pos & 7)) & np.uint8(1):
                food_bits[pos] = True

        food_count = np.int32(0)
        for pos in range(225):
            if food_bits[pos]:
                food_count += np.int32(1)

        # spawn with probability food_spawn_chance%, ALWAYS spawn if below min_food.
        # (previous condition was inverted: it skipped on roll <= chance, i.e. spawned
        # ~75% of ticks. Verify the real server's chance -- vanilla battlesnake is 15.)
        roll = int(np.random.randint(np.int32(1), np.int32(101)))
        if roll > food_spawn_chance and food_count >= min_food:
            continue

        # count no snake/food cells and choose a random one
        free_count = np.int32(0)
        for pos in range(225):
            if not occupied[pos] and not food_bits[pos]:
                free_count += np.int32(1)

        if free_count == np.int32(0):
            continue

        pick = int(np.random.randint(np.int32(0), free_count))
        idx = np.int32(0)
        for pos in range(225):
            if not occupied[pos] and not food_bits[pos]:
                if idx == pick:
                    food[g, pos >> 3] |= np.uint8(1) << np.uint8(pos & 7)
                    placed[g, 0] = np.uint8(pos % 15) # x
                    placed[g, 1] = np.uint8(pos // 15) # y
                    break
                idx += np.int32(1)

    return placed

#basic_game_state: game_state_types = {
# "data":      generate_init_data(batch_size),          # snake segment positions, (n, 4, 225) uint32
# "flags":     np.full(batch_size // 2, np.uint8(0xFF), dtype=np.uint8),  # alive-snake nibbles; 0xF = all 4 alive
# "food":      np.zeros((batch_size, 29), dtype=np.uint8),  # bit-packed 15×15 food grid (225 bits + 7 pad)
# "length":    np.full((batch_size, 4), np.uint8(3), dtype=np.uint8),  # segment count per snake
# "health":    np.full((batch_size, 4), np.uint8(100), dtype=np.uint8),
# "timestep":  np.uint32(0),
# "new_apple": np.full((batch_size, 2), np.uint8(255), dtype=np.uint8),  # x,y of food spawned this tick; 255,255 if none
#}

@jit(nopython=True, cache=True)
def step(data, flags, food, moves, length, health, timestep, game_count):
    """
    data       : (game_count, 4, 225) uint32
    flags      : (game_count//2,) uint8, is 2 nibbles/byte, 4-bit alive flags for each snake for each game
    food       : (game_count, 29) uint8, is packed bits, 225 positions
    moves      : (game_count,)uint8 is 2 bits/snake: snake k @ bits [2k,2k+1], 0,1,2,3 -> up down left right (y grows downward)
    length     : (game_count, 4) uint8, modified in-place
    timestep   : uint32
    returns (snakes_died (4,game_count) bool and food_to_add (game_count,4) uint8)
    """
    new_ring = int((timestep + 1) % 225) #values for ring buffer that snake data gets stored in so no need for a seperate pointer for head pos
    cur_ring = int(timestep % 225)

    flag_arr = np.zeros((4, game_count), dtype=np.bool_)
    snakes_died = np.zeros((4, game_count), dtype=np.bool_)
    # 0 = alive, 1 = starved, 2 = wall, 3 = head-to-head, 4 = body, 5 = invalid state (thankfully never happens from my testing)
    death_cause = np.zeros((4, game_count), dtype=np.uint8)

    # first step build game_over mask
    # (n & (n-1)) == 0 is true when n has at most one bit set,
    # covering both all-dead (n=0) and single-winner (n=power-of-2) situations.
    # it also black magic
    game_over = np.zeros(game_count, dtype=np.bool_)
    for i in range(game_count >> 1):
        byte = flags[i]
        lo = byte & np.uint8(0xF)
        hi = byte >> np.uint8(4)
        game_over[i * 2] = (lo & (lo - np.uint8(1))) == np.uint8(0)
        game_over[i * 2 + 1] = (hi & (hi - np.uint8(1))) == np.uint8(0)

    # age++ for all snake cells
    # vectorised numpy is slower for this then a sequential loop by like 10x :sob:
    for g in range(game_count):
        if game_over[g]:
            continue
        for sk in range(4):
            for ri in range(225):
                cell = data[g, sk, ri]
                if cell & np.uint32(0xF) != np.uint32(0xF):
                    data[g, sk, ri] = cell + np.uint32(0x100)

    # health-- for all alive snakes
    for g in range(game_count):
        if game_over[g]:
            continue
        nibble = (flags[g >> 1] >> np.uint8((g & 1) * 4)) & np.uint8(0xF)
        for sk in range(4):
            if (nibble >> np.uint8(sk)) & np.uint8(1):
                if health[g, sk] == np.uint8(0):
                    snakes_died[sk, g] = True # caught below in section 7
                    death_cause[sk, g] = np.uint8(1)
                else:
                    health[g, sk] -= np.uint8(1)
                    if health[g, sk] == np.uint8(0):
                        snakes_died[sk, g] = True
                        death_cause[sk, g] = np.uint8(1)

    # unpack snake alive/dead flags
    for g in range(game_count):
        if game_over[g]: # easier then queueing and removing completed games each step
            continue
        nibble = (flags[g >> 1] >> np.uint8((g & 1) * 4)) & np.uint8(0xF) # nibble nobble
        for sk in range(4):
            flag_arr[sk, g] = bool((nibble >> np.uint8(sk)) & np.uint8(1))

    # create 15x15 flattened boards of data for each game for game updates
    board_snake_id = np.full((game_count, 225), np.uint8(0xFF), dtype=np.uint8)
    board_ring_idx = np.zeros((game_count, 225), dtype=np.uint8)
    for g in range(game_count):
        if game_over[g]:
            continue # ignore completed games
        for sk in range(4):
            if not flag_arr[sk, g]:
                continue
            L = int(length[g, sk])
            for i in range(L): # bitshift and index into the uin32 thingies
                ri = (int(timestep) - L + 1 + i) % 225 # use timestep instead of dedicated pointer
                cell = data[g, sk, ri]
                # masking to get x and y values
                x = int(cell & np.uint32(0xF))
                if x == 0xF:
                    continue # empty sentinel: flattens to 240 and corrupts the heap
                y = int((cell >> np.uint32(4)) & np.uint32(0xF))
                board_snake_id[g, y * 15 + x] = np.uint8(sk)
                board_ring_idx[g, y * 15 + x] = np.uint8(ri)

    # do stuff: new head pos's, boundary kys's, new_heads grid
    new_head_pos = np.zeros((4, game_count), dtype=np.uint16)
    boundary_dead = np.zeros((4, game_count), dtype=np.bool_)
    new_heads = np.full((game_count, 225), np.uint8(0xFF), dtype=np.uint8)
    is_eating = np.zeros((4, game_count), dtype=np.bool_)
    for sk in range(4): # for each snake (pigeon)
        for g in range(game_count): # g for game (pigeon...)
            if game_over[g]:
                continue
            if not flag_arr[sk, g]:
                continue
            head_cell = data[g, sk, cur_ring] #cur ring is based on timestep
            hx = np.int32(head_cell & np.uint32(0xF))
            if hx == np.int32(0xF):
                # flagged alive but body already cleared. hy is 15 here, so the
                # boundary test (which compares against 14) lets it through and the
                # resulting pos indexes new_heads/board out of range.
                snakes_died[sk, g] = True
                if death_cause[sk, g] == np.uint8(0):
                    death_cause[sk, g] = np.uint8(5)
                continue
            hy = np.int32((head_cell >> np.uint32(4)) & np.uint32(0xF))
            mv = np.int32((moves[g] >> np.uint8(sk * 2)) & np.uint8(3))

            oob = ((mv == np.int32(0) and hy == np.int32(0))  or
                (mv == np.int32(1) and hy == np.int32(14)) or
                (mv == np.int32(2) and hx == np.int32(0))  or
                (mv == np.int32(3) and hx == np.int32(14)))
            boundary_dead[sk, g] = oob # oob = out of bounds
            if oob:
                continue

            if   mv == np.int32(0): hy -= np.int32(1)
            elif mv == np.int32(1): hy += np.int32(1)
            elif mv == np.int32(2): hx -= np.int32(1)
            else: hx += np.int32(1)

            pos = np.int32(hy * np.int32(15) + hx)
            new_head_pos[sk, g] = np.uint16(pos)

            old_val = new_heads[g, pos]
            if old_val == np.uint8(0xFF):
                new_heads[g, pos] = np.uint8(1 << sk)
            else:
                new_heads[g, pos] = old_val | np.uint8(1 << sk)

            if (food[g, pos >> np.int32(3)] >> np.uint8(pos & np.int32(7))) & np.uint8(1):
                is_eating[sk, g] = True

    # do stuff but per snake over full gamem
    food_to_add = np.zeros((game_count, 4), dtype=np.uint8)

    for sk in range(4):
        for g in range(game_count):
            if boundary_dead[sk, g]:
                snakes_died[sk, g] = True
                if death_cause[sk, g] == np.uint8(0):
                    death_cause[sk, g] = np.uint8(2)

        active_count = 0
        active_index = np.zeros(game_count, dtype=np.uint16)
        for g in range(game_count):
            if game_over[g]:
                continue
            if flag_arr[sk, g] and not snakes_died[sk, g]:
                active_index[active_count] = np.uint16(g)
                active_count += 1

        for ai in range(active_count):
            g = int(active_index[ai])
            pos = int(new_head_pos[sk, g])

            grid_val = new_heads[g, pos]
            other_bits = np.uint8(0)
            for j in range(4):
                if j != sk and (grid_val & np.uint8(1 << j)):
                    if flag_arr[j, g] and not snakes_died[j, g]:
                        other_bits |= np.uint8(1 << j)

            if other_bits != np.uint8(0):
                max_len = int(length[g, sk])
                for j in range(4):
                    if (other_bits & np.uint8(1 << j)) and int(length[g, j]) > max_len:
                        max_len = int(length[g, j])

                k_dies = int(length[g, sk]) < max_len
                for j in range(4):
                    if other_bits & np.uint8(1 << j):
                        if int(length[g, j]) < max_len:
                            snakes_died[j, g] = True
                            if death_cause[j, g] == np.uint8(0):
                                death_cause[j, g] = np.uint8(3)
                        elif int(length[g, j]) == max_len and int(length[g, sk]) == max_len:
                            snakes_died[j, g] = True
                            if death_cause[j, g] == np.uint8(0):
                                death_cause[j, g] = np.uint8(3)
                            k_dies = True
                if k_dies:
                    snakes_died[sk, g] = True
                    if death_cause[sk, g] == np.uint8(0):
                        death_cause[sk, g] = np.uint8(3)
                    continue

            occupant = int(board_snake_id[g, pos])
            if occupant != 255:
                j = occupant
                if flag_arr[j, g] and not snakes_died[j, g]:
                    j_tail_ring = (int(timestep) - int(length[g, j]) + 1) % 225
                    occ_ring = int(board_ring_idx[g, pos])

                    if j == sk:
                        if occ_ring == j_tail_ring and not is_eating[sk, g]:
                            pass
                        else:
                            snakes_died[sk, g] = True
                            if death_cause[sk, g] == np.uint8(0):
                                death_cause[sk, g] = np.uint8(4)
                            continue
                    elif j < sk:
                        snakes_died[sk, g] = True
                        if death_cause[sk, g] == np.uint8(0):
                            death_cause[sk, g] = np.uint8(4)
                        continue
                    else:
                        if occ_ring == j_tail_ring and not is_eating[j, g]:
                            pass
                        else:
                            snakes_died[sk, g] = True
                            if death_cause[sk, g] == np.uint8(0):
                                death_cause[sk, g] = np.uint8(4)
                            continue

            nx = pos % 15
            ny = pos // 15
            data[g, sk, new_ring] = (
                np.uint32(nx) |
                (np.uint32(ny) << np.uint32(4)) |
                ((np.uint32(timestep + 1) & np.uint32(0x3FF)) << np.uint32(16)) |
                (np.uint32(sk) << np.uint32(26))
            )

            if not is_eating[sk, g]:
                sk_tail_ring = (int(timestep) - int(length[g, sk]) + 1) % 225
                tail_cell = data[g, sk, sk_tail_ring]
                tx = int(tail_cell & np.uint32(0xF))
                ty = int((tail_cell >> np.uint32(4)) & np.uint32(0xF))
                data[g, sk, sk_tail_ring] = (
                    np.uint32(0xF) | (np.uint32(0xF) << np.uint32(4)) | (np.uint32(sk) << np.uint32(26))
                )
                if tx != 0xF:
                    board_snake_id[g, ty * 15 + tx] = np.uint8(0xFF)

            board_snake_id[g, pos] = np.uint8(sk)
            board_ring_idx[g, pos] = np.uint8(new_ring)

    # food is yummy
    for g in range(game_count):
        if game_over[g]:
            continue
        for sk in range(4):
            if flag_arr[sk, g] and not snakes_died[sk, g] and is_eating[sk, g]:
                food_to_add[g, sk] = np.uint8(1)
                pos = int(new_head_pos[sk, g])
                food[g, pos >> 3] &= np.uint8(0xFF) ^ (np.uint8(1) << np.uint8(pos & 7))
                length[g, sk] += np.uint8(1)
                health[g, sk] = np.uint8(100)

    # remove dead snake cells
    for g in range(game_count):
        if game_over[g]:
            continue
        for sk in range(4):
            if snakes_died[sk, g]:
                empty_cell = np.uint32(0xF) | (np.uint32(0xF) << np.uint32(4)) | (np.uint32(sk) << np.uint32(26))
                for ri in range(225):
                    data[g, sk, ri] = empty_cell

    # add new snake flags with new ded snakes
    for g in range(game_count):
        if game_over[g]:
            continue
        nibble = np.uint8(0)
        for sk in range(4):
            if flag_arr[sk, g] and not snakes_died[sk, g]:
                nibble |= np.uint8(1 << sk)
        nibble_shift = np.uint8((g & 1) * 4)
        comp_mask = np.uint8(0xF0) if (g & 1) == 0 else np.uint8(0x0F)
        flags[g >> 1] = (flags[g >> 1] & comp_mask) | (nibble << nibble_shift)

    return snakes_died, food_to_add, death_cause


# numba hates dicts
def step_wrapper(game_state, moves):
    snakes_died, food_to_add, death_cause = step(
        game_state['data'],
        game_state['flags'],
        game_state['food'],
        moves,
        game_state['length'],
        game_state['health'],
        game_state['timestep'],
        batch_size,
    )
    game_state['timestep'] = np.uint32(game_state['timestep'] + 1)
    game_state['new_apple'] = spawn_food(game_state['data'], game_state['flags'], game_state['food'], batch_size)
    return snakes_died, food_to_add, death_cause