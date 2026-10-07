#!/usr/bin/env bash
#
# Build shapeshifter natively on a Linux box, patched for a 15x15 board and a
# 25% food spawn chance.
#
#   curl -O <this file>  &&  bash build_shapeshifter.sh
#
# Idempotent: safe to re-run. Re-running skips anything already done and will not
# double-apply the source patches.
#
# Flags:
#   --dir PATH        where to clone/build            (default ~/shapeshifter)
#   --board N         board size                      (default 15)
#   --food N          food spawn chance, percent      (default 25)
#   --features LIST   cargo features                  (default tt,parallel_search,mcts_fallback)
#   --cpu TARGET      -C target-cpu value             (default native)
#   --no-deps         skip the package-manager step
#   --no-rust         skip the rustup step (toolchain already present)
#   --clean           cargo clean before building
#   -h, --help
#
# Notes:
#   * needs NIGHTLY rust: the crate uses generic_const_exprs and edition 2024.
#   * target-cpu=native is correct ONLY when this machine also runs the binary.
#     Building on a different host? pass --cpu x86-64-v2 or you risk SIGILL.
#   * do NOT use --features prod: it pulls in `spl`, whose code still references
#     the pre-refactor Bitboard<S,W,H,WRAP,HZSTACK,N> signature and will not compile.

set -euo pipefail

DIR="$HOME/shapeshifter"
REPO="https://github.com/JonathanArns/shapeshifter"
BOARD=15
FOOD=25
FEATURES="tt,parallel_search,mcts_fallback"
CPU="native"
DO_DEPS=1
DO_RUST=1
DO_CLEAN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dir)      DIR="$2"; shift 2 ;;
    --board)    BOARD="$2"; shift 2 ;;
    --food)     FOOD="$2"; shift 2 ;;
    --features) FEATURES="$2"; shift 2 ;;
    --cpu)      CPU="$2"; shift 2 ;;
    --no-deps)  DO_DEPS=0; shift ;;
    --no-rust)  DO_RUST=0; shift ;;
    --clean)    DO_CLEAN=1; shift ;;
    -h|--help)  sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1 (try --help)" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- deps
if [ "$DO_DEPS" = 1 ]; then
  say "installing build dependencies"
  SUDO=""; [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null && SUDO=sudo
  if   command -v apt-get >/dev/null; then
    $SUDO apt-get update -qq
    DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq build-essential curl git ca-certificates
  elif command -v dnf >/dev/null; then
    $SUDO dnf install -y -q gcc gcc-c++ make curl git
  elif command -v pacman >/dev/null; then
    $SUDO pacman -Sy --noconfirm --needed base-devel curl git
  elif command -v apk >/dev/null; then
    $SUDO apk add --no-cache build-base curl git
  else
    echo "unknown package manager; ensure a C compiler, curl and git are present"
  fi
else
  say "skipping dependency install (--no-deps)"
fi

command -v cc >/dev/null || command -v gcc >/dev/null \
  || die "no C compiler found - cargo needs one to link. Install build-essential/gcc."
command -v git >/dev/null || die "git not found"

# ---------------------------------------------------------------- rust
if [ "$DO_RUST" = 1 ] && ! command -v cargo >/dev/null && [ ! -x "$HOME/.cargo/bin/cargo" ]; then
  say "installing nightly rust (minimal profile)"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain nightly
fi
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
command -v cargo >/dev/null || die "cargo not on PATH; run: . \$HOME/.cargo/env"

if ! rustup toolchain list 2>/dev/null | grep -q nightly; then
  say "adding nightly toolchain"
  rustup toolchain install nightly --profile minimal
fi
say "toolchain: $(rustc +nightly --version)"

# ---------------------------------------------------------------- source
if [ -d "$DIR/.git" ]; then
  say "reusing existing checkout at $DIR"
else
  say "cloning into $DIR"
  git clone --depth 1 "$REPO" "$DIR"
fi
cd "$DIR"

MODE=src/bitboard/mode.rs
API=src/api.rs
RULES=src/bitboard/rules.rs
for f in "$MODE" "$API" "$RULES"; do
  [ -f "$f" ] && continue
  die "$f missing - is $DIR really a shapeshifter checkout?"
done

# ---------------------------------------------------------------- patch
# Board size and food chance are compile-time constants, not runtime config,
# so they have to be edited into the source before building.
say "patching board -> ${BOARD}x${BOARD}, food spawn -> ${FOOD}%"

if grep -q "const W: usize = ${BOARD};" "$MODE"; then
  echo "  mode.rs already at ${BOARD}"
else
  sed -i -E "s/const W: usize = [0-9]+;/const W: usize = ${BOARD};/; \
             s/const H: usize = [0-9]+;/const H: usize = ${BOARD};/" "$MODE"
fi

# Only the non-spl match arms are 4-tuples "(n, W, H, bool)". The spl arms are
# 5-tuples ending "..., true, false)" so this pattern cannot touch them.
if grep -qE ", ${BOARD}, ${BOARD}, (true|false)\)" "$API"; then
  echo "  api.rs already dispatching ${BOARD}x${BOARD}"
else
  sed -i -E "s/, [0-9]+, [0-9]+, (true|false)\)/, ${BOARD}, ${BOARD}, \1)/g" "$API"
fi

sed -i -E "s/gen_ratio\([0-9]+, 100\)/gen_ratio(${FOOD}, 100)/" "$RULES"

say "verifying patches"
n_mode=$(grep -c "usize = ${BOARD};" "$MODE" || true)
n_api=$(grep -cE ", ${BOARD}, ${BOARD}, (true|false)\)" "$API" || true)
n_food=$(grep -c "gen_ratio(${FOOD}, 100)" "$RULES" || true)
echo "  mode.rs  W/H constants : $n_mode   (expect 4)"
echo "  api.rs   dispatch arms : $n_api   (expect 24)"
echo "  rules.rs food spawn    : $n_food   (expect 1)"
[ "$n_mode" -ge 4 ] || die "mode.rs patch failed"
[ "$n_api"  -ge 1 ] || die "api.rs patch failed - dispatch arms not rewritten"
[ "$n_food" -ge 1 ] || die "rules.rs patch failed"

# ---------------------------------------------------------------- build
[ "$DO_CLEAN" = 1 ] && { say "cargo clean"; cargo clean; }

say "building (features: $FEATURES, target-cpu: $CPU)"
[ "$CPU" = "native" ] && cat <<'WARN'
  note: target-cpu=native bakes in this machine's instruction set. If the binary
        will run on different hardware, rebuild with --cpu x86-64-v2.
WARN

RUSTFLAGS="-C target-cpu=${CPU}" \
  cargo +nightly build --release --bin shapeshifter --features "$FEATURES"

BIN="$DIR/target/release/shapeshifter"
[ -x "$BIN" ] || die "build reported success but $BIN is missing"

say "done"
ls -lh "$BIN"
command -v file >/dev/null && file "$BIN"
if command -v objdump >/dev/null; then
  echo "  max glibc symbol required: $(objdump -T "$BIN" \
    | grep -o 'GLIBC_[0-9.]*' | sort -Vu | tail -1)"
fi

# ---------------------------------------------------------------- smoke test
if command -v curl >/dev/null; then
  say "smoke test on a ${BOARD}x${BOARD} board"
  "$BIN" >/tmp/shapeshifter-smoke.log 2>&1 &
  PID=$!
  trap 'kill $PID 2>/dev/null || true' EXIT
  sleep 2
  H=$((BOARD - 1))
  read -r -d '' STATE <<EOF || true
{"game":{"id":"smoke","ruleset":{"name":"standard","version":"v1"},"map":"standard",
"timeout":300,"source":"custom"},"turn":10,
"board":{"height":${BOARD},"width":${BOARD},"food":[{"x":7,"y":7}],"hazards":[],
"snakes":[
 {"id":"me","name":"me","health":90,"length":3,"head":{"x":3,"y":3},
  "body":[{"x":3,"y":3},{"x":3,"y":2},{"x":3,"y":1}],"shout":null,"squad":null,"next_move":null},
 {"id":"op","name":"op","health":90,"length":3,"head":{"x":${H},"y":${H}},
  "body":[{"x":${H},"y":${H}},{"x":$((H-1)),"y":${H}},{"x":$((H-2)),"y":${H}}],
  "shout":null,"squad":null,"next_move":null}]},
"you":{"id":"me","name":"me","health":90,"length":3,"head":{"x":3,"y":3},
 "body":[{"x":3,"y":3},{"x":3,"y":2},{"x":3,"y":1}],"shout":null,"squad":null,"next_move":null}}
EOF
  echo -n "  GET  /          -> "; curl -s --max-time 10 localhost:8080; echo
  echo -n "  POST /move      -> "
  curl -s --max-time 10 -X POST localhost:8080/move \
       -H 'Content-Type: application/json' -d "$STATE" || echo "(no response)"
  echo
  kill $PID 2>/dev/null || true
  trap - EXIT
fi

cat <<EOF

binary : $BIN
run    : PORT=8080 $BIN
EOF