#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BURN_DRAGON_TRAIN_BIN:-${ROOT}/target/release/examples/train_language}"
OUT="${RANDOM_SCAFFOLD_PARITY_OUT:-${ROOT}/target/experiments/random-scaffold-parity}"
BACKEND="${RANDOM_SCAFFOLD_PARITY_BACKEND:-cuda}"
POLL_SECONDS="${RANDOM_SCAFFOLD_PARITY_GPU_POLL_SECONDS:-1}"

variants=(
  "dense:ruliad-parity.dense.toml"
  "rank8:ruliad-parity.rank8.toml"
  "rs-rank16:ruliad-parity.rs-rank16.toml"
  "rs-rank32:ruliad-parity.rs-rank32.toml"
)
seeds=(29 30 31)

if [[ ! -x "${BIN}" ]]; then
  echo "missing release training binary: ${BIN}" >&2
  exit 1
fi

mkdir -p "${OUT}"
manifest="${OUT}/manifest.tsv"
printf 'variant\tseed\trun_id\telapsed_ms\tgpu_log\ttime_log\n' > "${manifest}"

monitor_pid=""
cleanup_monitor() {
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
    monitor_pid=""
  fi
}
trap cleanup_monitor EXIT INT TERM

for row in "${variants[@]}"; do
  variant="${row%%:*}"
  profile="${row#*:}"
  for seed in "${seeds[@]}"; do
    stem="${variant}-seed-${seed}"
    gpu_log="${OUT}/${stem}.gpu.csv"
    time_log="${OUT}/${stem}.time.txt"
    stdout_log="${OUT}/${stem}.stdout.log"
    profile_path="${ROOT}/config/language/experiments/random_scaffold/${profile}"
    seed_path="${ROOT}/config/language/experiments/random_scaffold/seeds/seed-${seed}.toml"

    nvidia-smi \
      --query-gpu=timestamp,power.draw,utilization.gpu \
      --format=csv,noheader,nounits \
      --loop="${POLL_SECONDS}" > "${gpu_log}" &
    monitor_pid="$!"

    started_ns="$(date +%s%N)"
    /usr/bin/time -v -o "${time_log}" \
      "${BIN}" \
      --backend "${BACKEND}" \
      --config "${profile_path}" \
      --config "${seed_path}" > "${stdout_log}" 2>&1
    finished_ns="$(date +%s%N)"
    cleanup_monitor

    run_id="$(tr -d '\n' < "${ROOT}/runs/latest")"
    elapsed_ms="$(( (finished_ns - started_ns) / 1000000 ))"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${variant}" "${seed}" "${run_id}" "${elapsed_ms}" "${gpu_log}" "${time_log}" \
      >> "${manifest}"
    echo "random-scaffold-parity variant=${variant} seed=${seed} run=${run_id} elapsed_ms=${elapsed_ms}"
  done
done

summary="${OUT}/summary.json"
"${ROOT}/scripts/random_scaffold_parity_analyze.py" \
  "${manifest}" \
  --require-gates > "${summary}"
echo "random-scaffold-parity manifest=${manifest} summary=${summary}"
