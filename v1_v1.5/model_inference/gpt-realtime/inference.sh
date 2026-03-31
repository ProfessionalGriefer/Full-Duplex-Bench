#!/bin/bash

# Usage: bash inference.sh {/path/to/dataset} {MAX_JOBS}
# Default MAX_JOBS=4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET_DIR="$1"
MAX_JOBS="${2:-4}"
ERROR_LOG="$DATASET_DIR/errors.log"

if [[ -z "$DATASET_DIR" ]]; then
  echo "Usage: $0 /path/to/dataset [MAX_JOBS]"
  exit 1
fi

echo "[inference] Dataset dir: $DATASET_DIR"
echo "[inference] Max parallel jobs: $MAX_JOBS"

# Clear previous error log
echo "" >"$ERROR_LOG"

# Count folders to process
TOTAL_FOLDERS=0
for FOLDER in "$DATASET_DIR"/*; do
  [[ -d "$FOLDER" ]] || continue
  BASENAME=$(basename "$FOLDER")
  [[ "$BASENAME" =~ ^[0-9]+$ ]] || continue
  TOTAL_FOLDERS=$((TOTAL_FOLDERS + 1))
done
echo "[inference] Found $TOTAL_FOLDERS folders to process"

COMPLETED=0
FAILED=0

# Function to process a single folder
process_folder() {
  FOLDER="$1"
  BASENAME=$(basename "$FOLDER")
  INPUT_WAV="$FOLDER/input.wav"
  OUTPUT_WAV="$FOLDER/combined.wav"
  GPT_WAV="$FOLDER/combined_gpt_response.wav"
  FINAL_WAV="$FOLDER/output.wav"

  echo "[inference] [$BASENAME] Starting processing..."

  if [[ -f "$FINAL_WAV" ]]; then
    echo "[inference] [$BASENAME] Skipping (output.wav already exists)"
    return 0
  fi

  if [[ ! -f "$INPUT_WAV" ]]; then
    echo "[inference] [$BASENAME] ERROR: input.wav not found at $INPUT_WAV"
    echo "[$FOLDER] input wav not found" >>"$ERROR_LOG"
    return 1
  fi

  echo "[inference] [$BASENAME] Running CLI: input=$INPUT_WAV output=$OUTPUT_WAV"

  # Run the CLI
  if ! node "$SCRIPT_DIR/cli.js" --input "$INPUT_WAV" --output "$OUTPUT_WAV" >"$FOLDER/cli.log" 2>&1; then
    echo "[inference] [$BASENAME] ERROR: CLI failed (exit code: $?). Log tail:"
    tail -20 "$FOLDER/cli.log" | sed "s/^/  [$BASENAME] /"
    echo "[$FOLDER] CLI failed (see $FOLDER/cli.log)" >>"$ERROR_LOG"
    return 1
  fi

  echo "[inference] [$BASENAME] CLI completed successfully"

  # Move/rename the GPT response file
  if [[ ! -f "$GPT_WAV" ]]; then
    echo "[inference] [$BASENAME] ERROR: $GPT_WAV not found after CLI. Files in folder:"
    ls -la "$FOLDER"/*.wav 2>/dev/null | sed "s/^/  [$BASENAME] /" || echo "  [$BASENAME] No .wav files found"
    echo "[$FOLDER] combined_gpt_response.wav not found after CLI" >>"$ERROR_LOG"
    return 1
  fi

  if ! mv -f "$GPT_WAV" "$FINAL_WAV"; then
    echo "[inference] [$BASENAME] ERROR: Failed to move $GPT_WAV -> $FINAL_WAV"
    echo "[$FOLDER] Failed to move combined_gpt_response.wav to output.wav" >>"$ERROR_LOG"
    return 1
  fi

  echo "[inference] [$BASENAME] Done -> $FINAL_WAV"
}

# Portable job control for parallelism
pids=()
for FOLDER in "$DATASET_DIR"/*; do
  [[ -d "$FOLDER" ]] || continue
  BASENAME=$(basename "$FOLDER")
  [[ "$BASENAME" =~ ^[0-9]+$ ]] || continue

  process_folder "$FOLDER" &
  pids+=("$!")

  # Limit parallel jobs
  while ((${#pids[@]} >= MAX_JOBS)); do
    for i in "${!pids[@]}"; do
      if ! kill -0 "${pids[i]}" 2>/dev/null; then
        wait "${pids[i]}"
        unset 'pids[i]'
      fi
    done
    # Remove empty elements
    pids=("${pids[@]}")
    sleep 0.5
  done

done

# Wait for all jobs to finish
for pid in "${pids[@]}"; do
  wait "$pid" || FAILED=$((FAILED + 1))
  COMPLETED=$((COMPLETED + 1))
done

echo ""
echo "[inference] ========================================="
echo "[inference] Processing complete: $COMPLETED/$TOTAL_FOLDERS folders processed"
if [[ -s "$ERROR_LOG" ]] && grep -q '\S' "$ERROR_LOG"; then
  ERROR_COUNT=$(grep -c '\S' "$ERROR_LOG")
  echo "[inference] $ERROR_COUNT errors logged to $ERROR_LOG"
else
  echo "[inference] No errors"
fi
echo "[inference] ========================================="

