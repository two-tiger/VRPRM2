#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  cat <<'USAGE'
Usage:
  bash VRPRM_v2.0/benchmarks/vlmeval_bon/download_table2_datasets.sh <target_lmu_data_dir> [download_cache_dir]

Example:
  bash VRPRM_v2.0/benchmarks/vlmeval_bon/download_table2_datasets.sh /data/VLMEvalData

If the download host has an expired certificate and you still want to proceed
with checksum validation:
  ALLOW_INSECURE_SSL=1 bash VRPRM_v2.0/benchmarks/vlmeval_bon/download_table2_datasets.sh /data/VLMEvalData

After copying the target directory to an offline machine:
  export LMUData=/data/VLMEvalData
USAGE
  exit 1
fi

TARGET_DIR="$1"
DOWNLOAD_DIR="${2:-${TARGET_DIR}/_downloads}"

mkdir -p "${TARGET_DIR}" "${DOWNLOAD_DIR}"

has_command() {
  command -v "$1" >/dev/null 2>&1
}

md5_file() {
  local path="$1"
  if has_command md5sum; then
    md5sum "${path}" | awk '{print $1}'
  elif has_command python; then
    python - "$path" <<'PY'
import hashlib
import sys

path = sys.argv[1]
h = hashlib.md5()
with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
print(h.hexdigest())
PY
  else
    echo "ERROR: md5sum or python is required for checksum validation." >&2
    exit 1
  fi
}

check_md5() {
  local path="$1"
  local expected="$2"
  [[ -f "${path}" ]] || return 1
  local actual
  actual="$(md5_file "${path}")"
  [[ "${actual}" == "${expected}" ]]
}

download_url() {
  local url="$1"
  local output="$2"
  local tmp="${output}.tmp"

  if has_command curl; then
    local curl_args=(-L --fail --retry 3 --retry-delay 5)
    if [[ "${ALLOW_INSECURE_SSL:-0}" == "1" ]]; then
      curl_args+=(-k)
    fi
    curl "${curl_args[@]}" -o "${tmp}" "${url}"
  elif has_command wget; then
    local wget_args=()
    if [[ "${ALLOW_INSECURE_SSL:-0}" == "1" ]]; then
      wget_args+=(--no-check-certificate)
    fi
    wget "${wget_args[@]}" -O "${tmp}" "${url}"
  else
    echo "ERROR: curl or wget is required for downloading." >&2
    exit 1
  fi
  mv "${tmp}" "${output}"
}

copy_verified() {
  local src="$1"
  local dst="$2"
  local expected="$3"

  if ! check_md5 "${src}" "${expected}"; then
    echo "ERROR: checksum mismatch for ${src}" >&2
    echo "       expected: ${expected}" >&2
    echo "       actual:   $(md5_file "${src}")" >&2
    exit 1
  fi
  cp -f "${src}" "${dst}"
}

fetch_one() {
  local alias="$1"
  local filename="$2"
  local url="$3"
  local expected_md5="$4"
  local cache_path="${DOWNLOAD_DIR}/${filename}"
  local target_path="${TARGET_DIR}/${filename}"

  if check_md5 "${target_path}" "${expected_md5}"; then
    echo "[skip] ${alias}: ${target_path}"
    return
  fi

  if ! check_md5 "${cache_path}" "${expected_md5}"; then
    echo "[download] ${alias}: ${url}"
    download_url "${url}" "${cache_path}"
  else
    echo "[cache] ${alias}: ${cache_path}"
  fi

  copy_verified "${cache_path}" "${target_path}" "${expected_md5}"
  echo "[ready] ${alias}: ${target_path}"
}

fetch_one \
  "MMMU" \
  "MMMU_DEV_VAL.tsv" \
  "https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv" \
  "585e8ad75e73f75dcad265dfd0417d64"

fetch_one \
  "MathVista" \
  "MathVista_MINI.tsv" \
  "https://opencompass.openxlab.space/utils/VLMEval/MathVista_MINI.tsv" \
  "f199b98e178e5a2a20e7048f5dcb0464"

fetch_one \
  "MathVision" \
  "MathVision.tsv" \
  "https://opencompass.openxlab.space/utils/VLMEval/MathVision.tsv" \
  "93f6de14f7916e598aa1b7165589831e"

fetch_one \
  "MathVerse-VO" \
  "MathVerse_MINI_Vision_Only.tsv" \
  "http://opencompass.openxlab.space/utils/benchmarks/MathVerse/MathVerse_MINIVOnly.tsv" \
  "68a11d4680014ac881fa37adeadea3a4"

fetch_one \
  "DynaMath" \
  "DynaMath.tsv" \
  "https://opencompass.openxlab.space/utils/VLMEval/DynaMath.tsv" \
  "b8425ad9a7114571fc9366e013699494"

fetch_one \
  "WeMath" \
  "WeMath.tsv" \
  "https://opencompass.openxlab.space/utils/VLMEval/WeMath.tsv" \
  "b5e969a075f01290a542411fb7766388"

fetch_one \
  "LogicVista" \
  "LogicVista.tsv" \
  "https://opencompass.openxlab.space/utils/VLMEval/LogicVista.tsv" \
  "41c5d33adf33765c399e0e6ae588c061"

cat <<EOF

All requested VLMEvalKit TSV files are ready in:
  ${TARGET_DIR}

Use this directory on the evaluation machine:
  export LMUData=${TARGET_DIR}
EOF
