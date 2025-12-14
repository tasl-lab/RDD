#!/usr/bin/env bash
set -euo pipefail

vid="${1:-}"
speed="${2:-}"

if [[ -z "${vid}" || -z "${speed}" ]]; then
  echo "Usage: $0 <input_video> <speed_factor>"
  echo "Example: $0 input.mp4 4"
  exit 1
fi

if [[ ! -f "${vid}" ]]; then
  echo "Error: input file not found: ${vid}"
  exit 1
fi

# speed must be a positive number (int/float)
if ! [[ "${speed}" =~ ^([0-9]*\.)?[0-9]+$ ]] || awk "BEGIN{exit !(${speed} > 0)}"; then
  : # ok
else
  echo "Error: speed_factor must be a positive number, got: ${speed}"
  exit 1
fi

dir="$(dirname "$vid")"
base="$(basename "$vid")"
name="${base%.*}"
out="${dir}/${name}_${speed}x.mp4"

# Detect available encoders
encoders="$(ffmpeg -hide_banner -encoders 2>/dev/null || true)"

has_encoder() {
  grep -qE "[[:space:]]$1([[:space:]]|$)" <<< "$encoders"
}

# Choose encoder (fastest first)
encoder="libx264"
extra_in=()
vopts=()

if has_encoder "h264_nvenc"; then
  encoder="h264_nvenc"
  # NVENC quality/speed knobs
  vopts=( -preset p5 -cq 23 )
elif has_encoder "h264_qsv"; then
  encoder="h264_qsv"
  vopts=( -global_quality 23 )
elif has_encoder "h264_videotoolbox"; then
  encoder="h264_videotoolbox"
  # videotoolbox uses bitrate/quality; CRF-like control is limited
  vopts=( -b:v 6M )
elif has_encoder "h264_vaapi"; then
  encoder="h264_vaapi"
  # VAAPI usually requires hwupload + device; try renderD128 by default
  extra_in=( -vaapi_device /dev/dri/renderD128 )
  # If your ffmpeg build supports it, this works broadly; quality controls vary by driver
  vopts=( -vf "format=nv12,hwupload,setpts=PTS/${speed}" -qp 23 )
else
  encoder="libx264"
  vopts=( -preset veryfast -crf 23 )
fi

echo "Input : $vid"
echo "Speed : ${speed}x"
echo "Output: $out"
echo "Encoder selected: $encoder"

# Build filters: for VAAPI we already embedded setpts in vopts above
if [[ "$encoder" == "h264_vaapi" ]]; then
  ffmpeg -hide_banner -y "${extra_in[@]}" -i "$vid" \
    -an \
    -c:v "$encoder" "${vopts[@]}" \
    "$out"
else
  ffmpeg -hide_banner -y "${extra_in[@]}" -i "$vid" \
    -vf "setpts=PTS/${speed}" \
    -an \
    -c:v "$encoder" "${vopts[@]}" \
    "$out"
fi
