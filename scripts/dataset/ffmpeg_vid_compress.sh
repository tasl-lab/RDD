#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./ffmpeg_vid_compress.sh input.mp4
#   ./ffmpeg_vid_compress.sh input.mp4 23 128k 1080
#
# Args:
#   $1 input video (required)
#   $2 quality (optional)   : for x264/x265 = CRF (lower=better, bigger file), default 23
#                             for GPU encoders = CQ/Quality (lower=better), default 23
#   $3 audio bitrate (opt)  : default 128k
#   $4 max height (opt)     : e.g., 1080 or 720; empty means no scaling

vid="${1:-}"
quality="${2:-23}"
abitrate="${3:-128k}"
max_h="${4:-}"

if [[ -z "$vid" ]]; then
  echo "Usage: $0 <input_video> [quality=23] [audio_bitrate=128k] [max_height=]"
  exit 1
fi

if [[ ! -f "$vid" ]]; then
  echo "Error: input file not found: $vid"
  exit 1
fi

# Output path: same folder, with suffix
dir="$(dirname "$vid")"
base="$(basename "$vid")"
name="${base%.*}"
out="${dir}/${name}_compressed.mp4"

# Build scale filter (optional)
# - keeps aspect ratio, caps height to max_h if provided
vf_scale=""
if [[ -n "$max_h" ]]; then
  if ! [[ "$max_h" =~ ^[0-9]+$ ]]; then
    echo "Error: max_height must be an integer (e.g., 1080). Got: $max_h"
    exit 1
  fi
  vf_scale="scale=-2:'min(${max_h},ih)'"
fi

# Query encoders once
encoders="$(ffmpeg -hide_banner -encoders 2>/dev/null || true)"
has_encoder() { grep -qE "[[:space:]]$1([[:space:]]|$)" <<<"$encoders"; }

# Select encoder (fastest typical paths first)
encoder="libx264"
extra_in=()
vflags=()
vf=""

if has_encoder "h264_nvenc"; then
  encoder="h264_nvenc"
  # NVENC: -cq is constant quality, -preset p1..p7 (p1 fastest, p7 best)
  vflags=( -preset p5 -cq "$quality" )
  vf="${vf_scale:-}"
elif has_encoder "h264_qsv"; then
  encoder="h264_qsv"
  # QSV quality scale uses -global_quality (lower is better; 18-28 typical)
  vflags=( -global_quality "$quality" )
  vf="${vf_scale:-}"
elif has_encoder "h264_videotoolbox"; then
  encoder="h264_videotoolbox"
  # VideoToolbox: easiest portable control is bitrate; map "quality" loosely to bitrate tiers
  # You can override by editing b:v below if desired.
  # 23 -> ~6M for 1080p-ish content; increase for higher motion/res.
  vflags=( -b:v 6M )
  vf="${vf_scale:-}"
elif has_encoder "h264_vaapi"; then
  encoder="h264_vaapi"
  # VAAPI often requires a device + hwupload and nv12.
  extra_in=( -vaapi_device /dev/dri/renderD128 )
  # Use qp if supported; otherwise VAAPI may ignore some quality knobs depending on driver.
  # For VAAPI we must include format+hwupload in vf.
  if [[ -n "$vf_scale" ]]; then
    vf="format=nv12,${vf_scale},hwupload"
  else
    vf="format=nv12,hwupload"
  fi
  vflags=( -qp "$quality" )
else
  encoder="libx264"
  # CPU path: preset trades speed vs size; "slow" compresses well, "veryfast" is quicker.
  vflags=( -preset slow -crf "$quality" )
  vf="${vf_scale:-}"
fi

echo "Input : $vid"
echo "Output: $out"
echo "Video encoder: $encoder"
echo "Quality: $quality   Audio: $abitrate   MaxHeight: ${max_h:-no-limit}"

# Assemble ffmpeg command
cmd=( ffmpeg -hide_banner -y )
cmd+=( "${extra_in[@]}" )
cmd+=( -i "$vid" )

# Video filters (if any)
if [[ -n "$vf" ]]; then
  cmd+=( -vf "$vf" )
fi

# Encode video + audio; force MP4-friendly settings
cmd+=( -c:v "$encoder" )
cmd+=( "${vflags[@]}" )
cmd+=( -c:a aac -b:a "$abitrate" -movflags +faststart )

cmd+=( "$out" )

# Run
"${cmd[@]}"
