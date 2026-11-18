#!/bin/bash
# Download TrackML files (Zenodo record 4730167, DOI 10.5281/zenodo.4730167, licence CC-BY-4.0) into
#     $HSF2_DATA/raw/     (default <Optimization>/hsf2_work/data/raw)
# Resumable (curl -C -) and checksum-verified (MD5 published by Zenodo).  Only train_1 (16 GB, ~1770 events)
# is needed for Part II; the zip is read directly by "python -m hsf2 trackml-prep" (no unzip).
#
# usage:   bash scripts/trackml_download.sh                 # detectors.csv + train_1.zip
#          bash scripts/trackml_download.sh train_2         # more files: train_2 ... train_5
# Prefer the HTCondor job hsf2_condor/download_trackml.sub: background jobs on lxplus lose EOS access when
# the SSH session ends (Phase-1 lesson).  Re-running continues an interrupted download.

HSF2_EOS_HOME=${HSF2_EOS_HOME:-/eos/user/$(id -un | cut -c1)/$(id -un)}   # your EOS home (id -un: also in batch jobs)
DATA="${HSF2_DATA:-$HSF2_EOS_HOME/my_acts/acts/Examples/Scripts/Optimization/hsf2_work/data}"
RAW="$DATA/raw"
BASE="https://zenodo.org/api/records/4730167/files"
declare -A MD5=(
  [detectors.csv]=440dbb939f62cc94de563de8cdae47d1
  [train_1.zip]=c1aa5e944b2a07942e8f6c3613497478
  [train_2.zip]=75974545e4e9314e4b1722a24637bbea
  [train_3.zip]=5e46b1ac5af91c7ac214925df7c80eb1
  [train_4.zip]=4a836f0079845a4d9dcbbb6e35cbfff3
  [train_5.zip]=9ddd9d9445a4a78d4c2d45c66f097e0b
)
declare -A SIZE=(
  [detectors.csv]=1942066
  [train_1.zip]=15941486907
  [train_2.zip]=15999064939
  [train_3.zip]=15967434796
  [train_4.zip]=15956023763
  [train_5.zip]=15959040049
)

FILES=("detectors.csv")
if [ $# -eq 0 ]; then FILES+=("train_1.zip"); fi
for a in "$@"; do
  case "$a" in *.zip|*.csv) f="$a" ;; *) f="$a.zip" ;; esac
  [[ " ${FILES[*]} " == *" $f "* ]] || FILES+=("$f")
done

mkdir -p "$RAW" || { echo "cannot create $RAW"; exit 2; }
echo "$(date '+%d-%m-%Y %H:%M:%S')  host $(hostname)  ->  $RAW"
df -h "$RAW" 2>/dev/null | tail -1

status=0
for f in "${FILES[@]}"; do
  want=${MD5[$f]}
  if [ -z "$want" ]; then echo "unknown file $f (known: ${!MD5[*]})"; status=2; continue; fi
  dest="$RAW/$f"
  if [ -f "$dest" ]; then
    echo "$f exists -- verifying MD5 ..."
    if [ "$(md5sum "$dest" | cut -d' ' -f1)" = "$want" ]; then echo "$f OK (already complete)"; continue; fi
    echo "$f has a wrong checksum -> moved to $dest.bad and downloaded again"; mv "$dest" "$dest.bad"
  fi
  for attempt in 1 2 3 4 5 6; do
    echo "$(date '+%d-%m-%Y %H:%M:%S')  $f: attempt $attempt (resumes from $(stat -c %s "$dest.part" 2>/dev/null || echo 0) bytes)"
    curl -L --fail --retry 5 --retry-delay 20 --connect-timeout 60 -sS -C - -o "$dest.part" "$BASE/$f/content"
    have=$(stat -c %s "$dest.part" 2>/dev/null || echo 0)
    if [ "$have" = "${SIZE[$f]}" ]; then break; fi
    echo "  incomplete: $have of ${SIZE[$f]} bytes"; sleep 30
  done
  have=$(stat -c %s "$dest.part" 2>/dev/null || echo 0)
  if [ "$have" != "${SIZE[$f]}" ]; then echo "$f FAILED: $have of ${SIZE[$f]} bytes -- run again to resume"; status=1; continue; fi
  got=$(md5sum "$dest.part" | cut -d' ' -f1)
  if [ "$got" != "$want" ]; then echo "$f FAILED: MD5 $got != $want (file kept as $dest.part.bad)"; mv "$dest.part" "$dest.part.bad"; status=1; continue; fi
  mv "$dest.part" "$dest"
  echo "$(date '+%d-%m-%Y %H:%M:%S')  $f OK ($have bytes, MD5 verified)"
done
ls -l "$RAW"
exit $status
