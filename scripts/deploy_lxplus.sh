#!/bin/bash
# Install (or update) HSF Part II INSIDE the Phase-1 Optimization folder, under hsf2* names only.
#
#   cd /eos/user/<u>/<user>/my_acts/acts/Examples/Scripts/Optimization          # the Phase-1 folder
#   rm -rf ~/hsf2_stage && mkdir -p ~/hsf2_stage && tar -xzf ~/hsf2_bundle_<stamp>.tgz -C ~/hsf2_stage
#   bash ~/hsf2_stage/hsf2_scripts/deploy_lxplus.sh ~/hsf2_stage              # first installation
#   bash ~/hsf2_stage/hsf2_scripts/deploy_lxplus.sh ~/hsf2_stage --update     # later code updates
#
# Guarantees: it only creates / replaces  hsf2/ hsf2_condor/ hsf2_scripts/ hsf2_tests/ hsf2_docs/  and creates
# hsf2_work/; it refuses to run outside a Phase-1 folder; every bundle file is checked against MANIFEST.txt first; a
# first installation refuses if any of those exist; an update refuses while Part-II HTCondor jobs are queued or running
# (--force overrides) and MOVES the previous code to hsf2_work/old_code/<stamp>/ (nothing is deleted; results in
# hsf2_work/ are never touched); hsf2* is added to the git exclude list of the enclosing repository BEFORE copying;
# afterwards it checks that every non-hsf2 entry of the folder is unchanged.
set -euo pipefail
STAGE=${1:?staging folder (where the bundle was unpacked)}
MODE=${2:-install}
FORCE=${3:-}
TARGET=$(pwd -P)
STAGE=$(cd "$STAGE" && pwd -P)
CODE=(hsf2 hsf2_condor hsf2_scripts hsf2_tests hsf2_docs)

[ -f "$TARGET/DirectQAOAFeedback.py" ] && [ -d "$TARGET/acts_qaoa" ] || {
  echo "run this from the Phase-1 Optimization folder (DirectQAOAFeedback.py and acts_qaoa/ expected in $TARGET)"; exit 2; }
[ "$STAGE" != "$TARGET" ] || { echo "staging folder must not be the target folder"; exit 2; }
for n in "${CODE[@]}"; do [ -d "$STAGE/$n" ] || { echo "bundle incomplete: $STAGE/$n missing"; exit 2; }; done
( cd "$STAGE" && sha1sum -c --quiet hsf2_docs/MANIFEST.txt ) || { echo "bundle files do not match MANIFEST.txt"; exit 2; }
echo "bundle verified: $(wc -l < "$STAGE/hsf2_docs/MANIFEST.txt") files, $(head -1 "$STAGE/hsf2_docs/DEPLOYED_REV")"

# git exclude first, so the hsf2* folders are never visible to the enclosing repository, even after a failed copy
if git -C "$TARGET" rev-parse --git-dir >/dev/null 2>&1; then
  # --git-path (git >= 2.5) prints the path relative to $TARGET unless it is absolute; lxplus's git predates
  # --path-format=absolute (2.31) and echoes it back, so resolve here and accept one .../info/exclude line only
  excl=$(git -C "$TARGET" rev-parse --git-path info/exclude)
  case "$excl" in /*) ;; *) excl="$TARGET/$excl" ;; esac
  if [[ "$excl" == *$'\n'* || "$excl" != */info/exclude ]]; then
    echo "could not locate the git exclude file (git said: $excl)"; exit 2
  fi
  mkdir -p "$(dirname "$excl")"
  excl="$(cd "$(dirname "$excl")" && pwd -P)/exclude"
  grep -qxF 'hsf2*' "$excl" 2>/dev/null || echo 'hsf2*' >> "$excl"
  echo "git: 'hsf2*' excluded in $excl"
fi

snapshot() { ( cd "$TARGET" && ls -A | { grep -v '^hsf2' || true; } | while IFS= read -r e; do stat -c '%n %Y' "$e"; done ) | sort; }
before=$(snapshot)

if [ "$MODE" = "--update" ]; then
  # node jobs run hsf2_condor/run.sh ... hsf2; a DAGMan job between two nodes only names its DAG (part2_stage*.dag,
  # hsf2_envtest.dag), so match both or an update could land in the middle of a DAG
  if [ "$FORCE" != "--force" ] && command -v condor_q >/dev/null 2>&1 && \
     condor_q -nobatch -af Cmd Args 2>/dev/null | grep -qE 'hsf2|part2_stage'; then
    echo "Part-II HTCondor jobs are queued or running: update after they finish (or add --force)"; exit 2
  fi
  old="$TARGET/hsf2_work/old_code/$(date +%d-%m-%Y-%H-%M-%S)"
  mkdir -p "$old"
  for n in "${CODE[@]}"; do [ -e "$TARGET/$n" ] && mv "$TARGET/$n" "$old/"; done
  echo "previous code moved to $old"
else
  for n in "${CODE[@]}" hsf2_work; do
    [ -e "$TARGET/$n" ] && { echo "$TARGET/$n exists: use --update (first installation refuses to overwrite)"; exit 2; }
  done
fi
for n in "${CODE[@]}"; do cp -r "$STAGE/$n" "$TARGET/$n"; done
chmod +x "$TARGET"/hsf2_scripts/*.sh "$TARGET"/hsf2_condor/*.sh
mkdir -p "$TARGET"/hsf2_work/{data,models,runs,figures,env}

after=$(snapshot)
if [ "$before" != "$after" ]; then
  echo "WARNING: a non-hsf2 entry of $TARGET changed during deployment (a Phase-1 job may be writing):"
  diff <(echo "$before") <(echo "$after") || true
else
  echo "Phase-1 entries of $TARGET unchanged ($(echo "$before" | wc -l) checked)"
fi
echo "installed: $(head -1 "$TARGET/hsf2_docs/DEPLOYED_REV")"
echo "next: [once] bash $TARGET/hsf2_scripts/setup_hsf2_venv.sh   then   source $TARGET/hsf2_scripts/env_lxplus.sh"
echo "      HTCondor: mkdir -p ~/hsf2_condor/logs && cp $TARGET/hsf2_condor/* ~/hsf2_condor/ && chmod +x ~/hsf2_condor/*.sh"
