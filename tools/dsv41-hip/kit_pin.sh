# kit_pin.sh (sourced by prepare_inputs.sh and verify_inputs.sh): the component kit the CPU input tools come from.
# tensorfold-strix, branch port/strix-halo, restored from tensorfold-strix.bundle (SHA-256 2c0112d2375f8c1f...);
# Apache-2.0 (its LICENSE, NOTICE, THIRD_PARTY_NOTICES.md, LICENSES/). lock.json records the bundle and both checkouts.
TFP_KIT_COMMIT=def87879ce82f7acaf8d8b441c7429776c87a221

# input_tools HERE: the directory holding prepare_inputs.py and friends. TFP_INPUT_TOOLS if set, else HERE when the
# tools are vendored there, else ~/tfp/kit/tensorfold-strix/tools/dsv41-hip; a kit directory must be the
# tools/dsv41-hip of a clean checkout at TFP_KIT_COMMIT (review m6), or this returns nonzero.
input_tools() {
  local here=$1 tools=${TFP_INPUT_TOOLS:-} top head
  [ -n "$tools" ] || { [ -f "$here/prepare_inputs.py" ] && tools=$here || tools=$HOME/tfp/kit/tensorfold-strix/tools/dsv41-hip; }
  tools=$(cd "$tools" && pwd -P) || return 1
  if [ "$tools" != "$(cd "$here" && pwd -P)" ]; then
    top=$(git -C "$tools" rev-parse --show-toplevel 2>/dev/null) || { echo "input tools $tools: not a git checkout of the kit" >&2; return 1; }
    head=$(git -C "$top" rev-parse HEAD)
    [ "$head" = "$TFP_KIT_COMMIT" ] || { echo "input tools: kit checkout $top is at $head, not $TFP_KIT_COMMIT" >&2; return 1; }
    [ -z "$(git -C "$top" status --porcelain)" ] || { echo "input tools: kit checkout $top has local changes" >&2; return 1; }
    [ "$tools" = "$(cd "$top/tools/dsv41-hip" && pwd -P)" ] || { echo "input tools: $tools is not $top/tools/dsv41-hip" >&2; return 1; }
  fi
  echo "$tools"
}
