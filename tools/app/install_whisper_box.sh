#!/usr/bin/env bash
# Whisper large-v3 lyric ASR for a GPU box: faster-whisper (SYSTRAN, MIT) int8 on CPU in a SEPARATE venv
# /workspace/asr_venv (never touches the ComfyUI venv /venv/main), model large-v3 prefetched into the default
# Hugging Face cache (the same place `WhisperModel("large-v3")` looks, as in
# _info/box_pull_58200117/extra/OWNVIDEO/asr.py: python asr.py <16k mono wav> <out.json>, device=cpu, compute_type=int8,
# cpu_threads=32, word_timestamps=True).
# Same steps as the post_install step "WHISPER large-v3 lyrics" of card-library:motion-graphic (0.3.0).
# Licence: faster-whisper MIT (SYSTRAN); ctranslate2 MIT; model weights Systran/faster-whisper-large-v3 = OpenAI Whisper large-v3, MIT.
# Size: model ~2.9 GB (model.bin 3.09 GB fp16) + venv ~0.5 GB. Run on a provisioned box:
#   scp tools/app/install_whisper_box.sh root@<host>:/workspace/ && ssh root@<host> bash /workspace/install_whisper_box.sh
# Idempotent. Exit 0 = installed + verified; exit 1 = a WARNING was printed (nothing else on the box is affected).
# Overrides: ASR_VENV (default /workspace/asr_venv), ASR_PYTHON (default python3.12, else /venv/main/bin/python),
#            FW_PIN (1.2.1), CT2_PIN (4.8.4), ASR_MODEL (large-v3).
# NOTE: the pins are NOT recorded from the old box (it was installed by hand, no log kept); 1.2.1 / 4.8.4 were the
#       current PyPI releases on 2026-10-30.

ASR_VENV="${ASR_VENV:-/workspace/asr_venv}"
FW_PIN="${FW_PIN:-1.2.1}"
CT2_PIN="${CT2_PIN:-4.8.4}"
ASR_MODEL="${ASR_MODEL:-large-v3}"

install_whisper() {
  echo "=== WHISPER ${ASR_MODEL} lyrics: faster-whisper==${FW_PIN} int8 CPU in ${ASR_VENV} (separate venv) ==="
  local PY="${ASR_PYTHON:-}"
  if [ -z "$PY" ]; then
    if command -v python3.12 >/dev/null 2>&1; then PY="$(command -v python3.12)"; else PY=/venv/main/bin/python; fi
  fi
  "$PY" --version >/dev/null 2>&1 || { echo "WARNING: no usable python ($PY) for the ASR venv"; return 1; }
  if [ ! -x "$ASR_VENV/bin/python" ]; then
    "$PY" -m venv "$ASR_VENV" >/dev/null 2>&1 \
      || { rm -rf "$ASR_VENV"; { [ -x /venv/main/bin/uv ] && /venv/main/bin/uv venv --python "$PY" "$ASR_VENV"; } \
           || uv venv --python "$PY" "$ASR_VENV"; } \
      || { echo "WARNING: could not create $ASR_VENV with $PY"; return 1; }
  fi
  local VPY="$ASR_VENV/bin/python"
  if "$VPY" -c "import faster_whisper, ctranslate2, numpy" >/dev/null 2>&1; then
    echo "faster-whisper already in $ASR_VENV: $("$VPY" -m pip show faster-whisper 2>/dev/null | sed -n 's/^Version: //p')"
  else
    local i ok=0
    for i in 1 2 3; do
      if "$VPY" -m pip --version >/dev/null 2>&1; then
        "$VPY" -m pip install -q "faster-whisper==${FW_PIN}" "ctranslate2==${CT2_PIN}" numpy && ok=1 && break
      else
        { /venv/main/bin/uv pip install -q --python "$VPY" "faster-whisper==${FW_PIN}" "ctranslate2==${CT2_PIN}" numpy \
          || uv pip install -q --python "$VPY" "faster-whisper==${FW_PIN}" "ctranslate2==${CT2_PIN}" numpy; } && ok=1 && break
      fi
      echo "  faster-whisper install attempt $i failed"; sleep 5
    done
    [ "$ok" = 1 ] || { echo "WARNING: faster-whisper pip install failed in $ASR_VENV (lyrics ASR unavailable; nothing else affected)"; return 1; }
  fi
  echo "--- prefetch + verify: import faster_whisper and load ${ASR_MODEL} (downloads Systran/faster-whisper-${ASR_MODEL} into the HF cache on first run) ---"
  local i ok=0
  for i in 1 2 3; do
    if "$VPY" -c "from faster_whisper import WhisperModel; import time; t=time.time(); WhisperModel('${ASR_MODEL}', device='cpu', compute_type='int8', cpu_threads=4); print('OK       WhisperModel(${ASR_MODEL}) loaded on CPU int8 in %.0f s' % (time.time()-t))"; then ok=1; break; fi
    echo "  model prefetch/load attempt $i failed"; sleep 10
  done
  [ "$ok" = 1 ] || { echo "WARNING: Whisper ${ASR_MODEL} model download/load failed (check HF reachability / disk); lyrics ASR unavailable, own-video route falls back to the no-lyrics path"; return 1; }
  du -sh "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--Systran--faster-whisper-${ASR_MODEL}" 2>/dev/null | sed 's/^/model cache: /'
  echo "Use: ${VPY} asr.py <16k mono wav> <out.json>   (asr.py: _info/box_pull_58200117/extra/OWNVIDEO/asr.py)"
  return 0
}

install_whisper
exit $?
