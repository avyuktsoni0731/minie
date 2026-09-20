# Minie

Wake, speak, act. A local Mac voice agent: say **Hey Minie**, keep talking, and it starts safe computer-use actions before you finish the sentence. Your real cursor stays yours. Background clicks go through [Cua Driver](https://cua.ai/cua-driver).

## What v1 does

1. Always-on listener for **Hey Minie** (on-device Whisper tiny).
2. After wake: streaming speech-to-text, stop/cancel barge-in, speculative **open app** / Calculator math.
3. Multi-step UI work via Gemini Flash + window snapshots.
4. Menu bar status: Idle / Listening / Acting.

Until it hears the wake word it does not screenshot or click.

## Requirements

- Apple Silicon Mac, macOS 14+
- Python **3.12** (this repo pins it; system 3.14 is not used)
- [uv](https://docs.astral.sh/uv/)
- PortAudio: `brew install portaudio`
- A free Gemini key: https://aistudio.google.com/apikey
- Cua Driver for hands: https://cua.ai/docs/how-to-guides/driver/install

## Install

```bash
brew install portaudio
curl -LsSf https://astral.sh/uv/install.sh | sh

cd minie
cp .env.example .env   # paste GEMINI_API_KEY

uv python install 3.12
uv sync --extra dev

# Hands (once)
/bin/bash -c "$(curl -fsSL https://cua.ai/driver/install.sh)"
open -n -g -a CuaDriver --args serve
cua-driver permissions grant
```

Grant:

- **Microphone** to Terminal (or wherever you launch `minie`)
- **Accessibility** and **Screen Recording** to **CuaDriver.app** (not to Python)

```bash
uv run minie doctor
uv run minie run
```

`minie run` shows a menu-bar extra (`M · Idle`). Use `uv run minie run --no-menubar` for the terminal only.

First start downloads Whisper weights (tiny + small). Later starts reuse the cache.

## Demo

1. Say **Hey Minie** — chime / “yes?”
2. **Open Calculator and compute six times seven** — Calculator should show 42 without moving your cursor
3. Say **stop** mid-task to cancel
4. Watch the menu bar: Idle → Listening → Acting → Idle

## How it works

```
mic → “Hey Minie” → streaming ASR → safe partial actions
                                ↘ Gemini Flash + Cua Driver snapshot/click loop
```

- Speculative (no confirm): open an app, type a draft, Calculator keys.
- Needs a finished sentence + “yes”: send, delete, purchase, lock.
- Never types passwords.

## Config

| Variable | Default |
| --- | --- |
| `GEMINI_API_KEY` | required for the planner |
| `MINIE_WAKE_MODEL` | `mlx-community/whisper-tiny` |
| `MINIE_ASR_MODEL` | `mlx-community/whisper-tiny` (same as wake, avoids model swapping) |
| `MINIE_GEMINI_MODEL` | `gemini-3.6-flash` |
| `MINIE_DEBUG` | `0` |
| `MINIE_DRY_RUN` | `0` (log Cua calls, do not click) |

## Not in v1

Custom trained wake models, a fully local planner, duplex spoken chat, or acting with no wake word.
