# infinite-livestream

An end-to-end, chat-driven, never-ending AI video broadcast. Viewers type
`!prompt <idea>` in Twitch or YouTube chat; an LLM expands each idea into a
styled sequence of scenes; the fast-h3 model generates them as 768p video
clips with synchronized audio; and the stream goes out over RTMP as one
uninterrupted broadcast.

Two halves, one contract:

| Folder | What it is | Runs on |
| --- | --- | --- |
| [`fast-h3/`](./fast-h3) | The model: a queue of prompt-driven clip generations with explicit/auto playback. A `reactor` CLI workspace. | [Reactor Runtime](https://github.com/reactor-team/reactor-runtime), 8x B300 |
| [`streaming-client/`](./streaming-client) | The client: chat → prompt upsampling → scene groups → the model's queue → paced RTMP output. | `reactor-sdk` (Python), any box with ffmpeg |

They meet on the wire: `fast-h3/fasth3_types.py` is the client-facing
contract (commands, messages, tracks), and the streaming client speaks
exactly that.

## Quickstart

Serve the model (locally with `reactor run` from `fast-h3/`, or deploy it),
then:

```sh
cd streaming-client
pip install -r requirements.txt     # ffmpeg must be on PATH for RTMP
cp .env.example .env                # keys, style, sink, chat channel
python main.py --local --sink noop  # dry run against a local runtime
python main.py                      # everything from .env
```

Each folder's README carries its own full documentation. `AGENTS.md` is the
map for coding agents — read it before changing anything.

## License

Apache License 2.0 — see [LICENSE](./LICENSE) and [NOTICE](./NOTICE).
