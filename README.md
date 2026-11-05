# robot-pizza-hack

ElRobot pizza policies from the Robotics Pizza Hackathon (Oct 2026): one SmolVLA conditioned on a
per-sector prompt, a failure checker that verifies every pepperoni step, and a runner that executes a
whole recipe on the arm through NormaCore Station.

Code only. Data, converted datasets and weights live outside the repo (see `.gitignore`):

| What | Where |
|---|---|
| Station recordings + manifests | `~/datasets/<src>.parquet`, `<src>.manifest.json` |
| Converted / merged datasets | `~/v2/station`, `~/v2/lerobot`, `~/v2/lerobot_compat/<variant>` |
| Policy checkpoints | `~/outputs/smolvla_v2_<variant>` |
| Checker models | `~/v2/checker/{roi.json, checker.pt, change.pt}` (reference `roi.json` is in `checker/`) |

## Layout

| Folder | Contents |
|---|---|
| `pipeline.sh` | Everything below in stages: `prep`, `policy`, `checkers` |
| `data/` | Prompts from manifests (`a_`), idle-frame cleaning (`b_`), merged training sets (`c_`), episode QA (`analyze.py`) |
| `checker/` | Sector ROIs (`d_`), topping checker (`e_`, `f_`), change checker (`g_`, `h_`), pizza locator, failure checker v2, its evaluation and docs |
| `robot/` | `run_challenge.py` (recipe runner with checker + retries), `run_lerobot_policy.py` (single policy, single prompt) |
| `v1/` | First experiment: cheese only, one ACT per quadrant or one SmolVLA |
| `norma/smolvla_py` | NormaCore SmolVLA trainer, patched to take the joint count from the data (8 on ElRobot) |

## Setup

    bash env/setup.sh          # conda env `lerobot` + ~/norma-core + ~/robot-pizza-cup, then env/check_env.py
    conda activate lerobot

- Python 3.10, `torch==2.7.1+cu128`, `lerobot[smolvla]==0.3.3` (datasets 3.6); every package is pinned in
  [env/requirements.lock](env/requirements.lock), top-level deps in [env/requirements.in](env/requirements.in).
- Needs an NVIDIA driver with CUDA 12.8 (>= 570); cu128 is also what RTX 50xx cards require.
- norma-core is cloned at the Station release used at the hackathon (`v0.1.0-beta.10`); an existing
  `~/norma-core` is kept (`NORMA_CORE_ROOT` to point elsewhere).
- Training decodes video with pyav (`--dataset.video_backend=pyav`): the torchcodec wheel that lerobot pulls in
  fails to load without the CUDA NPP libraries.
- `env/check_env.py` re-checks an env: GPU, lerobot policies, video decode, repo modules, Station client.

## Train

    bash pipeline.sh prep                    # ~/datasets -> ~/v2/lerobot_compat/{full,nogripper,nostate}
    VARIANT=full bash pipeline.sh policy     # one SmolVLA -> ~/outputs/smolvla_v2_full
    bash pipeline.sh checkers                # roi.json, checker.pt, change.pt

`prep` uses the organizers' exporter from [FunRobots/robot-pizza-cup](https://github.com/FunRobots/robot-pizza-cup)
(cloned to `~/robot-pizza-cup`). Cameras: cam0 = `top` (scene), cam1 = `front` (wrist).

## Run a challenge

`robot/run_challenge.py`: recipe -> steps (topping, sector) -> one prompt per step. After each pepperoni step
the failure checker ([checker/FAILURE_CHECKER_V2.md](checker/FAILURE_CHECKER_V2.md)) decides whether the step
is repeated (`--retries 2`).

    python robot/run_challenge.py ch2 --plan-only                       # steps and prompts, no robot
    python robot/run_challenge.py recipe --recipe "Build a pizza: ..." --plan-only
    python robot/run_challenge.py ch2 --checkpoint <pretrained_model> --bus-serial <FOLLOWER_SERIAL> \
        --checker-ref ~/v2/checker/ref_top.png --checker-log logs/run1  # add --dry-run to only locate the pizza

- Start with the arm at home: its pose at connect is the home pose, a step ends when the arm is back there.
- Prompts must match training exactly. The v2 prompts were in the manifests: pass `--prompts prompts.json`
  (`{"pepperoni:TL": "..."}`) or check with `--tasks-jsonl <dataset>/meta/tasks.jsonl`.
- `nogripper` / `nostate` checkpoints need `--zero-state-dims 7` / `--zero-state-dims 0,1,2,3,4,5,6,7`.
- Without `--checker-ref` the checker uses the fixed roi.json boxes (= v1).
- Needs norma-core next to it (`--norma-root`, default `~/norma-core`) for `station_py`.

## v1 (cheese only)

One episode = one quadrant, one round = 4 episodes clockwise: `ep` 1 = top-left, 2 = top-right,
3 = bottom-right, 4 = bottom-left. Prompt: `put the cheese on the {quadrant} quadrant of the pizza`.

    python v1/01_add_quadrant_prompts.py ~/datasets/cheese-fixed-s1-2026-10-24 ~/quadrant_src
    bash   v1/02_convert_to_lerobot.sh ~/quadrant_src ~/lerobot_cheese
    python v1/03_make_compat_datasets.py
    bash   v1/train_act.sh        # 4 ACT models -> ~/outputs/act_<quadrant>
    bash   v1/train_smolvla.sh    # 1 SmolVLA    -> ~/outputs/smolvla_quadrants

ACT ignores the prompt: run one checkpoint per quadrant with `robot/run_lerobot_policy.py --n-action-steps 10`
(data is 10 Hz). Round 3 episodes are flagged `cameras_swapped` in the manifest; verify visually.
