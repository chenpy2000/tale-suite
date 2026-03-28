import argparse
import datetime
import glob
import importlib
import json
import logging
import os
import sys
import time
from functools import partial
from os.path import join as pjoin

import gymnasium as gym
import pandas as pd
import wandb
from termcolor import colored
from tqdm import tqdm

import tales
from tales.logger import log, setup_logging
from tales.metrics import compute_doom_loop_count, compute_token_efficiency
from tales.utils import NumpyEncoder

os.environ["WANDB_MODE"] = "disabled"

def _append_goal_summary_once(agent, snapshot_log_path: str = None):
    if not snapshot_log_path:
        return
    long_term_goal = getattr(agent, "long_term_goal", None)
    if not long_term_goal:
        long_term_goal = "(none)"

    msg = (
        "\n=== Long-Term Goal Summary ===\n"
        f"{long_term_goal}\n"
        "==============================\n"
    )
    with open(snapshot_log_path, "a", encoding="utf-8") as f:
        f.write(msg)

def _append_map_summary_snapshot(step: int, stats: dict, snapshot_log_path: str = None):
    if not snapshot_log_path:
        return

    map_summary = (stats or {}).get("map_summary")
    map_summary = str(map_summary).strip() if map_summary else "(empty)"

    msg = (
        f"\n=== Map Summary @ step {step} ===\n"
        f"{map_summary}\n"
        f"=================================\n"
    )
    with open(snapshot_log_path, "a", encoding="utf-8") as f:
        f.write(msg)

def _print_planning_snapshot(step: int, agent, snapshot_log_path: str = None):
    # Agent may not have these fields; print only if present
    if not hasattr(agent, "current_task"):
        return

    current_task = getattr(agent, "current_task", None)
    completed_and_useful = sorted(
        getattr(agent, "completed_and_useful_tasks", getattr(agent, "success_tasks", []))
    )
    completed_but_useless = sorted(getattr(agent, "completed_but_useless_tasks", []))
    failed = sorted(getattr(agent, "failed_tasks", []))
    unprog = sorted(getattr(agent, "unprogressed_tasks", []))
    active_task_start_step = getattr(agent, "current_task_start_step", None)
    current_location = getattr(agent, "current_location", None)

    msg = (
        f"\n--- Planning Snapshot @ step {step} ---\n"
        + (f"Active task start step: {active_task_start_step}\n" if active_task_start_step is not None else "")
        + (f"Current location:\n  - {current_location}\n" if current_location else "Current location:\n  - (unknown)\n")
        + f"Current task:\n  - {current_task}\n"
        + f"All previous completed_and_useful tasks ({len(completed_and_useful)}):\n  - "
        + ("\n  - ".join(completed_and_useful) if completed_and_useful else "(none)")
        + "\n"
        + f"All previous completed_but_useless tasks ({len(completed_but_useless)}):\n  - "
        + ("\n  - ".join(completed_but_useless) if completed_but_useless else "(none)")
        + "\n"
        + f"All previous failed tasks ({len(failed)}):\n  - "
        + ("\n  - ".join(failed) if failed else "(none)")
        + "\n"
        + f"All previous unprogressed tasks ({len(unprog)}):\n  - "
        + ("\n  - ".join(unprog) if unprog else "(none)")
        + "\n"
        + f"--------------------------------------\n"
    )
    if snapshot_log_path:
        with open(snapshot_log_path, "a", encoding="utf-8") as f:
            f.write(msg)


def _append_plan_act_snapshot(step: int, stats: dict, snapshot_log_path: str = None):
    if not snapshot_log_path:
        return

    plan_output = (stats or {}).get("plan_response")
    act_output = (stats or {}).get("act_response")
    plan_output = str(plan_output).strip() if plan_output else "(empty)"
    act_output = str(act_output).strip() if act_output else "(empty)"

    msg = (
        f"\n=== Planner Output Snapshot @ step {step} ===\n"
        f"{plan_output}\n"
        f"=== Actor Output Snapshot @ step {step} ===\n"
        f"{act_output}\n"
        f"============================================\n"
    )
    with open(snapshot_log_path, "a", encoding="utf-8") as f:
        f.write(msg)


def evaluate(agent, env_name, args):
    env_params = (
        f"a{int(args.admissible_commands)}_s{args.game_seed}_steps{args.nb_steps}"
    )
    logdir = pjoin(args.log_dir, f"{env_name}")
    os.makedirs(logdir, exist_ok=True)
    summary_file = pjoin(logdir, f"{env_params}.json")
    rollouts_file = pjoin(logdir, f"{env_params}.jsonl")
    log_file = pjoin(logdir, f"{env_params}.log")
    snapshot_log_file = pjoin(logdir, "planning_snapshots.log")
    open(snapshot_log_file, "w", encoding="utf-8").close()

    # Create new file handler for this env evaluation.
    fh = log.add_new_file_handler(log_file)

    # Check if the game has already been evaluated.
    if not args.force_all and os.path.exists(summary_file):
        log.info(f"Previous evaluation found: {summary_file}")
        with open(summary_file) as reader:
            summary = json.load(reader)

        log.info(f"Previous evaluation status: {summary['status']}")
        if not args.force_failed or summary["status"] == "finished":
            log.info(colored("Skipped, already done.", "yellow"))
            log.removeHandler(fh)
            return summary

    run_name = f"{env_name} - {agent.uid}"
    if args.wandb and not args.force_all:
        wandb_api = wandb.Api()
        wandb_runs = wandb_api.runs(filters={"display_name": run_name})
        if wandb_runs:
            wandb_run = wandb_runs[0]
            log.info(f"Previous evaluation found: {wandb_run.url} ({wandb_run.state})")
            if wandb_run.state in ("finished", "running"):
                log.info(colored("Skipped, already exists.", "yellow"))
                log.removeHandler(fh)
                summary = {
                    "status": wandb_run.state,
                    "env_name": env_name,
                    "env_params": env_params,
                    "wandb_run_id": wandb_run.id,
                    "wandb_url": wandb_run.url,
                    "nb_steps": wandb_run.summary["total/Env. Steps"],
                    "nb_moves": wandb_run.summary["total/Game Moves"],
                    "nb_invalid_actions": wandb_run.summary["total/Invalid Actions"],
                    "nb_losts": wandb_run.summary["total/Losts"],
                    "nb_wins": wandb_run.summary["total/Wins"],
                    "nb_resets": wandb_run.summary["total/Resets"],
                    "highscore": wandb_run.summary["final/Highscore"],
                    "max_score": wandb_run.summary["final/Game Max Score"],
                    "norm_score": wandb_run.summary["final/Normalized Score"],
                    "duration": wandb_run.summary["final/Duration"],
                    "token_efficiency": wandb_run.summary.get("final/Token Efficiency"),
                    "doom_loop_count": wandb_run.summary.get("final/Doom Loop Count"),
                }
                return summary

    # initialize wandb
    wandb_config = {
        "version": tales.__version__,
        "game": env_name,
        "framework": tales.env2task[env_name],
        "agent": agent.uid,
        "max_steps": args.nb_steps,
        "game_seed": args.game_seed,
        "admissible_commands": args.admissible_commands,
        **agent.params,
    }
    wandb_run = wandb.init(
        project="tales",
        config=wandb_config,
        reinit=True,
        name=run_name,
    )

    env = gym.make(
        f"tales/{env_name}-v0",
        disable_env_checker=True,
        admissible_commands=args.admissible_commands,
    )

    log.debug(f"Using {env.__class__.__name__}")
    log.debug(f"Playing {env_name}")

    start_time = time.time()
    obs, info = env.reset(seed=args.game_seed)

    agent = agent.new()
    agent.reset(obs, info, env_name)

    log.debug(f"Environment reset.\n{obs}\n")

    status = "running"
    max_score = info["max_score"]
    step = 0
    nb_resets = 0
    nb_wins = 0
    nb_losts = 0
    nb_invalid_actions = 0
    moves = 0
    highscore = 0
    score = 0
    done = False
    results = []

    wandb_run.log(
        {
            "episode/moves": moves,
            "episode/score": score,
            "episode/highscore": highscore,
            "episode/normalized_score": score / max_score,
            "episode/normalized_highscore": highscore / max_score,
            "episode/token_usage": 0,
            "episode/token_usage_plan": 0,
            "episode/token_usage_act": 0,
        },
        step=0,
    )

    try:
        pbar = tqdm(
            range(1, args.nb_steps + 1), desc=f"  {env_name}", unit="steps", leave=False
        )

        for step in pbar:
            pbar.set_postfix_str(
                f"Score: {info['score']}/{info['max_score']} ({info['score']/info['max_score']:.1%})"
            )

            action, stats = agent.act(obs, score, done, info)
            log.debug(colored(f"> {action}", "green"))

            if args.debug:
                breakpoint()

            prev_obs = obs

            # Force one action per step.
            if "\n" in action.strip():
                obs = "The game only allows one action per step."
            else:
                obs, _, done, info = env.step(action)

            score = info["score"]
            moves = info["moves"]
            feedback = info["feedback"]
            norm_score = score / max_score
            highscore = max(score, highscore)
            norm_highscore = highscore / max_score

            if (
                args.admissible_commands
                and info.get("admissible_commands")
                and action not in info["admissible_commands"]
            ):
                nb_invalid_actions += 1

            msg = "{:5d}. Time: {:9.2f}\tScore: {:3d}\tMove: {:5d}\tAction: {:20s}"
            msg = msg.format(step, time.time() - start_time, score, moves, action)
            log.info(msg)

            if step == 1:
                _append_goal_summary_once(agent, snapshot_log_file)

            # Every 5 steps: print planning snapshot
            if step % 5 == 0:
                _append_map_summary_snapshot(step, stats, snapshot_log_file)
                _print_planning_snapshot(step, agent, snapshot_log_file)
                _append_plan_act_snapshot(step, stats, snapshot_log_file)

            wandb_run.log(
                {
                    "episode/moves": moves,
                    "episode/score": score,
                    "episode/highscore": highscore,
                    "episode/normalized_score": norm_score,
                    "episode/normalized_highscore": norm_highscore,
                    "episode/token_usage": stats.get("nb_tokens", 0),
                    "episode/token_usage_plan": stats.get("nb_tokens_plan", 0),
                    "episode/token_usage_act": stats.get("nb_tokens_act", 0),
                },
                step=step,
            )

            # Store extra plan/act info for analysis
            results.append(
                [
                    step,
                    score,
                    max_score,
                    norm_score,
                    moves,
                    prev_obs,
                    action,
                    feedback,
                    # Planning snapshots
                    stats.get("current_location"),
                    stats.get("current_task"),
                    json.dumps(stats.get("success_tasks", []), ensure_ascii=False),
                    json.dumps(stats.get("failed_tasks", []), ensure_ascii=False),
                    json.dumps(stats.get("unprogressed_tasks", []), ensure_ascii=False),
                    # Planner I/O
                    stats.get("plan_prompt"),
                    stats.get("plan_response"),
                    stats.get("plan_old_state"),
                    stats.get("plan_old_task"),
                    stats.get("plan_new_task"),
                    json.dumps(stats.get("plan_candidates", []), ensure_ascii=False),
                    stats.get("nb_tokens_plan_prompt", 0),
                    stats.get("nb_tokens_plan_response", 0),
                    # Actor I/O
                    stats.get("act_prompt"),
                    stats.get("act_response"),
                    stats.get("thinking"),
                    stats.get("nb_tokens_act_prompt", 0),
                    stats.get("nb_tokens_act_response", 0),
                    # Totals
                    stats.get("nb_tokens", 0),
                ]
            )

            if not done:
                log.debug(obs)

            if done:
                if info.get("won"):
                    nb_wins += 1
                    if highscore == max_score:
                        log.debug(obs)
                        break  # No reason to play that game more.
                elif info.get("lost"):
                    nb_losts += 1

                # Replay the game in the hope of achieving a better score.
                last_obs = obs
                obs, info = env.reset()
                obs = last_obs + "\n\n-= Restarting =-\n" + obs
                agent.reset(obs, info, env_name)
                nb_resets += 1

                log.debug(f"{obs}")

        status = "finished"

    except KeyboardInterrupt as e:
        status = "killed"
        log.critical(colored(f"{env_name} (killed)", "red"))
        log.error(str(e))
        time.sleep(1)
        if args.debug:
            raise

    except Exception as e:
        status = "failed"
        log.critical(colored(f"{env_name} ({e!r})", "red"))
        log.error(str(e), exc_info=True)
        if args.debug:
            raise

    env.close()

    stats = {
        "nb_steps": step,
        "nb_moves": moves,
        "nb_invalid_actions": nb_invalid_actions,
        "nb_losts": nb_losts,
        "nb_wins": nb_wins,
        "nb_resets": nb_resets,
        "highscore": highscore,
        "max_score": max_score,
        "norm_score": highscore / max_score,
        "duration": time.time() - start_time,
    }

    columns = [
        "Step",
        "Score",
        "Max Score",
        "Normalized Score",
        "Moves",
        "Observation",
        "Action",
        "Feedback",
        # Planning snapshots
        "Current Location",
        "Current Task",
        "Success Tasks",
        "Failed Tasks",
        "Unprogressed Tasks",
        # Planner I/O
        "Plan Prompt",
        "Plan Response",
        "Old Task State",
        "Old Task",
        "New Task",
        "Plan Candidates",
        "Plan Prompt Tokens",
        "Plan Response Tokens",
        # Actor I/O
        "Act Prompt",
        "Act Response",
        "Thinking",
        "Act Prompt Tokens",
        "Act Response Tokens",
        # Totals
        "Token Usage Total",
    ]

    df = pd.DataFrame(results, columns=columns)

    total_tokens = int(df["Token Usage Total"].sum()) if len(df) else 0
    token_efficiency = compute_token_efficiency(total_tokens, highscore)
    doom_loop_count = compute_doom_loop_count(df)
    stats["token_efficiency"] = token_efficiency
    stats["doom_loop_count"] = doom_loop_count

    df.to_json(rollouts_file, orient="records", lines=True)

    wandb_stats = {
        "total/Env. Steps": stats["nb_steps"],
        "total/Game Moves": stats["nb_moves"],
        "total/Invalid Actions": stats["nb_invalid_actions"],
        "total/Losts": stats["nb_losts"],
        "total/Wins": stats["nb_wins"],
        "total/Resets": stats["nb_resets"],
        "total/Tokens": total_tokens,
        "final/Highscore": stats["highscore"],
        "final/Game Max Score": stats["max_score"],
        "final/Normalized Score": stats["norm_score"],
        "final/Duration": stats["duration"],
        "final/Token Efficiency": stats["token_efficiency"],
        "final/Doom Loop Count": stats["doom_loop_count"],
    }

    wandb_run.log(
        {"episode/rollout": wandb.Table(dataframe=df), **wandb_stats},
        step=stats["nb_steps"],
    )

    summary = {
        "status": status,
        "env_name": env_name,
        "env_params": env_params,
        "wandb_run_id": wandb_run.id,
        "wandb_url": wandb_run.url,
        **stats,
        **wandb_stats,
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, cls=NumpyEncoder)

    wandb.save(rollouts_file)
    wandb.save(log_file)
    wandb.save(summary_file)

    wandb_run.finish(exit_code=int(status != "finished"))

    log.removeHandler(fh)
    return summary


def benchmark(agent, args):
    log.critical("Evaluating {} interactive text environments:".format(len(args.envs)))

    mean_score = 0
    total_time = 0.0
    total_steps = 0
    total_invalid = 0

    nb_envs = 0
    max_env_name = max(map(len, args.envs))
    for env in tqdm(args.envs, desc="Benchmarking", unit="game", leave=False):
        summary = evaluate(agent, env, args)

        nb_envs += 1
        total_time += summary["duration"]
        total_steps += summary["nb_steps"]
        total_invalid += summary["nb_invalid_actions"]
        token_eff = summary.get("token_efficiency", summary.get("final/Token Efficiency"))
        doom_loop = summary.get("doom_loop_count", summary.get("final/Doom Loop Count"))
        token_eff_str = f"{float(token_eff):8.2f}" if token_eff is not None else "   n/a  "
        doom_loop_str = f"{int(doom_loop):4d}" if doom_loop is not None else " n/a"

        msg = (
            f"{env.ljust(max_env_name)}"
            f"  Steps: {summary['nb_steps']:4d}/{args.nb_steps:4d}"
            f"  Time: {datetime.timedelta(seconds=int(summary['duration']))}"
            f"{summary['nb_resets']:4d} resets"
            f"  Score: {summary['highscore']:3d}/{summary['max_score']:3d} ({summary['norm_score']:6.2%})"
            f"  TokenEff: {token_eff_str}"
            f"  DoomLoop: {doom_loop_str}"
        )
        log.critical(msg)
        mean_score += summary["norm_score"]

    if nb_envs > 0 and total_time > 0:
        log.critical(
            f"Mean score (over {nb_envs} games) = {mean_score / nb_envs:8.2%} of total possible"
        )
        log.critical(f"Total time {total_time:9.2f} seconds")
        log.critical(f"Total {total_invalid} invalid actions")
        log.critical(f"Avg. speed: {total_steps / total_time:8.2f} steps per second")


def pretty_print_tasks(num_cols: int = 3, disable_print: bool = False):
    output = []

    max_justify = max(
        len(env_name) for task in tales.envs_per_task.values() for env_name in task
    )

    for task in sorted(tales.envs_per_task):
        task_output = f"{'=' * 5} {task} {'=' * 5}\n"

        for count, env_id in enumerate(sorted(tales.envs_per_task[task]), 1):
            task_output += env_id.ljust(max_justify) + " "
            if count % num_cols == 0:
                task_output = task_output.rstrip(" ")
                if count != len(tales.envs_per_task[task]):
                    task_output += "\n"

        output.append(task_output.rstrip(" "))

    if disable_print:
        return "\n".join(output)
    else:
        print("\n".join(output))


def exit_listing_agents(agent=None):
    msg = ""
    if agent is not None:
        msg += "Unknown agent: {}\n\n".format(agent)

    msg += "Available agents:\n  "
    msg += "\n  ".join(sorted(tales.agent.AGENTS))
    print(msg)
    sys.exit(1)


def _maybe_load_agent_module():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--agent",
        default="agents/*.py",
        help="Load external python file(s). Useful to register custom agent on-the-fly. Default: %(default)s",
    )
    args, _ = parser.parse_known_args()
    if args.agent:
        print(f"Importing agent(s) from {args.agent}.")
        for agent_file in glob.glob(args.agent):
            agent_dirname = os.path.dirname(agent_file)
            agent_filename, _ = os.path.splitext(os.path.basename(agent_file))
            if f"{agent_dirname}.{agent_filename}" in sys.modules:
                continue

            spec = importlib.util.spec_from_file_location(
                f"{agent_dirname}.{agent_filename}", agent_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)


def parse_args():
    description = "Benchmark some agent on interactive text environments (planning variant)."
    general_parser = argparse.ArgumentParser(add_help=False, description=description)
    general_parser.add_argument(
        "--agent",
        default="./agents/*",
        help="Load external python file(s). Useful to register custom agent on-the-fly. Default: %(default)s",
    )

    parser = argparse.ArgumentParser(parents=[general_parser])
    subparsers = parser.add_subparsers(dest="subcommand", title="Available agents to benchmark")

    def _add_general_settings(p):
        p.formatter_class = argparse.RawTextHelpFormatter
        general_group = p.add_argument_group("General settings")

        general_group.add_argument(
            "--envs",
            metavar="env",
            nargs="+",
            choices=tales.envs + tales.tasks,
            help="Interactive text environments to evaluate the agent(s)."
                 f" Available:\n{pretty_print_tasks(disable_print=True)}",
        )
        general_group.add_argument("--game-seed", type=int, help="Seed for the game. Default: game-specific one.")
        general_group.add_argument("--nb-steps", type=int, default=100, help="Maximum number of steps per game.")
        general_group.add_argument("--admissible-commands", action="store_true", help="Enable admissible commands.")
        general_group.add_argument("--log-dir", default="logs", help="Folder where to save verbose log information.")
        general_group.add_argument("--wandb", action="store_true", help="Log to wandb")
        general_group.add_argument("-ff", "--force-all", action="store_true", help="Force overwriting existing log files.")
        general_group.add_argument("-f", "--force-failed", action="store_true", help="Force overwriting only log files that have failed.")
        general_group.add_argument("--debug", action="store_true", help="Debug mode.")

        subgroup = general_group.add_mutually_exclusive_group()
        subgroup.add_argument(
            "-v", "--verbose", dest="logging_level",
            action="store_const", const=logging.INFO, default=logging.CRITICAL,
            help="Display actions taken.",
        )
        subgroup.add_argument(
            "-vv", "--very-verbose", dest="logging_level",
            action="store_const", const=logging.DEBUG, default=logging.CRITICAL,
            help="Display actions and game observations.",
        )
        subgroup.add_argument(
            "--logging-level", dest="logging_level", default=logging.CRITICAL,
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Set a specific logging level",
        )

    _add_general_settings(parser)

    for agent_name, (desc, _, add_agent_arguments) in tales.agent.AGENTS.items():
        agent_parser = subparsers.add_parser(agent_name, help=desc)
        add_agent_arguments(agent_parser)
        _add_general_settings(agent_parser)

    return parser.parse_args()


def main():
    _maybe_load_agent_module()
    args = parse_args()

    if args.subcommand is None:
        print("Need to specify which type of agent to benchmark.")
        exit_listing_agents(args.subcommand)

    _, Agent, _ = tales.agent.AGENTS[args.subcommand]
    agent = Agent(**vars(args))
    agent.new = partial(Agent, **vars(args))

    args.log_dir = pjoin(args.log_dir, f"tales_{agent.uid.replace('/', '-')}_plan")
    os.makedirs(args.log_dir, exist_ok=True)
    setup_logging(args)
    log.critical(colored(f"Logs will be saved in {os.path.abspath(args.log_dir)}", "magenta"))

    if args.wandb:
        os.environ["WANDB_MODE"] = "online"
        os.environ.pop("WANDB_RUN_ID", None)

    args.envs = args.envs or tales.envs
    args.envs = [
        env
        for task in args.envs
        for env in (tales.envs_per_task[task] if task in tales.tasks else [task])
    ]

    benchmark(agent, args)


if __name__ == "__main__":
    main()
