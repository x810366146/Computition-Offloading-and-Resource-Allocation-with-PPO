from typing import Callable

import gymnasium as gym
import torch
import torch.nn as nn


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    actor = Model[0](envs).to(device)
    qf1 = Model[1](envs).to(device)
    qf2 = Model[1](envs).to(device)
    actor_params, qf1_params, qf2_params = torch.load(model_path, map_location=device)
    actor.load_state_dict(actor_params)
    actor.eval()
    qf1.load_state_dict(qf1_params)
    qf2.load_state_dict(qf2_params)
    qf1.eval()
    qf2.eval()
    # note: qf1 and qf2 are not used in this script

    obs, _ = envs.reset()
    episodic_returns = []
    average_delays = []
    average_energy_cons = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
                if info and "average_delay" in info:
                    print(f"eval_episode={len(episodic_returns)}, average_delay={info['average_delay']}")
                    average_delays += [info['average_delay']]
                if info and "average_energy_cons" in info:
                    print(f"eval_episode={len(episodic_returns)}, average_energy_cons={info['average_energy_cons']}")
                    average_energy_cons += [info['average_energy_cons']]
        obs = next_obs

    return episodic_returns, average_delays, average_energy_cons


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    from cleanrl.td3_continuous_action import Actor, QNetwork, make_env

    model_path = hf_hub_download(
        repo_id="cleanrl/HalfCheetah-v4-td3_continuous_action-seed1", filename="td3_continuous_action.cleanrl_model"
    )
    evaluate(
        model_path,
        make_env,
        "HalfCheetah-v4",
        eval_episodes=10,
        run_name=f"eval",
        Model=(Actor, QNetwork),
        device="cpu",
        capture_video=False,
    )
