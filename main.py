import gym
from common import get_params
from brain import Agent
import psutil
from torch.utils.tensorboard import SummaryWriter
import time
import mujoco_py

if __name__ == "__main__":
    params = get_params()
    env = gym.make(params["env_name"])
    params.update({"n_states": env.observation_space.shape[0]})
    params.update({"n_actions": env.action_space.shape[0]})
    params.update({"action_bounds": [env.action_space.low[0], env.action_space.high[0]]})
    params.update({"max_episode_steps": env.spec.max_episode_steps})
    params.update({"max_episodes": params["max_steps"] // params["max_episode_steps"]})
    print("params:", params)

    agent = Agent(**params)
    to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024

    explore_steps = 0
    running_reward = 0
    if params["do_train"]:
        for episode in range(1, 1 + params["max_episodes"]):
            state = env.reset()
            episode_reward = 0
            for step in range(1, 1 + params["max_episode_steps"]):
                if explore_steps <= params["pure_explore_steps"]:
                    action = env.action_space.sample()
                    explore_steps += 1
                else:
                    action = agent.choose_action(state)

                next_state, reward, done, _ = env.step(action)
                agent.store(state, action, reward, done, next_state)
                agent.train()

                episode_reward += reward
                if done:
                    break
                state = next_state

            if episode == 1:
                running_reward = episode_reward
            else:
                running_reward = 0.99 * running_reward + 0.01 * episode_reward

            with SummaryWriter(params["env_name"] + "/logs/") as writer:
                writer.add_scalar("Episode running reward", running_reward, episode)
                writer.add_scalar("Episode reward", episode_reward, episode)

            if episode % params["interval"] == 0:
                agent.save_weights()
                ram = psutil.virtual_memory()
                print(f"E: {episode}| "
                      f"Reward: {episode_reward:.2f}| "
                      f"Running_Reward: {running_reward:.2f}| "
                      f"Memory length: {len(agent.memory)}| "
                      f"{to_gb(ram.used):.1f}/{to_gb(ram.total):.1f} GB RAM")

    else:
        agent.load_weights()
        env = gym.wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: True, force=True)
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.choose_action(state, eval=True)
            next_state, reward, done, _ = env.step(action)
            env.render()
            time.sleep(0.01)
            episode_reward += reward
            state = next_state

        print(f"Episode reward: {episode_reward: .1f}")
