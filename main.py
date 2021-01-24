import gym
from common import get_params
from brain import Agent

if __name__ == "__main__":
    params = get_params()
    env = gym.make(params["env_name"])
    params.update({"n_states": env.observation_space.shape[0]})
    params.update({"n_actions": env.action_space.shape[0]})
    params.update({"action_bounds": [env.action_space.low[0], env.action_space.high[0]]})
    params.update({"max_episode_steps": env.spec.max_episode_steps})
    params.update({"max_episodes": params["max_steps"] // params["max_episode_steps"]})

    agent = Agent(**params)

    for episode in range(1, 1 + params["max_episodes"]):
        state = env.reset()

        for step in range(1, 1 + params["max_episode_steps"]):
            if episode * step <= params["pure_explore_steps"]:
                action = env.action_space.sample()
            else:
                action = agent.choose_action(state)

            next_state, reward, done, _ = env.step(action)
            agent.store(state, action, reward, done, next_state)
            agent.train()

            if done:
                break

            state = next_state
