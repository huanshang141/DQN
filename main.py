import argparse
import gymnasium as gym
from argument import dqn_arguments


def parse():
    parser = argparse.ArgumentParser(description="SYSU_RL_HW2")
    parser.add_argument('--train_dqn', default=True, type=bool, help='whether train DQN')
    
    parser = dqn_arguments(parser)
    args = parser.parse_args()
    return args


def run(args):
    if args.train_dqn:
        env_name = args.env_name
        env = gym.make(env_name)
        from agent_dir.agent_dqn import AgentDQN
        agent = AgentDQN(env, args)
        agent.run()


if __name__ == '__main__':
    args = parse()
    run(args)
