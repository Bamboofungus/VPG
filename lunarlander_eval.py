import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
import torch
import os
from lunarlander_train import PolicyNN
import argparse

VIDEO_DIRECTORY = "./videos/"

#TODO move to file
def load_checkpoint(filepath, policy_class, policy_optimizer, value_class=None, value_optimizer=None):
    checkpoint = torch.load(filepath)
    policy_class.load_state_dict(checkpoint['policy_state_dict'])
    # policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
    # value_class.load_state_dict(checkpoint['f_approximator_state_dict'])
    # value_optimizer.load_state_dict(checkpoint['f_approximator_optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f'Checkpoint loaded from {filepath} (epoch {epoch})')
    return policy_class

def manual_lunarlander_policy(obs) -> int:
    """
    Naive policy that tries to orient lander legs toward ground and slow near the ground for comparison to
    trained policys
    """
    x, y, vx, vy, angle, angular_vel, left_contact, right_contact = obs
    # calculate current downward vector
    if left_contact or right_contact:
        return 0
    elif vy < -1 and y > 0.1:
        return 2
    else:
        if angle < -0.5:
            return 1
        elif angle > 0.5:
            return 3
    # default
    return 0

def evaluate(policy, episodes_to_run, will_record_video=False):
    env = gym.make("LunarLander-v2", render_mode="human")
    if will_record_video:
        env = RecordVideo(env, VIDEO_DIRECTORY)
    observation, _ = env.reset()
    for _ in range(episodes_to_run):
        terminated = truncated = False
        while not terminated or not truncated:
            action = policy(observation)
            observation, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                observation, _ = env.reset()

if __name__ == "__main__":
    policy_model = PolicyNN()

    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', type=str, help='Path to checkpoint containing policy')
    parser.add_argument('episodes_to_run', type=int, default=10, help='Episodes to evaluate policies')
    parser.add_argument('record_video', type=bool, default=False, help='Whether to record videos of the evaluation runs')

    args = parser.parse_args()
    filepath = args.filepath
    episodes_to_run = args.episodes_to_run
    record_video = args.record_video

    try:
        checkpoint = torch.load(filepath)
        assert 'policy_state_dict' in checkpoint, "Provided checkpoint missing policy!"
        policy_model.load_state_dict(checkpoint['policy_state_dict'])
        policy_model.eval()
    except:
        print("Failed to load policy from provided checkpoint!")
        quit()

    print("Evaluating manually coded policy")
    evaluate(manual_lunarlander_policy, episodes_to_run)
    print("Evaluating trained policy")
    evaluate(policy_model.act, episodes_to_run)
