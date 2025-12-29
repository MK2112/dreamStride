import env_wrapper
import spot_wrapper


def available_env_prefixes():
    return {
        "walker": "DeepMind Control Suite via env_wrapper.DeepMindControl",
        "spot": "WeBots Spot via spot_wrapper.SpotControl",
    }


def make_env(args):
    env = None

    if args.env.startswith("walker"):
        env = env_wrapper.DeepMindControl(args.env, args.seed)
    elif args.env.startswith("spot"):
        env = spot_wrapper.SpotControl(size=(64, 256))
    else:
        raise NotImplementedError(f"Unknown env '{args.env}'.")

    env = env_wrapper.ActionRepeat(env, args.action_repeat)
    env = env_wrapper.NormalizeActions(env)
    env = env_wrapper.TimeLimit(env, args.time_limit / args.action_repeat)
    return env
