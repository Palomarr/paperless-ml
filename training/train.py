import yaml
from trainer.trainer import train


def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    for exp in config["experiments"]:
        print(f"\n===== Running experiment: {exp['name']} =====")
        train(exp, config)


if __name__ == "__main__":
    main()