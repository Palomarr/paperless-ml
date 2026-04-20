import yaml
from trainer.trainer import train
from trainer.mlflow_helper import setup_mlflow


def main():
    setup_mlflow()

    with open("configs/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    for model_type, model_config in config["models"].items():
        experiments = model_config.get("experiments", [])

        for exp in experiments:
            print(f"\n===== Running {model_type} experiment: {exp['name']} =====")
            train(model_type, exp, model_config)


if __name__ == "__main__":
    main()