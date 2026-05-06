from omegaconf import OmegaConf


def main():
    cfg = OmegaConf.load("params.yaml")
    print(cfg)


if __name__ == "__main__":
    main()
