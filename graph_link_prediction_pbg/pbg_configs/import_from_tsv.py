from omegaconf import OmegaConf


def get_torchbiggraph_config():
    cfg = OmegaConf.load("params.yaml")

    if OmegaConf.is_list(cfg.relations):
        relations_list = [OmegaConf.to_container(rel, resolve=True) for rel in cfg.relations]
    else:
        relations_list = [OmegaConf.to_container(cfg.relations, resolve=True)]

    config = dict(
        entity_path=cfg.entity_path,
        edge_paths=[
            cfg.edge_paths.train,
            cfg.edge_paths.test,
        ],
        entities=OmegaConf.to_container(cfg.entities, resolve=True),
        relations=relations_list,
        workers=cfg.import_data.workers,
    )

    return config
