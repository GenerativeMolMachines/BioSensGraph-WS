from omegaconf import OmegaConf


def get_torchbiggraph_config():
    cfg = OmegaConf.load("params.yaml")

    config = dict(
        entity_path=cfg.entity_path,
        edge_paths=[
            cfg.edge_paths.train,
            cfg.edge_paths.test,
        ],
        checkpoint_path=cfg.train.checkpoint_path,
        checkpoint_preservation_interval=cfg.train.checkpoint_preservation_interval,
        entities=OmegaConf.to_container(cfg.entities, resolve=True),
        relations=(
            [OmegaConf.to_container(rel, resolve=True) for rel in cfg.relations]
            if OmegaConf.is_list(cfg.relations)
            else [OmegaConf.to_container(cfg.relations, resolve=True)]
        ) if cfg.relations else [],
        dimension=cfg.train.scoring_model.dimension,
        global_emb=cfg.train.global_emb,
        init_scale=cfg.train.init_scale,
        num_epochs=cfg.train.params.num_epochs,
        num_edge_chunks=cfg.train.params.num_edge_chunks,
        batch_size=cfg.train.params.batch_size,
        bucket_order=cfg.train.bucket_order,
        workers=cfg.train.workers,
        num_batch_negs=cfg.train.params.num_batch_negs,
        num_uniform_negs=cfg.train.params.num_uniform_negs,
        loss_fn=cfg.train.params.loss_fn,
        lr=cfg.train.params.lr,
        eval_fraction=cfg.train.eval_fraction,
        verbose=cfg.train.verbose,
        margin=cfg.train.params.margin
    )

    return config
