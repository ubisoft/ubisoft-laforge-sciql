import hydra
import os
from omegaconf import DictConfig
from sciql.utils.imports import instantiate_class, get_class
from sciql.utils.loggers import CSVTensorBoardLogger

@hydra.main(version_base='1.2', config_path='yamls/diverse_mujoco/sciql_joint/jax', config_name='mujoco_halfcheetah')
def main(cfg: DictConfig) -> None:

    # create experiment folder (if it does not exist)
    #################################################
    if not os.path.exists(cfg.log_dir): os.makedirs(cfg.log_dir)

    # agents_db
    ############
    agents_db = instantiate_class(cfg.agent_cfg.agents_db)

    # loggers for training and evaluations
    #######################################
    evaluations = [instantiate_class(evaluation_cfg) for evaluation_cfg in cfg.evaluations_cfg]
    logger = CSVTensorBoardLogger(
        directory=os.path.join(cfg.log_dir,"logs"),
        prefix=None,
        log_to_csv=True,
        csv_names=['training'] + [evaluation.name for evaluation in evaluations]
    )
    logger.log_params(cfg)

    # training and evaluation
    #########################
    trainer = get_class(cfg.algo_cfg.trainer)
    agent_iterator = trainer.train(
        algo_cfg=cfg.algo_cfg,
        agent_cfg=cfg.agent_cfg,
        agents_db=agents_db,
        logger=logger
    )
    for agent, infos in agent_iterator:
        
        for evaluation in evaluations:
            evaluation.launch(agent, infos, logger)

    logger.close()
    print('Good job training done!')

if __name__ == "__main__":
    main()