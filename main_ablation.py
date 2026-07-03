import copy
import torch
from causal_nf.preparators.scm import SCMPreparator
import glob
import wandb
import os
import causal_nf.config as causal_nf_config
import causal_nf.utils.training as causal_nf_train
import causal_nf.utils.wandb_local as wandb_local
from causal_nf.config import cfg
import causal_nf.utils.io as causal_nf_io
from datetime import datetime
import numpy as np
import json
from fans import FANSAnalyzer
from fans_ablation import FANSAblationAnalyzer

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'text.usetex': False})

os.environ["WANDB_NOTEBOOK_NAME"] = "causal_nf"


def convert_for_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


if __name__ == "__main__":
    start_time = datetime.now()
    args_list, args = causal_nf_config.parse_args()

    load_model = isinstance(args.load_model, str)
    if load_model:
        causal_nf_io.print_info(f"Loading model: {args.load_model}")

    config = causal_nf_config.build_config(
        config_file=args.config_file,
        args_list=args_list,
        config_default_file=args.config_default_file,
    )

    # Extract shifted_nodes from metadata
    shifted_nodes = None
    metadata_json = None
    if hasattr(cfg.dataset, 'metadata_path') and cfg.dataset.metadata_path:
        metadata_path = cfg.dataset.metadata_path
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata_json = json.load(f)
                shifted_nodes = metadata_json.get('shifted_nodes', None)
                causal_nf_io.print_info(f"Loaded shifted_nodes from metadata: {shifted_nodes}")

    causal_nf_config.assert_cfg_and_config(cfg, config)

    if cfg.device in ["cpu"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    causal_nf_train.set_reproducibility(cfg)

    # ============================================================
    # ENV1 model (same as main.py)
    # ============================================================
    causal_nf_io.print_info("=== Building ENV1 model ===")
    preparator = SCMPreparator.loader(cfg.dataset)
    preparator.prepare_data()

    loaders = preparator.get_dataloaders(
        batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers
    )

    for i, loader in enumerate(loaders):
        causal_nf_io.print_info(f"[{i}] num_batches: {len(loader)}")

    model = causal_nf_train.load_model(cfg=cfg, preparator=preparator)

    param_count = model.param_count()
    config["param_count"] = param_count

    if not load_model:
        assert isinstance(args.project, str)
        run = wandb.init(
            mode=args.wandb_mode,
            group=args.wandb_group,
            project=args.project,
            config=config,
            dir="/data1/statduck/"
        )

        import uuid
        if args.wandb_mode != "disabled":
            run_uuid = run.id
        else:
            run_uuid = str(uuid.uuid1()).replace("-", "")
    else:
        run_uuid = os.path.basename(args.load_model)

    external_dag_path = cfg.dataset.external_dag_path
    if external_dag_path.startswith("data/"):
        external_dag_path = external_dag_path[5:]
    if external_dag_path.endswith(".npy"):
        external_dag_path = external_dag_path[:-4]
    folder_name = external_dag_path.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")
    folder_name = "_".join(filter(None, folder_name.split("_")))

    if not load_model:
        base_dir = os.path.join(cfg.root_dir, folder_name)
        if os.path.exists(base_dir):
            existing_runs = [d for d in os.listdir(base_dir)
                            if os.path.isdir(os.path.join(base_dir, d))]
            for run_id in existing_runs:
                run_path = os.path.join(base_dir, run_id)
                ckpt_files = glob.glob(os.path.join(run_path, "*.ckpt"))
                if ckpt_files:
                    causal_nf_io.print_info(f"Found existing checkpoint in: {run_path}")
                    causal_nf_io.print_info(f"Using existing model instead of training")
                    args.load_model = run_path
                    load_model = True
                    run_uuid = os.path.basename(run_path)
                    break

    dirpath = os.path.join(cfg.root_dir, folder_name, run_uuid)
    logger_dir = os.path.join(cfg.root_dir, folder_name, run_uuid)

    trainer, logger = causal_nf_train.load_trainer(
        cfg=cfg,
        dirpath=dirpath,
        logger_dir=logger_dir,
        include_logger=True,
        model_checkpoint=cfg.train.model_checkpoint,
        cfg_early=cfg.early_stopping,
        preparator=preparator,
    )

    causal_nf_io.print_info(f"Experiment folder: {logger.save_dir}\n\n")
    wandb_local.log_config(dict(config), root=logger.save_dir)

    if not load_model:
        wandb_local.copy_config(
            config_default=causal_nf_config.DEFAULT_CONFIG_FILE,
            config_experiment=args.config_file,
            root=logger.save_dir,
        )
        trainer.fit(model, train_dataloaders=loaders[0], val_dataloaders=loaders[1])

    if isinstance(preparator.single_split, str):
        loaders = [loaders[0]]

    model.save_dir = dirpath

    if load_model:
        ckpt_name_list = [glob.glob(os.path.join(args.load_model, f"*ckpt"))]
        ckpt_name_list = [os.path.join(args.load_model, "last.ckpt")]
        for ckpt_file in ckpt_name_list:
            model = causal_nf_train.load_model(
                cfg=cfg, preparator=preparator, ckpt_file=ckpt_file
            )
            model.eval()
            model.save_dir = dirpath
            ckpt_name = preparator.get_ckpt_name(ckpt_file)
            for i, loader_i in enumerate(loaders):
                s_name = preparator.split_names[i]
                causal_nf_io.print_info(f"Testing {s_name} split")
                preparator.set_current_split(i)
                model.ckpt_name = ckpt_name
                _ = trainer.test(model=model, dataloaders=loader_i)
                metrics_stats = model.metrics_stats
                metrics_stats["current_epoch"] = trainer.current_epoch
                wandb_local.log_v2(
                    {s_name: metrics_stats, "epoch": ckpt_name},
                    root=trainer.logger.save_dir,
                )
    else:
        ckpt_name_list = ["last"]
        if cfg.early_stopping.activate:
            ckpt_name_list.append("best")
        for ckpt_name in ckpt_name_list:
            for i, loader_i in enumerate(loaders):
                s_name = preparator.split_names[i]
                causal_nf_io.print_info(f"Testing {s_name} split")
                preparator.set_current_split(i)
                model.ckpt_name = ckpt_name
                _ = trainer.test(ckpt_path=ckpt_name, dataloaders=loader_i)
                metrics_stats = model.metrics_stats
                metrics_stats["current_epoch"] = trainer.current_epoch
                wandb_local.log_v2(
                    {s_name: metrics_stats, "epoch": ckpt_name},
                    root=trainer.logger.save_dir,
                )

        run.finish()
        if args.delete_ckpt:
            for f in glob.iglob(os.path.join(logger.save_dir, "*.ckpt")):
                causal_nf_io.print_warning(f"Deleting {f}")
                os.remove(f)

    # ============================================================
    # FANS Analysis (same as main.py)
    # ============================================================
    causal_nf_io.print_info("Starting FANS analysis...")

    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    print('model.device: ', model.device)
    fans_analyzer = FANSAnalyzer(model.model, model.preparator, model.device, model.input_scaler,
                                 enable_visualizer=False)

    fans_output_dir = os.path.join(logger.save_dir, "fans_analysis")
    os.makedirs(fans_output_dir, exist_ok=True)

    test_data_env1 = torch.tensor(np.load(cfg.dataset.test_data_path), dtype=torch.float32)
    test_data_env2 = torch.tensor(np.load(cfg.dataset.env2_test_data_path), dtype=torch.float32)

    fans_results = fans_analyzer.analyze(
        test_data_env1, test_data_env2,
        save_dir=fans_output_dir,
        simultaneous_shift=True,
        shifted_nodes=shifted_nodes
    )

    fans_results_json = convert_for_json(fans_results)
    if not load_model:
        wandb_run_pattern = f"/data1/statduck/wandb/run-*-{run_uuid}/files/wandb-summary.json"
        matching_paths = glob.glob(wandb_run_pattern)
        wandb_summary_path = matching_paths[0]
        with open(wandb_summary_path, 'r') as f:
            wandb_summary = json.load(f)
            fans_results_json['runtime'] = wandb_summary['_runtime']
            causal_nf_io.print_info(f"Added runtime from wandb: {wandb_summary['_runtime']}s")

    with open(os.path.join(fans_output_dir, "fans_results.json"), 'w') as f:
        json.dump(fans_results_json, f, indent=2)

    causal_nf_io.print_info("FANS Analysis Results:")
    causal_nf_io.print_info(f"  Analyzed nodes: {list(fans_results['independence_results'].keys())}")
    causal_nf_io.print_info(f"FANS results saved to: {fans_output_dir}")

    # Save ENV1 model reference for ablation
    model_env1 = model
    dirpath_env1 = logger.save_dir

    # ============================================================
    # ENV2 model (swap data paths, same structure as above)
    # ============================================================
    causal_nf_io.print_info("\n=== Building ENV2 model (swapped data) ===")

    orig_data_path = cfg.dataset.external_data_path
    orig_env2_data_path = cfg.dataset.env2_external_data_path
    orig_test_path = cfg.dataset.test_data_path
    orig_env2_test_path = cfg.dataset.env2_test_data_path
    orig_root_dir = cfg.root_dir

    cfg.dataset.external_data_path = orig_env2_data_path
    cfg.dataset.env2_external_data_path = orig_data_path
    cfg.dataset.test_data_path = orig_env2_test_path
    cfg.dataset.env2_test_data_path = orig_test_path
    env2_root_dir = orig_root_dir + '_swap'
    cfg.root_dir = env2_root_dir

    preparator_env2 = SCMPreparator.loader(cfg.dataset)
    preparator_env2.prepare_data()

    loaders_env2 = preparator_env2.get_dataloaders(
        batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers
    )
    for i, loader in enumerate(loaders_env2):
        causal_nf_io.print_info(f"[ENV2] [{i}] num_batches: {len(loader)}")

    model_env2 = causal_nf_train.load_model(cfg=cfg, preparator=preparator_env2)

    import uuid
    run_uuid_env2 = str(uuid.uuid1()).replace("-", "")
    load_model_env2 = False

    base_dir_env2 = os.path.join(env2_root_dir, folder_name)
    if os.path.exists(base_dir_env2):
        existing_runs = [d for d in os.listdir(base_dir_env2)
                        if os.path.isdir(os.path.join(base_dir_env2, d))]
        for run_id in existing_runs:
            run_path = os.path.join(base_dir_env2, run_id)
            ckpt_files = glob.glob(os.path.join(run_path, "*.ckpt"))
            if ckpt_files:
                causal_nf_io.print_info(f"[ENV2] Found existing checkpoint in: {run_path}")
                load_model_env2 = True
                load_model_path_env2 = run_path
                run_uuid_env2 = os.path.basename(run_path)
                break

    dirpath_env2 = os.path.join(env2_root_dir, folder_name, run_uuid_env2)

    trainer_env2, logger_env2 = causal_nf_train.load_trainer(
        cfg=cfg,
        dirpath=dirpath_env2,
        logger_dir=dirpath_env2,
        include_logger=True,
        model_checkpoint=cfg.train.model_checkpoint,
        cfg_early=cfg.early_stopping,
        preparator=preparator_env2,
    )

    if not load_model_env2:
        causal_nf_io.print_info("[ENV2] Training model...")
        trainer_env2.fit(model_env2, train_dataloaders=loaders_env2[0], val_dataloaders=loaders_env2[1])

    if isinstance(preparator_env2.single_split, str):
        loaders_env2 = [loaders_env2[0]]

    model_env2.save_dir = dirpath_env2

    if load_model_env2:
        ckpt_file_env2 = os.path.join(load_model_path_env2, "last.ckpt")
        model_env2 = causal_nf_train.load_model(
            cfg=cfg, preparator=preparator_env2, ckpt_file=ckpt_file_env2
        )
        model_env2.eval()
        model_env2.save_dir = dirpath_env2
        ckpt_name = preparator_env2.get_ckpt_name(ckpt_file_env2)
        for i, loader_i in enumerate(loaders_env2):
            s_name = preparator_env2.split_names[i]
            causal_nf_io.print_info(f"[ENV2] Testing {s_name} split")
            preparator_env2.set_current_split(i)
            model_env2.ckpt_name = ckpt_name
            _ = trainer_env2.test(model=model_env2, dataloaders=loader_i)
    else:
        ckpt_name_list = ["last"]
        if cfg.early_stopping.activate:
            ckpt_name_list.append("best")
        for ckpt_name in ckpt_name_list:
            for i, loader_i in enumerate(loaders_env2):
                s_name = preparator_env2.split_names[i]
                causal_nf_io.print_info(f"[ENV2] Testing {s_name} split")
                preparator_env2.set_current_split(i)
                model_env2.ckpt_name = ckpt_name
                _ = trainer_env2.test(ckpt_path=ckpt_name, dataloaders=loader_i)

    model_env2 = model_env2.to('cuda' if torch.cuda.is_available() else 'cpu')
    model_env2.eval()

    # Restore original cfg
    cfg.dataset.external_data_path = orig_data_path
    cfg.dataset.env2_external_data_path = orig_env2_data_path
    cfg.dataset.test_data_path = orig_test_path
    cfg.dataset.env2_test_data_path = orig_env2_test_path
    cfg.root_dir = orig_root_dir

    # ============================================================
    # Ablation Verification
    # ============================================================
    causal_nf_io.print_info("\n=== Ablation: Cross-environment noise transfer ===")

    simultaneous_nodes = []
    if shifted_nodes and metadata_json:
        for node in shifted_nodes:
            shift_types = metadata_json['shift_types'].get(str(node), [])
            if len(shift_types) > 1:
                simultaneous_nodes.append(node)

    if not simultaneous_nodes:
        causal_nf_io.print_info("No simultaneous shift nodes found. Running on all shifted nodes.")
        simultaneous_nodes = shifted_nodes if shifted_nodes else []

    ablation_output_dir = os.path.join(dirpath_env1, "ablation_analysis")
    os.makedirs(ablation_output_dir, exist_ok=True)

    ablation_analyzer = FANSAblationAnalyzer(
        model_env1=model_env1.model,
        model_env2=model_env2.model,
        preparator=preparator,
        device=model_env1.device,
        scaler_env1=model_env1.input_scaler,
        scaler_env2=model_env2.input_scaler
    )

    # Reload test data with original paths
    test_data_env1 = torch.tensor(np.load(orig_test_path), dtype=torch.float32)
    test_data_env2 = torch.tensor(np.load(orig_env2_test_path), dtype=torch.float32)

    ablation_results = {}
    for node_idx in simultaneous_nodes:
        result = ablation_analyzer.verify(
            test_data_env1, test_data_env2,
            target_node=node_idx,
            save_dir=ablation_output_dir
        )
        ablation_results[node_idx] = result

    ablation_results_json = convert_for_json(ablation_results)
    with open(os.path.join(ablation_output_dir, "ablation_results.json"), 'w') as f:
        json.dump(ablation_results_json, f, indent=2)

    causal_nf_io.print_info(f"Ablation results saved to: {ablation_output_dir}")
    causal_nf_io.print_info(f"Total time: {datetime.now() - start_time}")
    print(f"Experiment folder: {dirpath_env1}")
