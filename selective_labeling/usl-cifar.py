# %%
import os
os.environ["USL_MODE"] = "USL"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import random
import numpy as np
import torch
import models.resnet_cifar_cld as resnet_cifar_cld
import utils
from utils import cfg, logger, print_b

utils.init(default_config_file="configs/cifar10_usl.yaml")

logger.info(cfg)

# %%
print_b("Loading model")

checkpoint = torch.load(cfg.MODEL.PRETRAIN_PATH)

model = resnet_cifar_cld.__dict__[cfg.MODEL.ARCH](
    low_dim=128, pool_len=4, normlinear=True).cuda()
model.load_state_dict(utils.single_model(checkpoint["model"]))
model.eval()

cifar100 = cfg.DATASET.NAME == "cifar100"
num_classes = 100 if cifar100 else 10

# if cfg.MODEL.USE_CLD:
#     state_dict = utils.single_model(checkpoint['train_model'])
# else:
#     state_dict = utils.single_model(checkpoint)
#     state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

# for k in list(state_dict.keys()):
#     if k.startswith('linear') or k.startswith('fc') or k.startswith('groupDis'):
#         del state_dict[k]

# if cfg.MODEL.USE_CLD:
#     model = resnet_cifar_cld.__dict__[cfg.MODEL.ARCH](
#         low_dim=128, pool_len=4, normlinear=False).cuda()
# else:
#     model = resnet_cifar.__dict__[cfg.MODEL.ARCH](num_classes=num_classes).cuda()

# mismatch = model.load_state_dict(state_dict, strict=False)

# logger.warning(
#     f"Key mismatches: {mismatch} (extra contrastive keys are intended)")

# # %%
print_b("Loading dataset")
assert cfg.DATASET.NAME in [
    "cifar10", "cifar100"], f"{cfg.DATASET.NAME} is not cifar10 or cifar100"
cifar100 = cfg.DATASET.NAME == "cifar100"
num_classes = 100 if cifar100 else 10

train_memory_dataset, train_memory_loader = utils.train_memory_cifar(
    root_dir=cfg.DATASET.ROOT_DIR,
    batch_size=cfg.DATALOADER.BATCH_SIZE,
    workers=cfg.DATALOADER.WORKERS, transform_name=cfg.DATASET.TRANSFORM_NAME, cifar100=cifar100)

targets = torch.tensor(train_memory_dataset.targets)
targets.shape


# %% imbalance
# print_b("Loading dataset")
# assert cfg.DATASET.NAME in [
#     "cifar10", "cifar100"], f"{cfg.DATASET.NAME} is not cifar10 or cifar100"
# cifar100 = cfg.DATASET.NAME == "cifar100"
# num_classes = 100 if cifar100 else 10

# train_memory_dataset, train_memory_loader = utils.train_memory_cifar(
#     root_dir=cfg.DATASET.ROOT_DIR,
#     batch_size=cfg.DATALOADER.BATCH_SIZE,
#     workers=cfg.DATALOADER.WORKERS, transform_name=cfg.DATASET.TRANSFORM_NAME, cifar100=cifar100)

# ## train_memory_dataset, imbalanced

# sample_db = utils.make_imb_data(cfg.MAX_NUM, cfg.CLASS_NUM, cfg.GAMMA)
# imb_idxs = utils.createImbIdxs(train_memory_dataset.targets, sample_db)
# random.shuffle(imb_idxs)

# train_memory_dataset = utils.CIFAR10_LT(root=cfg.DATASET.ROOT_DIR, indexs=imb_idxs)

# train_memory_loader = torch.utils.data.DataLoader(
#     train_memory_dataset, batch_size=cfg.DATALOADER.BATCH_SIZE, shuffle=False,
#     num_workers=cfg.DATALOADER.WORKERS, pin_memory=True, drop_last=False)


# targets = torch.tensor(train_memory_dataset.targets)
# targets.shape

# %%
print_b("Loading feat list")
feats_list = utils.get_feats_list(
    model, train_memory_loader, recompute=cfg.RECOMPUTE_ALL, force_no_extra_kwargs=True)

# %%
print_b("Calculating first order kNN density estimation")
d_knns, ind_knns = utils.partitioned_kNN(
    feats_list, K=cfg.USL.KNN_K, recompute=cfg.RECOMPUTE_ALL)
neighbors_dist = d_knns.mean(dim=1)
score_first_order = 1/neighbors_dist

# %%
num_centroids, final_sample_num = utils.get_sample_info_cifar(
    chosen_sample_num=cfg.CLASS_NUM, selected_num= cfg.USL.NUM_SELECTED_SAMPLES)
logger.info("num_centroids: {}, final_sample_num: {}".format(
    num_centroids, final_sample_num))

# %%
recompute_num_dependent = cfg.RECOMPUTE_ALL or cfg.RECOMPUTE_NUM_DEP
for kMeans_seed in cfg.USL.SEEDS:
    print_b(f"Running k-Means with seed {kMeans_seed}")
    if final_sample_num <= 40:
        # This is for better reproducibility, but has low memory usage efficiency.
        force_no_lazy_tensor = True
    else:
        force_no_lazy_tensor = False

    # This has side-effect: it calls torch.manual_seed to ensure the seed in k-Means is set.
    # Note: NaN in centroids happens when there is no corresponding sample which belongs to the centroid
    cluster_labels, centroids = utils.run_kMeans(feats_list, num_centroids, final_sample_num, Niter=cfg.USL.K_MEANS_NITERS,
                                                 recompute=recompute_num_dependent, seed=kMeans_seed, force_no_lazy_tensor=force_no_lazy_tensor)


    # cluster_labels = np.load("Pseudo_labels.npy")
    # cluster_labels = cluster_labels.astype(np.int64)
    # cluster_labels = torch.tensor(cluster_labels)


                                        
    print_b("Getting selections with regularization")
    selected_inds, selected_scores = utils.get_selection(utils.get_selection_with_reg, feats_list, neighbors_dist, cluster_labels, num_centroids, final_sample_num=final_sample_num, iters=cfg.USL.REG.NITERS, w=cfg.USL.REG.W,
                                        momentum=cfg.USL.REG.MOMENTUM, horizon_dist=cfg.USL.REG.HORIZON_DIST, alpha=cfg.USL.REG.ALPHA, verbose=True, seed=kMeans_seed, recompute=recompute_num_dependent, save=True)

    counts = np.bincount(np.array(train_memory_dataset.targets)[selected_inds])

    print("Class counts:", sum(counts > 0))
    print(counts.tolist())

    print("max: {}, min: {}".format(counts.max(), counts.min()))

    print("Number of selected indices:", len(selected_inds))
    print("Selected IDs:")
    print(repr(selected_inds))
    # print("culster labels:")
    # print(cluster_labels)
