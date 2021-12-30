import os
import sys
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.multiprocessing import set_start_method
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# 3DETR codebase specific imports
from models import build_model
from datasets import TelecomTowerDatasetConfig, TelecomTowerPredictionDataset
from utils.dist import init_distributed, is_distributed, is_primary, get_rank, barrier, all_gather_dict
from utils.misc import my_worker_init_fn
from utils.ap_calculator import parse_predictions, get_ap_config_dict, flip_axis_to_depth

# Pointivo codebase specific imports
from image_recognition.app.dvo.bbox_3d.oriented_bbox_3d import OrientedBbox3D
from image_recognition.app.dvo.ground_truths.object_detection_3d import ObjectDetectionLabeledData3D
from pv_3d_scene_components.localized_objects.non_planar_objects.cuboid_3d import Cuboid3D
from pv_3d_scene_components.localized_objects.point_3d import Point3D


def make_args_parser():
    parser = argparse.ArgumentParser("3D Detection Using Transformers", add_help=False)

    ##### Model #####
    parser.add_argument("--model_name", default="3detr", type=str, help="Name of the model", choices=["3detr"])
    ### Encoder
    parser.add_argument("--enc_type", default="vanilla", choices=["vanilla", "masked"])
    # Below options are only valid for vanilla encoder
    parser.add_argument("--enc_nlayers", default=3, type=int)
    parser.add_argument("--enc_dim", default=256, type=int)
    parser.add_argument("--enc_ffn_dim", default=128, type=int)
    parser.add_argument("--enc_dropout", default=0.1, type=float)
    parser.add_argument("--enc_nhead", default=4, type=int)
    parser.add_argument("--enc_pos_embed", default=None, type=str)
    parser.add_argument("--enc_activation", default="relu", type=str)

    ### Decoder
    parser.add_argument("--dec_nlayers", default=8, type=int)
    parser.add_argument("--dec_dim", default=256, type=int)
    parser.add_argument("--dec_ffn_dim", default=256, type=int)
    parser.add_argument("--dec_dropout", default=0.1, type=float)
    parser.add_argument("--dec_nhead", default=4, type=int)

    ### MLP heads for predicting bounding boxes
    parser.add_argument("--mlp_dropout", default=0.3, type=float)

    ### Other model params
    parser.add_argument("--preenc_npoints", default=2048, type=int)
    parser.add_argument("--pos_embed", default="fourier", type=str, choices=["fourier", "sine"])
    parser.add_argument("--nqueries", default=256, type=int)
    parser.add_argument("--use_color", default=False, action="store_true")
    parser.add_argument("--use_height", default=False, action="store_true")

    ##### Dataset #####
    parser.add_argument("--dataset_name", required=True, type=str, choices=["telecomtower"])
    parser.add_argument("--dataset_root_dir", type=str, default=None,
                        help="Root directory containing the dataset files."
                             "If None, default values from scannet.py/sunrgbd.py/telecomtower.py are used",
                        )
    parser.add_argument("--point_cloud_points", type=int, default=100000,
                        help="Number of points to randomly subsample from the point cloud", )
    parser.add_argument("--batchsize_per_gpu", default=8, type=int)
    parser.add_argument("--dataset_num_workers", default=4, type=int)

    ##### Testing #####
    parser.add_argument("--test_ckpt", default=None, type=str)
    parser.add_argument("--refine_predictions", default=False, action="store_true",
                        help="Refine output from the network (steps include NMS)")
    parser.add_argument("--nms_iou_thresh", default=0.1, type=float,
                        help="IoU threshold for NMS. Has no effect without --refine_predictions")
    parser.add_argument("--predictions_save_dir", default=None, type=str)
    parser.add_argument("--store_halved_bbox_size", default=False, action="store_true",
                        help="Halves the bbox size before saving. Added for consistency with votenet dataset format")

    parser.add_argument("--ngpus", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    return parser


@torch.no_grad()
def model_predict(args, models: Dict, dataloaders: Dict, post_process_config_dict: Dict = None) -> Dict:
    model, model_no_ddp = models["model"], models["model_no_ddp"]
    if args.test_ckpt is None or not os.path.isfile(args.test_ckpt):
        f"Please specify a test checkpoint using --test_ckpt. Found invalid value {args.test_ckpt}"
        sys.exit(1)

    ckpt = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
    model_no_ddp.load_state_dict(ckpt["model"])
    net_device = next(model.parameters()).device

    model.eval()
    barrier()

    predictions = {}
    for idx, batch_data in enumerate(dataloaders["loader"]):
        inputs = {
            "point_clouds": batch_data["point_clouds"],
            "point_cloud_dims_min": batch_data["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data["point_cloud_dims_max"],
        }
        for key in inputs:
            inputs[key] = inputs[key].to(net_device)
        outputs = model(inputs)
        # Memory intensive as it gathers point cloud GT tensor across all ranks
        outputs["outputs"] = all_gather_dict(outputs["outputs"])
        if is_primary():
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"Evaluating batch {idx} | Mem {mem_mb:0.2f}MB")
        barrier()

        # perform refinement through NMS and thresholding
        if post_process_config_dict:
            # refinement, same as in AP calculation
            parsed_pred_batch = parse_predictions(
                predicted_boxes=outputs["outputs"]["box_corners"], sem_cls_probs=outputs["outputs"]["sem_cls_prob"],
                objectness_probs=outputs["outputs"]["objectness_prob"], point_cloud=inputs["point_clouds"],
                config_dict=post_process_config_dict)

            def _convert_bbox_corners_to_bbox_params(bbox_corners):
                """
                The parse_predictions method takes input in the form of bbox params, but outputs bbox corners.
                This method converts the bbox corners back to bbox params.
                If the desired output format is bbox corners, this method can be skipped.
                """
                assert bbox_corners.shape == (8, 3)
                bbox_corners = flip_axis_to_depth(bbox_corners)
                vertex = Point3D(x=bbox_corners[0][0], y=bbox_corners[0][1], z=bbox_corners[0][2])
                pt1 = Point3D(x=bbox_corners[1][0], y=bbox_corners[1][1], z=bbox_corners[1][2])
                pt2 = Point3D(x=bbox_corners[3][0], y=bbox_corners[3][1], z=bbox_corners[3][2])
                pt3 = Point3D(x=bbox_corners[4][0], y=bbox_corners[4][1], z=bbox_corners[4][2])
                bbox3d = OrientedBbox3D.from_cuboid3d(
                    cuboid3d=Cuboid3D.from_points(vertex=vertex, points_connected_to_vertex=[pt1, pt2, pt3]),
                    class_label=class_label)
                return np.asarray([bbox3d.x, bbox3d.y, bbox3d.z,
                                   bbox3d.x_size, bbox3d.y_size, bbox3d.z_size,
                                   np.deg2rad(bbox3d.yaw_degree)])

            pred_out = {
                "box_corners": [], "center_unnormalized": [], "size_unnormalized": [], "angle_continuous": [],
                "sem_cls_prob": [], "class_label": []
            }
            for idx, parsed_preds in enumerate(parsed_pred_batch):
                pred_out_idx = {
                    "box_corners": [], "center_unnormalized": [], "size_unnormalized": [], "angle_continuous": [],
                    "sem_cls_prob": [], "class_label": []
                }

                for pred_idx, pred in enumerate(parsed_preds):
                    class_label = pred[0]

                    bbox_corners = pred[1]
                    bbox_params = _convert_bbox_corners_to_bbox_params(bbox_corners=bbox_corners)

                    parsed_pred_conf = pred[2]

                    pred_out_idx["box_corners"].append(bbox_corners)
                    pred_out_idx["center_unnormalized"].append(bbox_params[:3])
                    pred_out_idx["size_unnormalized"].append(bbox_params[3:6])
                    pred_out_idx["angle_continuous"].append(bbox_params[6])
                    pred_out_idx["sem_cls_prob"].append([parsed_pred_conf])
                    pred_out_idx["class_label"].append([class_label.astype(np.uint8)])

                for key in ["box_corners", "center_unnormalized", "size_unnormalized", "angle_continuous",
                            "sem_cls_prob", "class_label"]:
                    pred_out[key].append(pred_out_idx[key])

            for key in ["box_corners", "center_unnormalized", "size_unnormalized", "angle_continuous", "sem_cls_prob",
                        "class_label"]:
                pred_out[key] = np.asarray(pred_out[key])

        # no refinement
        else:
            pred_out = outputs["outputs"]
            for key in ["box_corners", "center_unnormalized", "size_unnormalized", "angle_continuous", "sem_cls_prob"]:
                pred_out[key] = pred_out[key].cpu().numpy()
            pred_out["class_label"] = np.expand_dims(
                np.argmax(pred_out["sem_cls_prob"], axis=-1), axis=-1).astype(np.uint8)

        # return a dict of {filename: {bbox_params, bbox_corners}}
        for i, filename in enumerate(batch_data["point_cloud_filename"]):
            box_corners = pred_out["box_corners"][i, :]
            center = pred_out["center_unnormalized"][i, :]
            size = pred_out["size_unnormalized"][i, :]
            angle = pred_out["angle_continuous"][i, :]
            cls_label = pred_out["class_label"][i, :]
            cls_prob = pred_out["sem_cls_prob"][i, :]
            predictions[filename] = {
                'bbox_params': np.concatenate(
                    (center, size, np.expand_dims(angle, axis=-1), cls_label, cls_prob), axis=-1),
                'bbox_corners': box_corners,
            }

    return predictions


def main(local_rank, args):
    if args.ngpus > 1:
        print("Initializing Distributed Training. This is in BETA mode and hasn't been tested thoroughly. "
              "Use at your own risk :)")
        print("To get the maximum speed-up consider reducing evaluations on val set by setting --eval_every_epoch "
              "to greater than 50")
        init_distributed(
            local_rank,
            global_rank=local_rank,
            world_size=args.ngpus,
            dist_url=args.dist_url,
            dist_backend="nccl",
        )

    print(f"Called with args: {args}")
    torch.cuda.set_device(local_rank)
    np.random.seed(args.seed + get_rank())
    torch.manual_seed(args.seed + get_rank())
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + get_rank())

    # dataset
    dataset_config = TelecomTowerDatasetConfig()
    dataset = TelecomTowerPredictionDataset(
        path_to_data_dir=Path(args.dataset_root_dir), num_points=args.point_cloud_points, use_color=args.use_color,
        use_height=args.use_height)  # build_dataset(args)
    sampler = DistributedSampler(dataset, shuffle=False) if is_distributed() else \
        torch.utils.data.SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batchsize_per_gpu,
                            num_workers=args.dataset_num_workers, worker_init_fn=my_worker_init_fn)
    dataloaders = {"loader": dataloader, "sampler": sampler}

    # model
    model, bbox_processor = build_model(args, dataset_config)
    model = model.cuda(local_rank)
    model_no_ddp = model
    if is_distributed():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )
    models = {"model": model, "model_no_ddp": model_no_ddp, "processor": bbox_processor}

    # post-process config
    post_process_config_dict = None if not args.refine_predictions else \
        get_ap_config_dict(remove_empty_box=True, use_3d_nms=True, nms_iou=args.nms_iou_thresh, conf_thresh=0.05,
                           dataset_config=dataset_config)

    # prediction
    preds = model_predict(args=args, models=models, dataloaders=dataloaders,
                          post_process_config_dict=post_process_config_dict)

    for name, pred in preds.items():
        if args.store_halved_bbox_size:  # for consistency with votenet dataset creation
            pred["bbox_params"][:, 3:6] *= 0.5

        bounding_boxes_3d = []
        bbox_params = pred["bbox_params"]
        for idx in range(bbox_params.shape[0]):
            obb3d = OrientedBbox3D(x=bbox_params[idx][0], y=bbox_params[idx][1], z=bbox_params[idx][2],
                                   x_size=bbox_params[idx][3], y_size=bbox_params[idx][4], z_size=bbox_params[idx][5],
                                   roll_degree=0, pitch_degree=0, yaw_degree=np.rad2deg(bbox_params[idx][6]),
                                   class_label=dataset_config.class2type[int(bbox_params[idx][7])],
                                   confidence_score=bbox_params[idx][8])
            bounding_boxes_3d.append(obb3d)
        # assuming that point cloud files are saved as: project_id.npz
        odld3d = ObjectDetectionLabeledData3D(project_id=name, bounding_boxes_3d=bounding_boxes_3d)
        odld3d.to_json_file(Path(args.predictions_save_dir) / f"{name}.od3d_predicted.json")


def launch_distributed(args):
    world_size = args.ngpus
    if world_size == 1:
        main(local_rank=0, args=args)
    else:
        torch.multiprocessing.spawn(main, nprocs=world_size, args=(args,))


if __name__ == "__main__":
    """
    Run example:
    $ python predict.py --nqueries 512 --preenc_npoints 4096 --enc_type masked --enc_nlayers 3 --enc_dim 256
        --enc_ffn_dim 128 --enc_nhead 8 --dec_nlayers 8 --dec_dim 256 --dec_ffn_dim 256 --dec_nhead 8
        --dataset_name telecomtower --dataset_root_dir "path-to-dataset" --predictions_save_dir "path-to-preds"
        --dataset_num_workers 0 --batchsize_per_gpu 1 --ngpus 1 --test_ckpt "path-to-.pth-ckpt-file"
        --refine_predictions --nms_iou_thresh 0.1
    """
    parser = make_args_parser()
    args = parser.parse_args()
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    launch_distributed(args)
