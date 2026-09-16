import argparse
import os
from pathlib import Path

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
from torch.optim import lr_scheduler


def build_parser():
    parser = argparse.ArgumentParser(description="PyTorch ASL multi-label training")
    parser.add_argument("data", nargs="?", metavar="DIR", help="path to the COCO dataset")
    parser.add_argument("--dataset", choices=("coco", "tobacco"), default="coco")
    parser.add_argument("--manifest", type=str, help="path to the Tobacco CSV manifest")
    parser.add_argument("--data-root", type=str, help="root directory for Tobacco images")
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--model-name", default="tresnet_m")
    parser.add_argument("--model-path", default="./tresnet_m.pth", type=str)
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("-j", "--workers", default=8, type=int, metavar="N")
    parser.add_argument("--image-size", type=int, metavar="N")
    parser.add_argument("--thre", default=0.8, type=float, metavar="N")
    parser.add_argument("-b", "--batch-size", default=128, type=int, metavar="N")
    parser.add_argument("--print-freq", "-p", default=64, type=int, metavar="N")
    parser.add_argument("--output-dir", default="models", type=str)
    return parser


parser = build_parser()


def parse_args(argv=None):
    args = parser.parse_args(argv)
    if args.dataset == "tobacco":
        if args.manifest is None:
            parser.error("--manifest is required when --dataset=tobacco")
        if args.data_root is None:
            parser.error("--data-root is required when --dataset=tobacco")
        if args.num_classes is None:
            args.num_classes = 18
        elif args.num_classes != 18:
            parser.error("Tobacco-18 requires --num-classes=18")
        if args.image_size is None:
            args.image_size = 384
    else:
        if args.data is None:
            parser.error("the COCO data directory is required")
        if args.num_classes is None:
            args.num_classes = 80
        if args.image_size is None:
            args.image_size = 224

    args.do_bottleneck_head = False
    return args


def build_transforms(args):
    resize = transforms.Resize((args.image_size, args.image_size))
    if args.dataset == "tobacco":
        train_transform = transforms.Compose([
            resize,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        val_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ])
        return train_transform, val_transform

    from randaugment import RandAugment
    from src.helper_functions.helper_functions import CutoutPIL

    train_transform = transforms.Compose([
        resize,
        CutoutPIL(cutout_factor=0.5),
        RandAugment(),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    return train_transform, val_transform


def build_datasets(args, train_transform, val_transform):
    if args.dataset == "tobacco":
        from src.datasets import TobaccoManifestDataset

        train_dataset = TobaccoManifestDataset(
            args.manifest,
            args.data_root,
            split="train",
            transform=train_transform,
        )
        val_dataset = TobaccoManifestDataset(
            args.manifest,
            args.data_root,
            split="val",
            transform=val_transform,
        )
        return train_dataset, val_dataset

    from src.helper_functions.helper_functions import CocoDetection

    instances_path_val = os.path.join(args.data, "annotations/instances_val2014.json")
    instances_path_train = os.path.join(args.data, "annotations/instances_train2014.json")
    data_path_val = os.path.join(args.data, "val2014")
    data_path_train = os.path.join(args.data, "train2014")
    val_dataset = CocoDetection(data_path_val, instances_path_val, val_transform)
    train_dataset = CocoDetection(data_path_train, instances_path_train, train_transform)
    return train_dataset, val_dataset


def build_loaders(args, train_dataset, val_dataset):
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )
    return train_loader, val_loader


def prepare_target(target, dataset):
    if dataset == "coco":
        return target.max(dim=1)[0]
    return target


def calculate_mAP_from_logits(targets, logits):
    if targets.ndim != 2 or logits.ndim != 2 or targets.shape != logits.shape:
        raise ValueError(
            f"mAP expects matching [N,C] tensors, got targets={tuple(targets.shape)} "
            f"and logits={tuple(logits.shape)}"
        )
    from src.helper_functions.helper_functions import mAP

    probabilities = torch.sigmoid(logits)
    return mAP(targets.numpy(), probabilities.numpy())


def load_pretrained_backbone(model, model_path):
    state = torch.load(model_path, map_location="cpu")
    if not isinstance(state, dict) or "model" not in state:
        raise ValueError("pretrained checkpoint must contain a 'model' state dict")

    checkpoint_state = state["model"]
    model_state = model.state_dict()
    head_keys = {key for key in model_state if key.startswith("head.fc")}
    unexpected = sorted(key for key in checkpoint_state if key not in model_state)
    shape_mismatches = sorted(
        key
        for key, value in checkpoint_state.items()
        if key in model_state and key not in head_keys and value.shape != model_state[key].shape
    )
    loadable = {
        key: value
        for key, value in checkpoint_state.items()
        if key in model_state and key not in head_keys and value.shape == model_state[key].shape
    }
    missing_backbone = sorted(
        key for key in model_state if key not in head_keys and key not in loadable
    )
    if unexpected or shape_mismatches or missing_backbone:
        raise RuntimeError(
            "pretrained backbone is incompatible: "
            f"unexpected={unexpected}, shape_mismatches={shape_mismatches}, "
            f"missing_backbone={missing_backbone}"
        )

    result = model.load_state_dict(loadable, strict=False)
    unexpected_after_load = list(result.unexpected_keys)
    missing_after_load = set(result.missing_keys)
    if unexpected_after_load or missing_after_load != head_keys:
        raise RuntimeError(
            "unexpected load_state_dict result: "
            f"missing={sorted(missing_after_load)}, unexpected={unexpected_after_load}"
        )
    print(
        f"loaded {len(loadable)} pretrained backbone tensors from {model_path}; "
        f"reinitialized classifier tensors: {sorted(head_keys)}"
    )


def select_validation_model(model, ema_model, scores):
    selected_name = "ema" if scores["ema"] > scores["regular"] else "regular"
    selected_model = ema_model.module if selected_name == "ema" else model
    return selected_model, selected_name, scores[selected_name]


def main():
    args = parse_args()

    from src.models import create_model

    print("creating model...")
    model = create_model(args).cuda()
    if args.model_path:
        load_pretrained_backbone(model, args.model_path)
    print("done\n")

    train_transform, val_transform = build_transforms(args)
    train_dataset, val_dataset = build_datasets(args, train_transform, val_transform)
    print("len(val_dataset):", len(val_dataset))
    print("len(train_dataset):", len(train_dataset))
    train_loader, val_loader = build_loaders(args, train_dataset, val_dataset)
    train_multi_label(model, train_loader, val_loader, args)


def train_multi_label(model, train_loader, val_loader, args):
    from src.helper_functions.helper_functions import ModelEma, add_weight_decay
    from src.loss_functions.losses import AsymmetricLoss

    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82

    epochs = 80
    stop_epoch = 40
    weight_decay = 1e-4
    criterion = AsymmetricLoss(
        gamma_neg=4,
        gamma_pos=0,
        clip=0.05,
        disable_torch_grad_focal_loss=True,
    )
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=args.lr, weight_decay=0)
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.2,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    highest_mAP = 0
    train_info = []
    scaler = GradScaler()
    for epoch in range(epochs):
        if epoch > stop_epoch:
            break
        for i, (input_data, target) in enumerate(train_loader):
            input_data = input_data.cuda()
            target = prepare_target(target, args.dataset).cuda()
            with autocast():
                output = model(input_data).float()
            loss = criterion(output, target)
            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)

            if i % 100 == 0:
                train_info.append([epoch, i, loss.item()])
                print(
                    "Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}".format(
                        epoch,
                        epochs,
                        str(i).zfill(3),
                        str(steps_per_epoch).zfill(3),
                        scheduler.get_last_lr()[0],
                        loss.item(),
                    )
                )

        torch.save(
            model.state_dict(),
            output_dir / f"model-{epoch + 1}-{i + 1}.ckpt",
        )

        scores = validate_multi(
            val_loader,
            model,
            ema,
            dataset=args.dataset,
            num_classes=args.num_classes,
        )
        model.train()
        selected_model, selected_name, selected_score = select_validation_model(
            model,
            ema,
            scores,
        )
        if selected_score > highest_mAP:
            highest_mAP = selected_score
            torch.save(selected_model.state_dict(), output_dir / "model-highest.ckpt")
        print(
            "current_mAP = {:.2f} ({}), highest_mAP = {:.2f}\n".format(
                selected_score,
                selected_name,
                highest_mAP,
            )
        )


def validate_multi(val_loader, model, ema_model, dataset="coco", num_classes=None):
    print("starting validation")
    model.eval()
    logits_regular = []
    logits_ema = []
    targets = []
    with torch.inference_mode():
        for input_data, target in val_loader:
            target = prepare_target(target, dataset)
            with autocast():
                output_regular = model(input_data.cuda()).float().cpu()
                output_ema = ema_model.module(input_data.cuda()).float().cpu()
            logits_regular.append(output_regular)
            logits_ema.append(output_ema)
            targets.append(target.float().cpu())

    all_targets = torch.cat(targets)
    all_logits_regular = torch.cat(logits_regular)
    all_logits_ema = torch.cat(logits_ema)
    if num_classes is not None and all_targets.shape[1] != num_classes:
        raise ValueError(
            f"validation expected {num_classes} classes, got targets {tuple(all_targets.shape)}"
        )

    score_regular = calculate_mAP_from_logits(all_targets, all_logits_regular)
    score_ema = calculate_mAP_from_logits(all_targets, all_logits_ema)
    print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(score_regular, score_ema))
    return {"regular": score_regular, "ema": score_ema}


if __name__ == "__main__":
    main()
