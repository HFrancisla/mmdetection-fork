#!/usr/bin/env python3
"""Visualize the trained one-level DWT-FDSP front end and detections."""

from __future__ import annotations

import argparse
import copy
import json
import math
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import matplotlib
from matplotlib import font_manager

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from mmcv.transforms import Compose

from mmdet.apis import inference_detector, init_detector


CHINESE_FONT = Path(
    '/home/ipr4090/2024_hzf/deprecated/_Thesis/'
    'lowlight_detection_sota/IAFE-YOLO/model_data/simhei.ttf')
if CHINESE_FONT.is_file():
    font_manager.fontManager.addfont(str(CHINESE_FONT))
    matplotlib.rcParams['font.family'] = font_manager.FontProperties(
        fname=str(CHINESE_FONT)).get_name()
    matplotlib.rcParams['axes.unicode_minus'] = False

GROUP_LABELS = {
    'extremely dark': '极暗场景',
    'relatively brighter / dark': '相对较亮的暗场景',
    'uneven illumination': '光照不均场景',
    'small target / visible edges': '小目标 / 明显边缘',
}
CLASS_LABELS_ZH = {
    'Bicycle': '自行车', 'Boat': '船', 'Bottle': '瓶子', 'Bus': '公交车',
    'Car': '汽车', 'Cat': '猫', 'Chair': '椅子', 'Cup': '杯子',
    'Dog': '狗', 'Face': '人脸', 'Motorbike': '摩托车', 'People': '行人',
    'Table': '桌子',
}

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_FLOOR = math.log(1e-6)
LOG_CEIL = math.log(2.0)


def read_ground_truth(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    objects = []
    for obj in root.findall('object'):
        box = obj.find('bndbox')
        if box is None:
            continue
        x1, y1, x2, y2 = [
            float(box.find(key).text) - 1
            for key in ('xmin', 'ymin', 'xmax', 'ymax')
        ]
        objects.append({
            'name': obj.findtext('name', default='unknown'),
            'bbox': [x1, y1, x2, y2],
            'difficult': int(obj.findtext('difficult', default='0')),
        })
    return objects


def image_metrics(rgb: np.ndarray, objects: list[dict]):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    tile_means = cv2.resize(gray, (4, 4), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(gray, 60, 140)
    h, w = gray.shape
    box_areas = []
    for obj in objects:
        x1, y1, x2, y2 = obj['bbox']
        box_areas.append(max(0.0, x2 - x1) * max(0.0, y2 - y1) / (h * w))
    return {
        'mean_luminance_0_255': float(gray.mean()),
        'median_luminance_0_255': float(np.median(gray)),
        'illumination_tile_std': float(tile_means.std()),
        'edge_pixel_fraction': float((edges > 0).mean()),
        'ground_truth_count': len(objects),
        'smallest_box_area_fraction': min(box_areas) if box_areas else None,
    }


def resolve_image_path(data_root: Path, image_name: str,
                       image_subdir: str) -> Path:
    relative_path = Path(image_name)
    image_base = data_root / image_subdir / relative_path
    candidates = [image_base]
    if not relative_path.suffix:
        candidates.extend(image_base.with_suffix(ext)
                          for ext in ('.jpg', '.jpeg', '.png', '.bmp'))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f'Image {image_name!r} not found under {data_root / image_subdir}')


def scan_test_split(data_root: Path, split_file: str,
                    image_subdir='JPEGImages',
                    annotation_subdir='Annotations'):
    records = []
    split = Path(split_file)
    if not split.is_absolute():
        split = data_root / split
    for line in split.read_text().splitlines():
        name = line.strip()
        if not name:
            continue
        image_path = resolve_image_path(data_root, name, image_subdir)
        image_stem = Path(name).stem
        xml_path = data_root / annotation_subdir / f'{image_stem}.xml'
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None or not xml_path.is_file():
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        objects = read_ground_truth(xml_path)
        metrics = image_metrics(rgb, objects)
        records.append({
            'name': name,
            'image_path': image_path,
            'xml_path': xml_path,
            'objects': objects,
            'metrics': metrics,
        })
    if not records:
        raise RuntimeError(f'No readable samples found in {split}')
    return records


def select_representatives(records):
    selected = []

    def pick(key, label, reverse=False):
        candidates = [r for r in records if r['name'] not in {
            item['name'] for item in selected
        }]
        chosen = sorted(candidates, key=key, reverse=reverse)[0]
        chosen['visualization_group'] = label
        selected.append(chosen)

    pick(lambda r: r['metrics']['mean_luminance_0_255'], 'extremely dark')

    brightness_values = np.array([
        r['metrics']['mean_luminance_0_255'] for r in records
    ])
    target_brightness = float(np.quantile(brightness_values, 0.55))
    pick(lambda r: abs(r['metrics']['mean_luminance_0_255'] -
                       target_brightness), 'relatively brighter / dark')

    pick(lambda r: r['metrics']['illumination_tile_std'],
         'uneven illumination', reverse=True)

    dark_limit = float(np.quantile(brightness_values, 0.75))
    dark_candidates = [
        r for r in records
        if r['name'] not in {item['name'] for item in selected}
        and r['metrics']['mean_luminance_0_255'] <= dark_limit
        and r['metrics']['smallest_box_area_fraction'] is not None
    ]
    if not dark_candidates:
        dark_candidates = [r for r in records if r['name'] not in {
            item['name'] for item in selected
        } and r['metrics']['smallest_box_area_fraction'] is not None]
    area_limit = float(np.quantile([
        r['metrics']['smallest_box_area_fraction'] for r in dark_candidates
    ], 0.25))
    small_and_dark = [r for r in dark_candidates
                       if r['metrics']['smallest_box_area_fraction'] <= area_limit]
    chosen = max(small_and_dark,
                 key=lambda r: r['metrics']['edge_pixel_fraction'])
    chosen['visualization_group'] = 'small target / visible edges'
    selected.append(chosen)
    return selected


def cpu_tensor(value):
    return value.detach().float().cpu().numpy()


def infer_and_capture(model, image_path: Path):
    dwtnet = model.front_net.dwtnet
    mean = np.asarray(dwtnet.input_mean, dtype=np.float32)
    std = np.asarray(dwtnet.input_std, dtype=np.float32)
    captured = {}

    def pre_hook(module, inputs):
        captured['normalized_input'] = cpu_tensor(inputs[0])

    def dwt_pre_hook(module, inputs):
        captured['dwt_input_rgb01_padded'] = cpu_tensor(inputs[0])

    def dwt_hook(module, inputs, output):
        captured['dwt_output'] = cpu_tensor(output)

    def idwt_hook(module, inputs, output):
        captured['idwt_output_padded'] = cpu_tensor(output)

    def tensor_hook(key):
        def hook(module, inputs, output):
            captured[key] = cpu_tensor(output)
        return hook

    def fuse_pre_hook(module, inputs):
        # This is the concatenated spatial/spectral tensor consumed by FuseNet.
        captured['frontend_cat'] = cpu_tensor(inputs[0])

    def recording_process(module, subband):
        result = original_process(subband)
        log_subband = torch.log(subband.clamp(min=1e-6))
        captured['ll_pre'] = cpu_tensor(subband)
        captured['log_ll'] = cpu_tensor(log_subband)
        captured['fdsp_raw'] = cpu_tensor(module._fdsp(log_subband))
        captured['ll_prime'] = cpu_tensor(result)
        return result

    original_process = dwtnet._process_ll
    dwtnet._process_ll = types.MethodType(recording_process, dwtnet)
    handles = [
        dwtnet.register_forward_pre_hook(pre_hook),
        dwtnet.dwt.register_forward_pre_hook(dwt_pre_hook),
        dwtnet.dwt.register_forward_hook(dwt_hook),
        dwtnet.idwt.register_forward_hook(idwt_hook),
        model.front_net.spatial_net.register_forward_hook(
            tensor_hook('frontend_spatial')),
        dwtnet.register_forward_hook(tensor_hook('frontend_dwtnet')),
        model.front_net.spectral_net.register_forward_hook(
            tensor_hook('frontend_spectral')),
        model.front_net.fuse_net.register_forward_pre_hook(fuse_pre_hook),
        model.front_net.fuse_net.register_forward_hook(
            tensor_hook('frontend_output')),
    ]

    pipeline_cfg = copy.deepcopy(model.cfg.test_dataloader.dataset.pipeline)
    pipeline_cfg = [
        transform for transform in pipeline_cfg
        if transform.get('type') != 'LoadAnnotations'
    ]
    pipeline = Compose(pipeline_cfg)
    try:
        result = inference_detector(
            model, str(image_path), test_pipeline=pipeline)
    finally:
        for handle in handles:
            handle.remove()
        del dwtnet.__dict__['_process_ll']

    meta = result.metainfo
    img_h, img_w = [int(v) for v in meta['img_shape'][:2]]
    norm = captured['normalized_input'][0, :, :img_h, :img_w]
    resized_rgb = np.clip(
        norm * std[:, None, None] + mean[:, None, None], 0, 255)
    resized_rgb = np.moveaxis(resized_rgb, 0, -1).astype(np.uint8)

    dwt_input = captured['dwt_input_rgb01_padded'][0, :, :img_h, :img_w]
    bands = np.split(captured['dwt_output'][0], 4, axis=0)
    band_h, band_w = math.ceil(img_h / 2), math.ceil(img_w / 2)
    bands = [b[:, :band_h, :band_w] for b in bands]
    idwt = captured['idwt_output_padded'][0, :, :img_h, :img_w]
    ll_pre = captured['ll_pre'][0, :, :band_h, :band_w]
    log_ll = captured['log_ll'][0, :, :band_h, :band_w]
    fdsp_raw = captured['fdsp_raw'][0, :, :band_h, :band_w]
    ll_prime = captured['ll_prime'][0, :, :band_h, :band_w]

    pred = result.pred_instances
    boxes = pred.bboxes.detach().cpu().numpy()
    scores = pred.scores.detach().cpu().numpy()
    labels = pred.labels.detach().cpu().numpy()
    return {
        'result': result,
        'meta': meta,
        'resized_rgb': resized_rgb,
        'dwt_input_rgb01': dwt_input,
        'll': bands[0],
        'lh': bands[1],
        'hl': bands[2],
        'hh': bands[3],
        'll_pre': ll_pre,
        'log_ll': log_ll,
        'fdsp_raw': fdsp_raw,
        'll_prime': ll_prime,
        'idwt_output': idwt,
        # Exact tensors through the front end. The full (possibly padded)
        # tensors are retained here; the flow PNG crops them to img_shape.
        'frontend_spatial': captured['frontend_spatial'][0],
        'frontend_dwtnet': captured['frontend_dwtnet'][0],
        'frontend_spectral': captured['frontend_spectral'][0],
        'frontend_cat': captured['frontend_cat'][0],
        'frontend_output': captured['frontend_output'][0],
        'frontend_padded_shape': tuple(captured['frontend_output'].shape),
        'pred_boxes': boxes,
        'pred_scores': scores,
        'pred_labels': labels,
    }


def signed_limit(arrays, percentile=99.0):
    values = np.concatenate([np.abs(x).reshape(-1) for x in arrays])
    return max(float(np.percentile(values, percentile)), 1e-6)


def log_display(log_chw):
    mapped = (log_chw - LOG_FLOOR) / (LOG_CEIL - LOG_FLOOR)
    return np.moveaxis(mapped.clip(0, 1), 0, -1)


def lowpass_rgb(ll_chw):
    return np.moveaxis((ll_chw / 2.0).clip(0, 1), 0, -1)


def feature_scalar(chw):
    return chw.mean(axis=0)


def gradient_magnitude(chw):
    scalar = feature_scalar(chw).astype(np.float32)
    gx = cv2.Sobel(scalar, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(scalar, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def add_signed(ax, image, limit, title, cmap='coolwarm'):
    ax.imshow(image, cmap=cmap, norm=TwoSlopeNorm(
        vmin=-limit, vcenter=0.0, vmax=limit), interpolation='nearest')
    ax.set_title(title)
    ax.axis('off')


def add_image(ax, image, title, cmap=None, vmin=None, vmax=None):
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax,
              interpolation='nearest')
    ax.set_title(title)
    ax.axis('off')


def make_key_summary(samples, output_dir, llprime_limit, idwt_limit):
    fig, axes = plt.subplots(len(samples), 4, figsize=(16, 4 * len(samples)),
                             squeeze=False)
    headers = ['原始输入（原分辨率）', 'LL/2（RGB低频代理图）',
               "FDSP(LL) = LL'（RGB有符号均值）",
               "IDWT(LL', LH, HL, HH)（RGB有符号均值）"]
    for row, sample in enumerate(samples):
        cap = sample['capture']
        axes[row, 0].imshow(sample['original_rgb'])
        axes[row, 0].text(
            0.5, -0.04,
            f"{GROUP_LABELS[sample['visualization_group']]} — {sample['name']}",
            transform=axes[row, 0].transAxes, ha='center', va='top',
            fontsize=9)
        axes[row, 1].imshow(lowpass_rgb(cap['ll']))
        add_signed(axes[row, 2], feature_scalar(cap['ll_prime']),
                   llprime_limit, '')
        add_signed(axes[row, 3], feature_scalar(cap['idwt_output']),
                   idwt_limit, '')
        for ax in axes[row]:
            ax.axis('off')
    for col, title in enumerate(headers):
        axes[0, col].set_title(title, fontsize=12, pad=10)
    fig.suptitle('一级 DWT-FDSP：四阶段特征总览',
                 fontsize=15, y=1.002)
    fig.tight_layout()
    fig.savefig(output_dir / 'key_four_column_summary.png', dpi=180,
                bbox_inches='tight')
    plt.close(fig)


def make_sample_summary(sample, output_dir, llprime_limit, idwt_limit):
    cap = sample['capture']
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        f"{sample['name']} — {GROUP_LABELS[sample['visualization_group']]}",
                 fontsize=14)
    axes[0].imshow(sample['original_rgb'])
    axes[0].set_title('原始输入（原分辨率）')
    axes[1].imshow(lowpass_rgb(cap['ll']))
    axes[1].set_title('LL/2（RGB低频代理图）')
    add_signed(axes[2], feature_scalar(cap['ll_prime']), llprime_limit,
               "FDSP(LL) = LL'（RGB有符号均值）")
    add_signed(axes[3], feature_scalar(cap['idwt_output']), idwt_limit,
               'IDWT输出（RGB有符号均值）')
    for ax in axes:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(output_dir / f"{Path(sample['name']).stem}_four_stage.png",
                dpi=180, bbox_inches='tight')
    plt.close(fig)


def activation_rms(chw):
    """Channel RMS activation map for a multi-channel feature tensor."""
    return np.sqrt(np.mean(np.square(chw.astype(np.float32)), axis=0))


def make_frontend_flow(sample, output_dir, scales):
    """Show the six tensors along the actual two-branch detection front end."""
    cap = sample['capture']
    h, w = cap['resized_rgb'].shape[:2]

    # The data preprocessor may pad H/W to a divisor. Crop only for display;
    # the complete tensors, including padding, remain in the NPZ.
    spatial = cap['frontend_spatial'][:, :h, :w]
    dwtnet_output = cap['frontend_dwtnet'][:, :h, :w]
    spectral = cap['frontend_spectral'][:, :h, :w]
    cat = cap['frontend_cat'][:, :h, :w]
    output = cap['frontend_output'][:, :h, :w]

    fig, axes = plt.subplots(2, 3, figsize=(19, 12), squeeze=False)
    fig.suptitle(
        f"{sample['name']} — 前端双分支到检测器输入 | "
        f"显示有效区域 {h}×{w}（已裁去 batch padding）",
        fontsize=16)

    add_image(axes[0, 0], cap['resized_rgb'],
              '01 输入图（RGB）')
    add_image(axes[0, 1], activation_rms(spatial),
              '02 空间支路输出（24 通道 RMS）',
              cmap='magma', vmin=0, vmax=scales['spatial'])
    add_signed(axes[0, 2], feature_scalar(dwtnet_output), scales['dwtnet'],
               '03 DWTNet 频域输出（IDWT 后通道均值）')
    add_image(axes[1, 0], activation_rms(spectral),
              '04 频域支路输出（24 通道 RMS）',
              cmap='magma', vmin=0, vmax=scales['spectral'])
    add_image(axes[1, 1], activation_rms(cat),
              '05 空间 / 频域 Cat（48 通道 RMS）',
              cmap='magma', vmin=0, vmax=scales['cat'])

    # The final tensor has three channels but is an internal feature tensor,
    # not natural RGB. Apply one shared signed scale to all three channels for
    # an honest false-colour view (no independent per-channel normalization).
    pseudo_rgb = np.moveaxis(
        np.clip((output + scales['output']) / (2 * scales['output']), 0, 1),
        0, -1)
    add_image(axes[1, 2], pseudo_rgb,
              '06 前端最终输出（Backbone 输入）')

    # State both the true padded tensor shape and the cropped display shape.
    batch, _, padded_h, padded_w = cap['frontend_padded_shape']
    shape_labels = [
        f'实际输入 {batch}×3×{padded_h}×{padded_w}\n'
        f'RGB显示 3×{h}×{w}',
        f'张量 {batch}×{spatial.shape[0]}×{padded_h}×{padded_w}\n'
        f'图示 {spatial.shape[0]}×{h}×{w}',
        f'张量 {batch}×{dwtnet_output.shape[0]}×{padded_h}×{padded_w}\n'
        f'图示 {dwtnet_output.shape[0]}×{h}×{w}',
        f'张量 {batch}×{spectral.shape[0]}×{padded_h}×{padded_w}\n'
        f'图示 {spectral.shape[0]}×{h}×{w}',
        f'张量 {batch}×{cat.shape[0]}×{padded_h}×{padded_w}\n'
        f'图示 {cat.shape[0]}×{h}×{w}',
        f'Backbone输入 {batch}×{output.shape[0]}×{padded_h}×{padded_w}\n'
        f'图示 {output.shape[0]}×{h}×{w}',
    ]
    for ax, shape_label in zip(axes.flat, shape_labels):
        ax.text(0.5, -0.04, shape_label, transform=ax.transAxes,
                ha='center', va='top', fontsize=9, linespacing=1.25)
        ax.set_title(ax.get_title(), fontsize=12, linespacing=1.3, pad=12)

    fig.text(
        0.5, 0.015,
        'RMS 图显示多通道激活强度；DWTNet 图显示通道均值。'
        '最后一格是 Backbone 的 3 通道输入伪 RGB（非自然 RGB）。'
        '色阶在四张样本间共享；形状同时注明含 padding 的张量与裁剪显示区。',
        ha='center', fontsize=10)
    fig.subplots_adjust(left=0.035, right=0.99, top=0.89, bottom=0.14,
                        wspace=0.08, hspace=0.5)
    fig.savefig(output_dir / f"{Path(sample['name']).stem}_frontend_flow.png",
                dpi=180, bbox_inches='tight')
    plt.close(fig)


def make_dwt_bands_panel(sample, output_dir, ll_limit, detail_limit):
    """Show the RGB -> DWT bands and the lowpass signal used by log(LL)."""
    cap = sample['capture']
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), squeeze=False)
    fig.suptitle(f"{sample['name']} — RGB 输入与一级 DWT 子带",
                 fontsize=16)

    add_image(axes[0, 0], cap['resized_rgb'],
              '01 输入图（RGB）')
    add_signed(axes[0, 1], feature_scalar(cap['ll']), ll_limit,
               '02 LL 原始系数（RGB 均值）')
    add_image(axes[0, 2], lowpass_rgb(cap['ll']),
              '03 LL/2 RGB 低频图')
    for ax, band, title in zip(
            axes[1], ('lh', 'hl', 'hh'),
            ('04 LH 细节带（RGB 均值）',
             '05 HL 细节带（RGB 均值）',
             '06 HH 细节带（RGB 均值）')):
        add_signed(ax, feature_scalar(cap[band]), detail_limit, title)

    for ax in axes.flat:
        ax.title.set_fontsize(12)
        ax.title.set_linespacing(1.25)
        ax.set_title(ax.get_title(), fontsize=12, linespacing=1.25, pad=10)
    fig.text(
        0.5, 0.02,
        '先将归一化输入还原到 [0,1] RGB，再进行 DWT。02 是原始 LL 系数的有符号通道均值；'
        '03 是便于按亮度观察的 LL/2 RGB 低频图。log(LL) 只进入 FDSP 支路。'
        'LH / HL / HH 与 LL 系数均显示有符号 RGB 通道均值。',
        ha='center', fontsize=10)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.89, bottom=0.13,
                        wspace=0.08, hspace=0.32)
    fig.savefig(output_dir / f"{Path(sample['name']).stem}_dwt_bands.png",
                dpi=180, bbox_inches='tight')
    plt.close(fig)


def make_feature_panels(sample, output_dir, scales):
    cap = sample['capture']
    fig, axes = plt.subplots(5, 4, figsize=(17, 19), squeeze=False)
    fig.suptitle(
        f"{sample['name']} — {GROUP_LABELS[sample['visualization_group']]} | "
        'DWT-FDSP 中间特征诊断', fontsize=15)

    add_image(axes[0, 0], cap['resized_rgb'], '01 输入图（检测缩放，RGB）')
    add_image(axes[0, 1], np.moveaxis(cap['dwt_input_rgb01'], 0, -1),
              '02 DWT输入 RGB（[0,1]）')
    add_image(axes[0, 2], lowpass_rgb(cap['ll']),
              '03 LL/2（RGB低频代理图）')
    add_signed(axes[0, 3], feature_scalar(cap['idwt_output']),
               scales['idwt'],
               '04 IDWT输出（RGB有符号均值）')

    add_signed(axes[1, 0], feature_scalar(cap['lh']), scales['detail'],
               '05 LH细节带（RGB有符号均值）')
    add_signed(axes[1, 1], feature_scalar(cap['hl']), scales['detail'],
               '06 HL细节带（RGB有符号均值）')
    add_signed(axes[1, 2], feature_scalar(cap['hh']), scales['detail'],
               '07 HH细节带（RGB有符号均值）')
    add_signed(axes[1, 3], feature_scalar(cap['ll_prime']),
               scales['ll_prime'], "08 IDWT前的LL'（无IDWT特征视图）")

    for col, channel in enumerate(('R', 'G', 'B')):
        ll_chan = cap['log_ll'][col]
        panel_number = {'R': 9, 'G': 10, 'B': 11}[channel]
        add_image(axes[2, col], log_display(ll_chan[None])[..., 0],
                  f'{panel_number:02d} FDSP输入 log(LL_{channel})',
                  cmap='gray', vmin=0, vmax=1)
        add_signed(axes[3, col], cap['fdsp_raw'][col],
                   scales['fdsp_raw'],
                   f'{panel_number + 4:02d} FDSP原始响应 {channel}（仿射变换前）')
        add_signed(axes[4, col], cap['ll_prime'][col],
                   scales['ll_prime'],
                   f"{panel_number + 8:02d} LL'通道 {channel}（仿射变换后）")

    add_signed(axes[2, 3], feature_scalar(cap['ll_pre']),
               scales['ll_coeff'], '12 LL系数（RGB有符号均值）')
    add_signed(axes[3, 3], feature_scalar(cap['ll_prime']),
               scales['ll_prime'], "16 Cat(FDSP_R, G, B) = LL'（通道均值）")

    grad_ll = gradient_magnitude(cap['log_ll'])
    grad_fdsp = gradient_magnitude(cap['ll_prime'])
    add_image(axes[4, 3], grad_fdsp,
              "20 LL'梯度幅值（共享色阶）",
              cmap='magma', vmin=0, vmax=scales['gradient'])
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(output_dir / f"{Path(sample['name']).stem}_diagnostics.png",
                dpi=180, bbox_inches='tight')
    plt.close(fig)

    edge = grad_fdsp >= np.percentile(grad_fdsp, 85)
    edge_fig, edge_axes = plt.subplots(1, 3, figsize=(14, 4))
    add_image(edge_axes[0], grad_ll, 'log(LL)梯度幅值',
              cmap='magma', vmin=0, vmax=scales['gradient'])
    add_image(edge_axes[1], grad_fdsp, "LL'梯度幅值",
              cmap='magma', vmin=0, vmax=scales['gradient'])
    add_image(edge_axes[2], edge.astype(np.uint8),
              "LL'梯度最高15%区域（逐图阈值）",
              cmap='gray', vmin=0, vmax=1)
    edge_fig.tight_layout()
    edge_fig.savefig(
        output_dir / f"{Path(sample['name']).stem}_gradient_edges.png",
        dpi=180, bbox_inches='tight')
    plt.close(edge_fig)

    idwt_fig, idwt_axes = plt.subplots(1, 2, figsize=(10, 4))
    add_signed(idwt_axes[0], feature_scalar(cap['idwt_output']),
               scales['idwt'], 'IDWT输出（RGB有符号均值）')
    idwt_pseudo_rgb = np.moveaxis(np.clip(
        (cap['idwt_output'] + scales['idwt']) / (2 * scales['idwt']),
        0, 1), 0, -1)
    add_image(idwt_axes[1], idwt_pseudo_rgb,
              'IDWT输出伪 RGB（共享有符号色阶）')
    idwt_fig.tight_layout()
    idwt_fig.savefig(
        output_dir / f"{Path(sample['name']).stem}_idwt_views.png",
        dpi=180, bbox_inches='tight')
    plt.close(idwt_fig)


def draw_boxes(ax, image, boxes, labels, scores, class_names, color,
               score_labels, max_boxes=5):
    ax.imshow(image)
    count = len(boxes) if max_boxes is None else min(len(boxes), max_boxes)
    for i in range(count):
        x1, y1, x2, y2 = boxes[i]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                               fill=False, edgecolor=color, linewidth=1.4))
        cls = int(labels[i])
        name = (CLASS_LABELS_ZH.get(class_names[cls], class_names[cls])
                if 0 <= cls < len(class_names)
                else str(cls))
        text = name if not score_labels else f'{name} {scores[i]:.2f}'
        ax.text(x1, max(0, y1 - 2), text, color='white', fontsize=7,
                bbox=dict(facecolor=color, alpha=0.85, pad=1,
                          edgecolor='none'))
    ax.axis('off')
    return count


def make_detection_panel(sample, class_names, output_dir):
    image = cv2.imread(str(sample['image_path']), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    objects = sample['objects']
    gt_boxes = np.asarray([obj['bbox'] for obj in objects], dtype=np.float32)
    gt_labels = np.asarray([
        class_names.index(obj['name']) if obj['name'] in class_names else -1
        for obj in objects
    ], dtype=np.int64)
    cap = sample['capture']
    order = np.argsort(-cap['pred_scores'])
    boxes = cap['pred_boxes'][order]
    scores = cap['pred_scores'][order]
    labels = cap['pred_labels'][order]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].set_title(f'真实标注（{len(gt_boxes)}个目标）')
    shown_gt = draw_boxes(axes[0], image, gt_boxes, gt_labels,
                          np.ones(len(gt_boxes)), class_names, '#15a34a',
                          score_labels=False, max_boxes=None)
    axes[1].set_title(
        f"模型预测（分数 >= 0.05；共{len(boxes)}个；"
        f'显示前{min(len(boxes), 5)}个）')
    shown_pred = draw_boxes(axes[1], image, boxes, labels, scores,
                            class_names, '#dc2626', score_labels=True)
    fig.suptitle(f"{sample['name']} — 模型预测与真实标注",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(
        output_dir / f"{Path(sample['name']).stem}_detections.png",
        dpi=180, bbox_inches='tight')
    plt.close(fig)
    return {'ground_truth_displayed': shown_gt,
            'predictions_total': len(boxes),
            'predictions_displayed': shown_pred,
            'prediction_score_threshold': 0.05}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize a FrontNet DWT-FDSP checkpoint on an XML dataset.')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path,
                        help='Dataset root; defaults to the configured test dataset root.')
    parser.add_argument('--split',
                        help='Test split file; defaults to the configured ann_file.')
    parser.add_argument('--image-subdir',
                        help='Image directory; defaults to the dataset setting or JPEGImages.')
    parser.add_argument('--annotation-subdir',
                        help='XML directory; defaults to the dataset setting or Annotations.')
    parser.add_argument('--output-dir', type=Path,
                        help='Output directory; defaults to work_dirs/visualizations/<checkpoint>.')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f'Missing config or checkpoint: {config_path}, {checkpoint_path}')

    model = init_detector(
        str(config_path), str(checkpoint_path), device=args.device)
    if not hasattr(model, 'front_net') or not hasattr(model.front_net, 'dwtnet'):
        raise RuntimeError('Loaded model does not expose front_net.dwtnet')

    dataset_cfg = model.cfg.test_dataloader.dataset
    while isinstance(dataset_cfg, dict) and 'dataset' in dataset_cfg:
        dataset_cfg = dataset_cfg['dataset']
    data_prefix = dataset_cfg.get('data_prefix', {})
    sub_data_root = data_prefix.get('sub_data_root', '')
    if args.data_root is not None:
        data_root = args.data_root.expanduser()
    else:
        configured_root = dataset_cfg.get('data_root')
        if not configured_root:
            raise ValueError('No data_root in test dataset config; pass --data-root.')
        data_root = Path(configured_root).expanduser()
    if sub_data_root:
        data_root = data_root / sub_data_root
    data_root = data_root.resolve()

    split_file = args.split or dataset_cfg.get('ann_file', 'test.txt')
    image_subdir = args.image_subdir or dataset_cfg.get(
        'img_subdir', 'JPEGImages')
    annotation_subdir = args.annotation_subdir or dataset_cfg.get(
        'ann_subdir', 'Annotations')
    dataset_name = dataset_cfg.get('type', 'XML dataset')
    output_dir = args.output_dir or (
        REPO_ROOT / 'work_dirs' / 'visualizations' / checkpoint_path.stem)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / 'samples'
    samples_dir.mkdir(exist_ok=True)

    records = scan_test_split(
        data_root, split_file, image_subdir, annotation_subdir)
    samples = select_representatives(records)
    print(f'Scanned {len(records)} test images; selected:')
    for sample in samples:
        print(f"  {sample['visualization_group']}: {sample['name']} "
              f"(mean Y={sample['metrics']['mean_luminance_0_255']:.1f}, "
              f"GT={sample['metrics']['ground_truth_count']})")

    class_names = list(model.dataset_meta['classes'])
    dwtnet = model.front_net.dwtnet
    fdsp_mean = list(dwtnet.input_mean)
    fdsp_std = list(dwtnet.input_std)

    for sample in samples:
        sample_dir = samples_dir / Path(sample['name']).stem
        sample_dir.mkdir(exist_ok=True)
        sample['sample_dir'] = sample_dir
        source_bgr = cv2.imread(str(sample['image_path']), cv2.IMREAD_COLOR)
        if source_bgr is None:
            raise RuntimeError(f"Could not read image: {sample['image_path']}")
        sample['original_rgb'] = cv2.cvtColor(
            source_bgr, cv2.COLOR_BGR2RGB)
        sample['capture'] = infer_and_capture(model, sample['image_path'])
        cap = sample['capture']
        print(f"Captured {sample['name']}: input {cap['resized_rgb'].shape}, "
              f"LL {cap['ll'].shape}, predictions "
              f"{len(cap['pred_scores'])}")
        np.savez_compressed(
            sample_dir / f"{Path(sample['name']).stem}_features.npz",
            original_rgb=sample['original_rgb'],
            resized_rgb=cap['resized_rgb'],
            dwt_input_rgb01=cap['dwt_input_rgb01'],
            ll=cap['ll'], lh=cap['lh'], hl=cap['hl'], hh=cap['hh'],
            log_ll=cap['log_ll'],
            ll_pre=cap['ll_pre'], fdsp_raw=cap['fdsp_raw'],
            ll_prime=cap['ll_prime'], idwt_output=cap['idwt_output'],
            frontend_spatial=cap['frontend_spatial'],
            frontend_dwtnet=cap['frontend_dwtnet'],
            frontend_spectral=cap['frontend_spectral'],
            frontend_cat=cap['frontend_cat'],
            frontend_output=cap['frontend_output'],
            frontend_padded_shape=np.asarray(cap['frontend_padded_shape']),
            pred_boxes=cap['pred_boxes'], pred_scores=cap['pred_scores'],
            pred_labels=cap['pred_labels'])

    detail_limit = signed_limit([
        feature_scalar(sample['capture'][band])
        for sample in samples for band in ('lh', 'hl', 'hh')
    ])
    ll_coeff_limit = signed_limit([
        feature_scalar(sample['capture']['ll_pre']) for sample in samples
    ])
    raw_limit = signed_limit([
        sample['capture']['fdsp_raw'] for sample in samples
    ])
    llprime_limit = signed_limit([
        sample['capture']['ll_prime'] for sample in samples
    ])
    idwt_limit = signed_limit([
        feature_scalar(sample['capture']['idwt_output']) for sample in samples
    ])
    gradients = [
        gradient_magnitude(sample['capture']['log_ll'])
        for sample in samples
    ] + [
        gradient_magnitude(sample['capture']['ll_prime'])
        for sample in samples
    ]
    gradient_limit = max(float(np.percentile(
        np.concatenate([x.reshape(-1) for x in gradients]), 99)), 1e-6)

    flow_hws = [sample['capture']['resized_rgb'].shape[:2]
                for sample in samples]
    spatial_maps, spectral_maps, cat_maps = [], [], []
    dwt_means, output_values = [], []
    for sample, (h, w) in zip(samples, flow_hws):
        cap = sample['capture']
        spatial_maps.append(activation_rms(cap['frontend_spatial'][:, :h, :w]))
        spectral_maps.append(activation_rms(cap['frontend_spectral'][:, :h, :w]))
        cat_maps.append(activation_rms(cap['frontend_cat'][:, :h, :w]))
        dwt_means.append(feature_scalar(cap['frontend_dwtnet'][:, :h, :w]))
        output_values.append(cap['frontend_output'][:, :h, :w])

    def positive_limit(arrays):
        values = np.concatenate([x.reshape(-1) for x in arrays])
        return max(float(np.percentile(values, 99)), 1e-6)

    flow_scales = {
        'spatial': positive_limit(spatial_maps),
        'spectral': positive_limit(spectral_maps),
        'cat': positive_limit(cat_maps),
        'dwtnet': signed_limit(dwt_means),
        'output': signed_limit(output_values),
    }
    scales = {'detail': detail_limit, 'll_coeff': ll_coeff_limit,
              'fdsp_raw': raw_limit, 'll_prime': llprime_limit,
              'gradient': gradient_limit, 'idwt': idwt_limit}

    make_key_summary(samples, output_dir, llprime_limit, idwt_limit)
    sample_manifest = []
    for sample in samples:
        sample_dir = sample['sample_dir']
        make_sample_summary(sample, sample_dir, llprime_limit, idwt_limit)
        make_feature_panels(sample, sample_dir, scales)
        make_frontend_flow(sample, sample_dir, flow_scales)
        make_dwt_bands_panel(sample, sample_dir, ll_coeff_limit, detail_limit)
        detection_summary = make_detection_panel(
            sample, class_names, sample_dir)
        sample_manifest.append({
            'image': sample['name'],
            'group': sample['visualization_group'],
            'source_path': str(sample['image_path']),
            'metrics': sample['metrics'],
            'sample_dir': str(sample_dir),
            'features_npz': str(sample_dir /
                                f"{Path(sample['name']).stem}_features.npz"),
            'frontend_flow_png': str(sample_dir /
                                     f"{Path(sample['name']).stem}_frontend_flow.png"),
            'dwt_bands_png': str(sample_dir /
                                 f"{Path(sample['name']).stem}_dwt_bands.png"),
            'detection': detection_summary,
        })

    test_split_path = Path(split_file)
    if not test_split_path.is_absolute():
        test_split_path = data_root / test_split_path
    manifest = {
        'dataset': dataset_name,
        'test_split': str(test_split_path),
        'effective_config': str(config_path),
        'checkpoint': str(checkpoint_path),
        'model_settings': {
            'fdsp_alpha': dwtnet.alpha,
            'fdsp_use_atan': dwtnet.use_atan,
            'fdsp_direction': dwtnet.fdsp_direction,
            'dwt': 'one-level Haar DWT',
        },
        'input_transform': {
            'resize': 'configured by the test pipeline',
            'color': 'RGB', 'mean': fdsp_mean, 'std': fdsp_std,
            'dwt_input': 'normalized input restored to RGB [0,1]',
            'log': 'applied to first-level LL only, before FDSP',
        },
        'visualization_note': (
            "The IDWT panel shows the reconstructed tensor with log-FDSP "
            "processed LL and original RGB-domain detail bands. Exact tensors are "
            "preserved in the NPZ files. LL' is shown before IDWT as a "
            "feature view. The frontend flow PNG shows spatial/spectral "
            "activations, their concatenation, and the exact three-channel "
            "tensor passed to the backbone; feature maps and pseudo-RGB are "
            "visual summaries, with full tensors stored in the NPZ. The "
            "DWT PNG shows the RGB input, raw LL coefficients, an LL/2 "
            "lowpass RGB view, and all three detail bands."),
        'shared_color_scales': scales,
        'frontend_flow_color_scales': flow_scales,
        'samples': sample_manifest,
    }
    (output_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    (output_dir / 'README_可视化说明.md').write_text(
        '# FrontNet DWT-FDSP 可视化\n\n'
        f'数据集：`{dataset_name}`，测试划分：`{test_split_path}`。'
        '脚本按图像亮度、照明均匀度、目标框大小和边缘密度选取代表图。\n\n'
        '- `key_four_column_summary.png`：跨样本的输入、LL、FDSP 和 IDWT 总览。\n'
        '- `samples/<图片名>/`：单样本的特征诊断图、前端双分支流程图、'
        'DWT 子带图、检测结果图及中间张量 NPZ。\n'
        '- `manifest.json`：数据、配置、FDSP 参数、色标和样本记录。\n')
    print(f'Wrote visualization artifacts to {output_dir}')


if __name__ == '__main__':
    main()
