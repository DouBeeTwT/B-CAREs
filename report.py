"""Generate a single-patient B-CAREs report from one ultrasound image.

Examples
--------
    python report.py --config configs/infer.yaml
    python report.py --config configs/infer.yaml --image dataset/test/img/IDC_0436.png --device cpu
    python report.py --config configs/infer.yaml --llm        # needs SILICONFLOW_API_KEY

Pipeline: three task models (BI-RADS, histological subtype, sentinel lymph node
metastasis) -> morphological features from the segmentation mask -> weighted risk
score -> HTML report.  The narrative paragraphs are written by an LLM and are
optional; with ``llm.enabled: false`` the report is generated entirely offline.

Outputs, in <output_dir>/<image>_report/:
    report.html     the report
    img.png         input image
    mask.png        segmentation overlay
    score1-4.png    per-task score bars
    attention.png   SHAP attention map (only with --explain)
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from matplotlib.patches import Rectangle
from jinja2 import Environment, FileSystemLoader
from PIL import Image
from torch.nn import functional as F

from bcares.config import INFER_KEYMAP, parse_with_config, resolve_device
from bcares.labels import (
    BIRADS_CLASSES,
    SLN_CLASSES,
    birads_label,
    birads_score,
    sln_label,
    sln_score,
    subtype_category,
    subtype_code,
    subtype_name,
    subtype_score,
)

SUBTYPE_OUTPUT_NAMES = ["FBB", "BIC", "DICS", "IDC", "PBC", "LBC", "UDF"]
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

BMIL_MISSING = """
The `BMIL` package is not importable, so no model can be loaded.

Run report.py from the repository root (the directory that contains BMIL/) or
install the package with `pip install -e .`.  If you only want to check that the
report pipeline is wired correctly, use the demo instead:

    python demo.py --config configs/demo.yaml --random-weights
"""


# --------------------------------------------------------------------------- #
# files and rendering
# --------------------------------------------------------------------------- #
def report_mkdir(file_path, root="Reports"):
    """Create <root>/<image>_report/, copy the input image into it, return the path."""
    time0 = time.time()
    dir_name = os.path.basename(file_path).split(".")[0] + "_report"
    dir_name = os.path.join(root, dir_name)
    os.makedirs(dir_name, exist_ok=True)
    shutil.copy(file_path, os.path.join(dir_name, "img.png"))
    print("[{:5.2f} s] - Dir {} is ready".format(time.time() - time0, dir_name))
    return dir_name


def plt_mask(dir_name, img_bg, img_mk, box):
    time0 = time.time()
    figure = plt.figure(figsize=(4, 4))
    ax = plt.subplot(111)
    ax.imshow(img_bg, cmap="gray")
    ax.imshow(np.asarray(img_mk).transpose(1, 0), cmap="Greens", alpha=0.5)
    ax.add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                           edgecolor="#4D8872", facecolor=(0, 0, 0, 0), linewidth=1))
    ax.set_axis_off()
    out = os.path.join(dir_name, "mask.png")
    plt.savefig(out, bbox_inches="tight", pad_inches=0)
    plt.close(figure)
    print("[{:5.2f} s] - Mask image written to {}".format(time.time() - time0, out))


def plt_attention(dir_name, device, weights_path, random_weights=False, max_evals=200):
    """SHAP attention map for the subtype model (slow: ~minutes on CPU)."""
    import shap
    from shap.plots import colors

    image = Image.open(os.path.join(dir_name, "img.png")).convert("RGB")
    image = T.ToTensor()(image).unsqueeze(0).to(device)
    x = image.permute(0, 2, 3, 1)

    def _predict(batch):
        return predict(batch, weights_path, device, random_weights=random_weights)

    masker_blur = shap.maskers.Image("blur(128, 128)", x[0].shape)
    explainer = shap.Explainer(_predict, masker_blur, output_names=SUBTYPE_OUTPUT_NAMES)
    values = explainer(x[0:1], max_evals=max_evals, batch_size=min(100, max_evals),
                       outputs=shap.Explanation.argsort.flip[:1])

    values.data = values.data.cpu().numpy()
    values.values = [val for val in np.moveaxis(values.values, -1, 0)]
    attention = np.asarray(values.values)[0, 0, :, :, 0]

    figure = plt.figure(figsize=(4, 4))
    ax = plt.subplot(111)
    ax.imshow(attention, cmap=colors.red_transparent_blue)
    ax.imshow(values.data[0], cmap="gray", alpha=0.5)
    plt.axis("off")
    plt.savefig(os.path.join(dir_name, "attention.png"), bbox_inches="tight", pad_inches=0)
    plt.close(figure)


def plot_scores(score, labels, fig_name, dir_name):
    figure = plt.figure(figsize=(4, 4))
    ax = plt.subplot(111)
    bars = ax.bar(np.arange(len(score)), score, color="#4D8872")
    ax.bar_label(bars, fmt="%.3f", rotation=45, padding=4)
    ax.set_ylim(0, 1.2)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    plt.xticks(rotation=45)
    ax.set_xticks(np.arange(0, len(score), 1))
    ax.set_xticklabels(labels)
    plt.savefig(os.path.join(dir_name, fig_name), bbox_inches="tight", pad_inches=0)
    plt.close(figure)


# --------------------------------------------------------------------------- #
# model inference
# --------------------------------------------------------------------------- #
def _load_model(model_pth, backbone_name, class_num, device, random_weights=False):
    import BMIL
    import BMIL.model
    import BMIL.utils.decoder

    model = BMIL.model.BMIL(backbone_name=backbone_name, class_num=class_num,
                            base_num=4, attention_size=14)
    if not random_weights:
        if not os.path.isfile(model_pth):
            raise FileNotFoundError(
                "checkpoint not found: {}\n"
                "Download the weights listed in docs/WEIGHTS.md and place them at this path, "
                "or run demo.py --random-weights to test the pipeline without checkpoints."
                .format(model_pth))
        state = torch.load(model_pth, map_location="cpu")
        model.load_state_dict(state)
    return model.to(device).eval()


def get_score(model_pth, backbone_name, device, class_num, transform, image_pth,
              bool_mask=False, bool_box=False, random_weights=False):
    """Run one task model on one image and return scores (and mask/box if asked)."""
    time0 = time.time()
    import BMIL.utils.decoder

    model = _load_model(model_pth, backbone_name, class_num, device, random_weights)
    decoder = BMIL.utils.decoder.FCOSDecoder(max_object_num=7, topn=50, base_num=4,
                                             attention_size=14, class_num=class_num)
    image = Image.open(image_pth).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    preds, _ = model(image)
    preds = decoder(preds)

    mask = preds["msk"][0, 0].detach().cpu().numpy() if bool_mask else None
    box = preds["bboxes"][0, 0].detach().cpu().numpy() if bool_box else None
    if bool_mask and bool_box:
        img_bg = cv2.imread(image_pth, cv2.IMREAD_GRAYSCALE)
        box = np.array(box, dtype=np.int32)
        mask = cv2.resize(mask, (box[3] - box[1], box[2] - box[0]),
                          interpolation=cv2.INTER_NEAREST)
        img_mk = np.zeros((img_bg.shape[0], img_bg.shape[1]))
        img_mk[box[0]:box[2], box[1]:box[3]] = mask
        img_mk = np.where(img_mk > 0, 1, 0)
        mask = torch.tensor(img_mk).unsqueeze(0).unsqueeze(0)
        img_tensor = torch.tensor(img_bg).unsqueeze(0)
        box = [torch.tensor(box)]
    else:
        img_tensor = None

    score = preds["cls"].detach().cpu().numpy()[0]
    score = score.max(axis=0)
    print("[{:5.2f} s] - {:6s} | {}".format(time.time() - time0, backbone_name,
                                            os.path.basename(image_pth)))
    return {"score": score, "mask": mask, "box": box, "img": img_tensor}


def predict(img, weights_path, device="cuda:0", random_weights=False):
    """Subtype-model forward pass used by the SHAP explainer (img: NHWC tensor)."""
    import BMIL
    import BMIL.model
    import BMIL.utils.decoder

    img = torch.as_tensor(img).float().permute(0, 3, 1, 2).to(device)
    model = _load_model(weights_path, "ResNet101", 7, device, random_weights)
    decoder = BMIL.utils.decoder.FCOSDecoder(max_object_num=7, topn=50, base_num=4,
                                             attention_size=14, class_num=7)
    preds, _ = model(img)
    preds = decoder(preds)
    output, _ = preds["cls"].max(dim=1)
    return output


# --------------------------------------------------------------------------- #
# morphological features
# --------------------------------------------------------------------------- #
def calc_features(msk, img, box):
    """Roundness / intensity / uniformity / calcification / aspect from the mask."""
    features = {}
    if len(msk.shape) == 3:
        batch, width, height = msk.shape[0], 224, 224
        msk = msk.reshape(batch, 1, width, height)
    msk = msk.float()

    area = msk.sum(dim=(2, 3)).squeeze(1)
    kernel = torch.tensor([[1.0, 1.0, 1.0],
                           [1.0, -8.0, 1.0],
                           [1.0, 1.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3).to(msk.device)
    edges = F.conv2d(msk, kernel, padding=1)
    perimeter = (edges > 0).sum(dim=(1, 2, 3))
    features["Roundness"] = (1 - 4 * math.pi * area / (perimeter ** 2)).item()

    batch, width, height = img.shape
    if len(msk.shape) == 4 and msk.shape[1] == 1:
        msk = msk.squeeze(1)
    msk = msk.reshape(msk.shape[0], -1)
    img = (img / 255).reshape(batch, -1)
    img = torch.mul(msk, img)
    for one_img in img:
        one_img_p = one_img[one_img > 0]
        features["Intensity"] = (0.5 + torch.mean(one_img[one_img == 0])
                                 - torch.mean(one_img[one_img > 0])).item()
        features["Intensity"] = (features["Intensity"] - (-0.139794)) / (0.474896 - (-0.139794))
        features["Uniformity"] = torch.std(one_img_p).item() / 0.255344
        sorted_pixels, _ = torch.sort(one_img_p)
        features["Calcification"] = torch.mean(sorted_pixels[-int(msk.sum() * 0.01):]).item()

    features["Aspect"] = (abs(box[3] - box[1]) / abs(box[2] - box[0])).item()
    features["Aspect"] = (features["Aspect"] - 0.069767) / (2.854839 - 0.069767)
    return features


def switch(name: str, index: int, score_mode: str = "paper"):
    """Map a class index to the human-readable value used in the report."""
    feature_labels = {
        "Roundness": ["regular edge", "not smooth edge"],
        "Intensity": ["no echo", "equal echo", "low echo"],
        "Uniformity": ["uniformality", "non-uniformality"],
        "Calcification": ["no", "yes"],
        "Aspect": ["horizontally growth", "vertically growth"],
    }
    if name == "Subtype":
        return subtype_category(index)
    if name == "Subtype_score":
        return subtype_score(index, score_mode)
    if name == "BIRADS":
        return birads_label(index)
    if name == "Lymph":
        return sln_label(index)
    if name in feature_labels:
        return feature_labels[name][index]
    raise KeyError("unknown label group: {}".format(name))


# --------------------------------------------------------------------------- #
# LLM-written narrative (optional)
# --------------------------------------------------------------------------- #
def _llm_client(base_url: str, api_key_env: str):
    from openai import OpenAI

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            "{} is not set. Either export the key, put it in a .env file (see "
            ".env.example) or run with llm.enabled: false.".format(api_key_env))
    return OpenAI(api_key=api_key, base_url=base_url)


def deepseek_feature(clinical_info, language="English", base_url="https://api.siliconflow.cn/v1",
                     model="deepseek-ai/DeepSeek-V3", api_key_env="SILICONFLOW_API_KEY"):
    prompt = (
        "乳腺癌超声图中有一个{Aspect}，{Roundness}，{Mean}，回声{Std}，{Max}钙化的病灶区域。"
        "请用一句话将这五个特点转述为对超声结节的特征描述话语，使语句通顺像医生一样专业,"
        "总结于超声特征标题下。所有结果不要markdown标记，使用纯文字输出，用{language}输出。"
    ).format(Aspect=clinical_info["Aspect"], Roundness=clinical_info["Roundness"],
             Mean=clinical_info["Intensity"], Std=clinical_info["Uniformity"],
             Max=clinical_info["Calcification"], language=language)
    response = _llm_client(base_url, api_key_env).chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "你是一名专业的乳腺癌医生"},
                  {"role": "user", "content": prompt}],
        stream=False,
    )
    return response.choices[0].message.content


def deepseek_conclusion(clinical_info, language="English", base_url="https://api.siliconflow.cn/v1",
                        model="deepseek-ai/DeepSeek-V3", api_key_env="SILICONFLOW_API_KEY"):
    prompt = (
        "我是一名{age}岁{gender}性乳腺癌患者，{menopause}停经,{family}家族史，{smoke}吸烟史，"
        "{drink}饮酒史，超声检测BIRADS {BIRADS}级，{Subtype}乳腺癌，{Lymph}淋巴结转移。"
        "请用一句话分析这是一名什么样的患者，然后总结一个初步结论，给出你觉得这个患者是高风险"
        "还是中等风险还是低风险患者。所有结果不要markdown标记，使用纯文字输出，用{language}输出。"
    ).format(age=clinical_info["age"], gender=clinical_info["gender"],
             menopause=clinical_info["menopause"], family=clinical_info["family"],
             smoke=clinical_info["smoke"], drink=clinical_info["drink"],
             BIRADS=clinical_info["BIRADS"], Subtype=clinical_info["Subtype"],
             Lymph=clinical_info["Lymph"], language=language)
    response = _llm_client(base_url, api_key_env).chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "你是一名专业的乳腺癌医生"},
                  {"role": "user", "content": prompt}],
        stream=False,
    )
    return response.choices[0].message.content


# --------------------------------------------------------------------------- #
# end-to-end report
# --------------------------------------------------------------------------- #
def ABCGMS(clinical_info, file_path, device="cuda:0", weights=None, output_root="Reports",
           llm=None, explain=False, shap_max_evals=200, subtype_score_mode="paper",
           random_weights=False):
    """Build the full report for one image and return the output directory."""
    weights = weights or {}
    llm = llm or {}
    dir_name = report_mkdir(file_path, root=output_root)
    transform = T.Compose([T.ToTensor()])

    # BI-RADS
    results_birads = get_score(model_pth=weights.get("birads", "pth/QPMC_BIRADS_Exp001/Best.pth"),
                               device=device, backbone_name="StarNet", class_num=5,
                               transform=transform, image_pth=file_path,
                               random_weights=random_weights)
    idx = int(results_birads["score"].argmax())
    clinical_info["BIRADS"] = birads_label(idx)
    clinical_info["BIRADS_score"] = birads_score(idx)
    plot_scores(score=results_birads["score"], labels=[c.label for c in BIRADS_CLASSES],
                fig_name="score1.png", dir_name=dir_name)

    # Histological subtype
    results_subtype = get_score(model_pth=weights.get("subtype", "pth/BMIL_v4_Exp010/Last.pth"),
                                device=device, backbone_name="ResNet101", class_num=7,
                                transform=transform, image_pth=file_path,
                                bool_mask=True, bool_box=True, random_weights=random_weights)
    idx = int(results_subtype["score"].argmax())
    clinical_info["Subtype"] = subtype_category(idx)
    clinical_info["Subtype_code"] = subtype_code(idx)
    clinical_info["Subtype_name"] = subtype_name(idx)
    clinical_info["Subtype_score"] = subtype_score(idx, subtype_score_mode)
    plot_scores(score=results_subtype["score"], labels=SUBTYPE_OUTPUT_NAMES,
                fig_name="score2.png", dir_name=dir_name)

    # Sentinel lymph node metastasis
    results_lymph = get_score(model_pth=weights.get("lymph", "pth/QPMC_Lymph_Exp002/Best.pth"),
                              device=device, backbone_name="StarNet", class_num=3,
                              transform=transform, image_pth=file_path,
                              random_weights=random_weights)
    idx = int(results_lymph["score"].argmax())
    clinical_info["Lymph"] = sln_label(idx)
    clinical_info["Lymph_score"] = sln_score(idx)
    plot_scores(score=results_lymph["score"], labels=[c.label for c in SLN_CLASSES],
                fig_name="score3.png", dir_name=dir_name)

    # Morphological features
    features = calc_features(msk=results_subtype["mask"], img=results_subtype["img"],
                             box=results_subtype["box"][0])
    clinical_info["features_score"] = 0
    features_score, features_label = [], []
    for key, value in features.items():
        if value is None:
            features[key] = 0.0
        features_score.append(features[key])
        features_label.append(key)
        # NOTE: the published pipeline thresholds every feature on two bins;
        # kept as-is for numerical reproducibility (see docs/KNOWN_ISSUES.md).
        bins = 2
        clinical_info[key] = switch(key, min(int(features[key] * bins), bins - 1))
        clinical_info["features_score"] += int(features[key] * bins)
    plot_scores(score=features_score, labels=features_label,
                fig_name="score4.png", dir_name=dir_name)

    # Narrative paragraphs
    if llm.get("enabled"):
        clinical_info["feature"] = deepseek_feature(
            clinical_info, language=llm.get("language", "English"),
            base_url=llm.get("base_url", "https://api.siliconflow.cn/v1"),
            model=llm.get("model", "deepseek-ai/DeepSeek-V3"),
            api_key_env=llm.get("api_key_env", "SILICONFLOW_API_KEY"))
        clinical_info["conclusion"] = deepseek_conclusion(
            clinical_info, language=llm.get("language", "English"),
            base_url=llm.get("base_url", "https://api.siliconflow.cn/v1"),
            model=llm.get("model", "deepseek-ai/DeepSeek-V3"),
            api_key_env=llm.get("api_key_env", "SILICONFLOW_API_KEY"))
    else:
        clinical_info["feature"] = "LLM narrative disabled (llm.enabled: false)."
        clinical_info["conclusion"] = "LLM narrative disabled (llm.enabled: false)."

    # Figures
    if explain:
        plt_attention(dir_name, device=device, weights_path=weights.get(
            "subtype", "pth/BMIL_v4_Exp010/Last.pth"), random_weights=random_weights,
            max_evals=shap_max_evals)
    plt_mask(dir_name=dir_name, img_bg=results_subtype["img"].squeeze(0),
             img_mk=results_subtype["mask"][0, 0, :, :], box=results_subtype["box"][0])

    # HTML
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template("report_template.html")
    html = template.render(
        Age=clinical_info["age"], Gender=clinical_info["gender"],
        FamilyHistory=clinical_info["family"], Smoke=clinical_info["smoke"],
        Drink=clinical_info["drink"], Menopause=clinical_info["menopause"],
        BIRADS=clinical_info["BIRADS"], risk_score_1=clinical_info["BIRADS_score"],
        p_1=round(float(results_birads["score"].max()) * 100, 2),
        Subtype=clinical_info["Subtype"], Subtype_name=clinical_info["Subtype_name"],
        risk_score_2=clinical_info["Subtype_score"],
        p_2=round(float(results_subtype["score"].max()) * 100, 2),
        Lymph=clinical_info["Lymph"], risk_score_3=clinical_info["Lymph_score"],
        p_3=round(float(results_lymph["score"].max()) * 100, 2),
        features=clinical_info["feature"], risk_score_4=clinical_info["features_score"],
        total_risk_score=(clinical_info["BIRADS_score"] + clinical_info["Subtype_score"]
                          + clinical_info["Lymph_score"] + clinical_info["features_score"]),
        conclusion=clinical_info["conclusion"],
        Date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        Explain=explain,
    )
    with open(os.path.join(dir_name, "report.html"), "w", encoding="utf-8") as handle:
        handle.write(html)
    return dir_name


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate one B-CAREs report.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="YAML configuration file")
    parser.add_argument("--image", default="dataset/test/img/BIC_0009.png", type=str,
                        help="path to one ultrasound image")
    parser.add_argument("--birads_weights", default="pth/QPMC_BIRADS_Exp001/Best.pth", type=str)
    parser.add_argument("--subtype_weights", default="pth/BMIL_v4_Exp010/Last.pth", type=str)
    parser.add_argument("--lymph_weights", default="pth/QPMC_Lymph_Exp002/Best.pth", type=str)
    parser.add_argument("--device", default="auto", type=str, help="auto | cpu | cuda:0 ...")
    parser.add_argument("--output_dir", default="Reports", type=str)
    parser.add_argument("--subtype_score", default="paper", choices=["paper", "corrected"],
                        help="subtype risk-score lookup (see docs/KNOWN_ISSUES.md)")
    parser.add_argument("--age", default=52, type=int)
    parser.add_argument("--gender", default="Female", type=str)
    parser.add_argument("--menopause", default="No", type=str)
    parser.add_argument("--family", default="No", type=str)
    parser.add_argument("--smoke", default="No", type=str)
    parser.add_argument("--drink", default="No", type=str)
    parser.add_argument("--llm", action="store_true", help="write the narrative with an LLM")
    parser.add_argument("--llm_base_url", default="https://api.siliconflow.cn/v1", type=str)
    parser.add_argument("--llm_model", default="deepseek-ai/DeepSeek-V3", type=str)
    parser.add_argument("--llm_api_key_env", default="SILICONFLOW_API_KEY", type=str)
    parser.add_argument("--language", default="English", type=str)
    parser.add_argument("--explain", action="store_true", help="add a SHAP attention map (slow)")
    parser.add_argument("--shap_max_evals", default=200, type=int)
    parser.add_argument("--random_weights", action="store_true",
                        help="skip checkpoints and use random initialisation (pipeline test)")
    return parser


def main(argv=None) -> int:
    args = parse_with_config(build_parser(), INFER_KEYMAP, argv)
    args.device = resolve_device(args.device)

    if not os.path.isfile(args.image):
        raise SystemExit("image not found: {}".format(args.image))

    clinical_info = {"age": args.age, "gender": args.gender, "menopause": args.menopause,
                     "family": args.family, "smoke": args.smoke, "drink": args.drink}
    llm = {"enabled": args.llm, "base_url": args.llm_base_url, "model": args.llm_model,
           "api_key_env": args.llm_api_key_env, "language": args.language}

    out = ABCGMS(clinical_info, args.image, device=args.device,
                 weights={"birads": args.birads_weights, "subtype": args.subtype_weights,
                          "lymph": args.lymph_weights},
                 output_root=args.output_dir, llm=llm, explain=args.explain,
                 shap_max_evals=args.shap_max_evals, subtype_score_mode=args.subtype_score,
                 random_weights=args.random_weights)
    print("Report written to {}".format(os.path.join(out, "report.html")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
