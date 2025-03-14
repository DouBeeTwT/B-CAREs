import cv2, os, torch, BMIL, math, time, shutil, shap
from shap.plots import colors
from torch.nn import functional as F
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle
from openai import OpenAI
from jinja2 import Environment, FileSystemLoader
from datetime import datetime
import matplotlib.font_manager as mfg
import matplotlib as mpl

def report_mkdir(file_path, root="Reports", temp_path="Temp"):
    time0 = time.time()
    dir_name = os.path.basename(file_path).split(".")[0]+"_report"
    dir_name = os.path.join(root, dir_name)
    if not os.path.isdir(dir_name):
        os.makedirs(name=dir_name)
    time1 = time.time()
    print("[{:5.2f} s] - Dir {} has been created already".format(time1-time0, dir_name))
    shutil.copy(file_path, os.path.join(dir_name, "img.png"))
    time2 = time.time()
    print("[{:5.2f} s]   - [Copy] original ultrasound image".format(time2-time1))
    shutil.copy(os.path.join(temp_path, "temp.html"), os.path.join(dir_name, "temp.html"))
    time3 = time.time()
    print("[{:5.2f} s]   - [Copy] temp.html".format(time3-time2))

    return dir_name

def predict(img):
    # img [1 3 224 224] 
    img = torch.tensor(img).float().permute(0,3,1,2)
    img = img.to("cuda:1")
    model = BMIL.model.BMIL(backbone_name="ResNet101", class_num=7, base_num=4, attention_size=14)
    model.load_state_dict(torch.load("pth/BMIL_v4_Exp010/Last.pth", map_location="cpu"))
    model.to("cuda:1").eval()
    decoder = BMIL.utils.decoder.FCOSDecoder(max_object_num=7, topn=50, base_num=4, attention_size=14, class_num=7)
    preds,_ = model(img)
    preds = decoder(preds)
    output,_ = preds["cls"].max(dim=1)
    return output

def plt_mask(dir_name, img_bg, img_mk, box):
    time0 = time.time()
    # display
    figure = plt.figure(figsize=(4,4))
    ax = plt.subplot(111)
    ax.imshow(img_bg, cmap="gray")
    ax.imshow(img_mk.transpose(1,0), cmap="Greens", alpha=0.5)
    ax.add_patch(Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                        edgecolor="#4D8872",facecolor=(0, 0, 0, 0), linewidth=1))
    ax.set_axis_off()
    plt.savefig(os.path.join(dir_name,"mask.png"), bbox_inches='tight', pad_inches=0)
    time1 = time.time()
    print("[{:5.2f} s] - Draw mask image save at {}".format(time1-time0, os.path.join(dir_name,"mask.png")))

def plt_attention(dir_name, device):
    time0 = time.time()
    image = Image.open(os.path.join(dir_name, "img.png")).convert("RGB")
    image = T.ToTensor()(image).unsqueeze(0).to(device)
    X = image.permute(0,2,3,1)

    model = BMIL.model.BMIL(backbone_name="ResNet101", class_num=7, base_num=4, attention_size=14)
    model.load_state_dict(torch.load("pth/BMIL_v4_Exp010/Last.pth", map_location="cpu"))
    model.to(device).eval()
    masker_blur = shap.maskers.Image("blur(128, 128)", X[0].shape)
    explainer = shap.Explainer(predict, masker_blur, output_names=["FBB", "BIC", "DICS", "IDC", "PBC", "LBC", "UDF"])
    shap_values = explainer(X[0:1], max_evals=1000, batch_size=100, outputs=shap.Explanation.argsort.flip[:1])

    shap_values.data = shap_values.data.cpu().numpy()
    shap_values.values = [val for val in np.moveaxis(shap_values.values, -1, 0)]
    a = np.asarray(shap_values.values)[0,0,:,:,0]

    plt.figure(figsize=(4,4))
    ax = plt.subplot(111)
    ax.imshow(a, cmap=colors.red_transparent_blue)
    ax.imshow(shap_values.data[0], cmap="gray", alpha=0.5)
    plt.axis('off')
    plt.savefig(os.path.join(dir_name,"attention.png"), bbox_inches='tight', pad_inches=0)
    time1 = time.time()
    print("[{:5.2f} s] - Draw attention image save at {}".format(time1-time0,
                                                                os.path.join(dir_name,"attention.png")))

def get_score(model_pth,backbone_name, device, class_num, transform, image_pth, bool_mask=False, bool_box=False):
    time0 = time.time()
    model = BMIL.model.BMIL(backbone_name=backbone_name, class_num=class_num, base_num=4, attention_size=14)
    model.load_state_dict(torch.load(model_pth, map_location="cpu"))
    model.to(device).eval()
    decoder = BMIL.utils.decoder.FCOSDecoder(max_object_num=7, topn=50, base_num=4, attention_size=14, class_num=class_num)
    image = Image.open(image_pth).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    preds, _, = model(image)
    preds = decoder(preds) # dict_keys(['cls', 'reg', 'ctr', 'atn', 'fea', 'pos', 'map', 'scores', 'labels', 'bboxes', 'msk'])
    mask = preds['msk'][0,0].detach().cpu().numpy() if bool_mask else None
    box = preds['bboxes'][0,0].detach().cpu().numpy() if bool_box else None
    if bool_mask and bool_box:
        img_bg = cv2.imread(image_pth, cv2.IMREAD_GRAYSCALE)
        box = np.array(box, dtype=np.int32)
        mask = cv2.resize(mask, (box[3]-box[1], box[2]-box[0]), interpolation=cv2.INTER_NEAREST)
        img_mk = np.zeros((img_bg.shape[0],img_bg.shape[1]))
        img_mk[box[0]:box[2], box[1]:box[3]] = mask
        img_mk = np.where(img_mk > 0, 1, 0)
        mask = torch.tensor(img_mk).unsqueeze(0).unsqueeze(0)
        img_tensor = torch.tensor(img_bg).unsqueeze(0)
        box = [torch.tensor(box)]
    else:
        img_tensor = None
    score = preds["cls"].detach().cpu().numpy()[0]
    score = score.max(axis=0)
    time1 = time.time()
    print("[{:5.2f} s]- Predict from model pth: {}".format(time1-time0, model_pth))
    return {"score":score, "mask":mask, "box":box, "img":img_tensor}

def calc_features(msk, img, box):
    time0 = time.time()
    features = {}
    # msk [Batch, 1, 224, 224]
    if len(msk.shape) == 3:
        B, W, H = msk.shape[0], 224, 224
        msk = msk.reshape(B, 1, W, H)
    msk = msk.float()
    A = msk.sum(dim=(2, 3)).squeeze(1) # A [B] 面积         
    kernel = torch.tensor([[1., 1., 1.],
                           [1., -8., 1.],
                           [1., 1., 1.]], 
                           dtype=torch.float32, requires_grad=False)
    kernel = kernel.view(1, 1, 3, 3).to(msk.device)
    edges = F.conv2d(msk, kernel, padding=1)
    P = (edges>0).sum(dim=(1,2,3)) # P [B] 周长
    R = 1-4*math.pi*A / (P**2)
    del kernel
    features["Roundness"] = R.item()
    
    B, W, H = img.shape
    if len(msk.shape)==4 and msk.shape[1]==1:
        msk = msk.squeeze(1)
    msk = msk.reshape(msk.shape[0], -1)
    img = img /255
    img = img.reshape(B, -1)
    img = torch.mul(msk, img) # [Batch, 224*224]
    mean, std, Max = [], [], []
    for one_img in img:
        one_img_p = one_img[one_img>0]
        features["Intensity"] =(0.5+torch.mean(one_img[one_img==0])-torch.mean(one_img[one_img>0])).item()
        features["Intensity"] = (features["Intensity"]-(-0.139794))/(0.474896-(-0.139794))
        features["Uniformity"] =(torch.std(one_img_p)).item()
        features["Uniformity"] = features["Uniformity"]/0.255344
        sort,_ = torch.sort(one_img_p)
        features["Calcification"] =(torch.mean(sort[-int(msk.sum()*0.01):])).item()

    
    features["Aspect"] = (abs(box[3]-box[1])/abs(box[2]-box[0])).item()
    features["Aspect"] = (features["Aspect"]-0.069767)/(2.854839-0.069767)
    time1 = time.time()
    print("[{:5.2f} s]- Calculate echo features from image".format(time1-time0))
    return features

def switch(name:str, index:int):
    Dict= {"Subtype": ["benign","invasive","in-situ","invasive","in-situ", "invasive", "invasive"],
           "Subtype_score": [0,3,1,2,1,2,3],
           "BIRADS": ["1-3", "4a", "4b", "4c", "5-6"],
           "Lymph": ["0", "1 or 2", "more than 3"],
           "Roundness": ["regular edge","not smooth edge"],
           "Intensity": ["no echo", "equal echo", "low echo"],
           "Uniformity": ["uniformality", "non-uniformality"],
           "Calcification": ["no", "yes"],
           "Aspect": ["horizontally growth", "vertically growth"],
           "bi": ["No", "Yes"]
           }
    return Dict[name][index]

def deepseek_feature(clinical_info=None, language="中文"):
    time0 = time.time()
    client = OpenAI(api_key="sk-vpmlzisjwdchdrrkkgcisbyqffifmlsxtvkivumhjmgmwuwv", base_url="https://api.siliconflow.cn/v1")
    infomation_sentence = "乳腺癌超声图中有一个{Aspect}，{Roundness}，{Mean}，回声{Std}，{Max}钙化的病灶区域。请用一句话将这五个特点转述为对超声结节的特征描述话语，使语句通顺像医生一样专业,总结于超声特征标题下。所有结果不要markdown标记，使用纯文字输出，用{language}输出。".format(
        Aspect=clinical_info["Aspect"], Roundness=clinical_info["Roundness"], Mean=clinical_info["Intensity"],
        Std=clinical_info["Uniformity"], Max=clinical_info["Calcification"], language=language
    )
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[
            {"role": "system", "content": "你是一名专业的乳腺癌医生"},
            {"role": "user", "content": infomation_sentence},
        ],
        stream=False
    )
    time1 = time.time()
    print("[{:5.2f} s]- DeepSeek v3 write echo features".format(time1-time0))
    return response.choices[0].message.content

def deepseek_conclusion(clinical_info=None, language="中文"):
    time0 = time.time()
    client = OpenAI(api_key="sk-vpmlzisjwdchdrrkkgcisbyqffifmlsxtvkivumhjmgmwuwv", base_url="https://api.siliconflow.cn/v1")
    infomation_sentence = "我是一名{age}岁{gender}性乳腺癌患者，{menopause}停经,{family}家族史，{smoke}吸烟史，{drink}饮酒史，超声检测BIRADS {BIRADS}级，{Subtype}乳腺癌，{Lymph}淋巴结转移。请用一句话分析这是一名什么样的患者，然后总结一个初步结论，给出你觉得这个患者是高风险还是中等风险还是低风险患者。所有结果不要markdown标记，使用纯文字输出，用{language}输出。".format(
        age=clinical_info["age"], gender=clinical_info["gender"],menopause=clinical_info["menopause"],
        family=clinical_info["family"], smoke=clinical_info["smoke"], drink=clinical_info["drink"],
        BIRADS=clinical_info["BIRADS"], Subtype=clinical_info["Subtype"], Lymph=clinical_info["Lymph"], language=language
    )
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[
            {"role": "system", "content": "你是一名专业的乳腺癌医生"},
            {"role": "user", "content": infomation_sentence},
        ],
        stream=False
    )
    time1=time.time()
    print("[{:5.2f} s]- DeepSeek v3 write conclusion".format(time1-time0))
    return response.choices[0].message.content

def plot_scores(score, labels, fig_name, dir_name):
    time0=time.time()
    figurue = plt.figure(figsize=(4,4))
    mfg.fontManager.addfont("../../.fonts/Bahnschrift.ttf")
    mpl.rcParams['font.family'] = 'Bahnschrift'
    ax = plt.subplot(111)
    bars = ax.bar(np.arange(len(score)), score, color = "#4D8872")
    ax.bar_label(bars, fmt="%.3f", rotation=45, padding=4)
    ax.set_ylim(0,1.2)
    ax.set_yticks(np.arange(0,1.01,0.2))
    plt.xticks(rotation=45)
    ax.set_xticks(np.arange(0,len(score),1))
    ax.set_xticklabels(labels)
    plt.savefig(os.path.join(dir_name, fig_name), bbox_inches='tight', pad_inches=0)
    time1=time.time()
    print("[{:5.2f} s]- Draw score image save at {}".format(time1-time0, os.path.join(dir_name, fig_name)))




def ABCGMS(clinical_info, file_path="Temp/Figure.png", language="英文", device="cuda:1"):
    dir_name = report_mkdir(file_path)
    # BIRADS
    results_BIRADS  = get_score(model_pth="pth/QPMC_BIRADS_Exp001/Best.pth", device="cuda:1", backbone_name="StarNet",
                            class_num=5, transform=T.Compose([T.ToTensor()]), 
                            image_pth=file_path)
    clinical_info["BIRADS"] = switch(name="BIRADS", index=results_BIRADS["score"].argmax())
    clinical_info["BIRADS_score"] = results_BIRADS["score"].argmax()
    score1_png = plot_scores(score=results_BIRADS["score"],
                             labels=["1-3", "4a", "4b", "4c", "5-6"],
                             fig_name="score1.png", dir_name=dir_name)
    # Subtype
    results_Subtype = get_score(model_pth="pth/BMIL_v4_Exp010/Last.pth", device="cuda:1", backbone_name="ResNet101",
                                class_num=7, transform=T.Compose([T.ToTensor()]), 
                                image_pth=file_path, bool_mask=True, bool_box=True)
    clinical_info["Subtype"] = switch(name="Subtype", index=results_Subtype["score"].argmax())
    clinical_info["Subtype_score"] = switch(name="Subtype_score", index=results_Subtype["score"].argmax())
    score2_png = plot_scores(score=results_Subtype["score"],
                            labels=["BF", "IBCN", "DCIS", "IDC", "LCIS", "IPBC", "IBCU"],
                            fig_name="score2.png", dir_name=dir_name)
    # Lymph
    results_Lymph   = get_score(model_pth="pth/QPMC_Lymph_Exp002/Best.pth", device="cuda:1", backbone_name="StarNet",
                                class_num=3, transform=T.Compose([T.ToTensor()]), 
                                image_pth=file_path)
    clinical_info["Lymph"] = switch(name="Lymph", index=results_Lymph["score"].argmax())
    clinical_info["Lymph_score"] = results_Lymph["score"].argmax()
    score3_png = plot_scores(score=results_Lymph["score"],
                            labels=["0", "1-2", ">3"],
                            fig_name="score3.png", dir_name=dir_name)
    # Features
    features = calc_features(msk=results_Subtype["mask"], img=results_Subtype["img"], box=results_Subtype["box"][0])
    clinical_info["features_score"] = 0
    features_score, features_label = [], []
    for key in features.keys():
        if features[key] == None:
            features[key] = 0.0
        features_score.append(features[key])
        features_label.append(key)
        n = 3 if key == "Mean" else 2
        clinical_info[key] = switch(name=key, index=int(features[key]*n))
        clinical_info["features_score"] += int(features[key]*n)
    score4_png = plot_scores(score=features_score,
                            labels=features_label,
                            fig_name="score4.png", dir_name=dir_name)
    # deepseek v3
    clinical_info["feature"] = deepseek_feature(clinical_info=clinical_info, language="英文")
    clinical_info["conclusion"] = deepseek_conclusion(clinical_info=clinical_info, language="英文")
    clinical_info["Date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Attention
    decoder = BMIL.utils.decoder.FCOSDecoder(max_object_num=7, topn=50, base_num=4, attention_size=14, class_num=7)
    plt_attention(dir_name, device=device)
    # Mask
    plt_mask(dir_name=dir_name, img_bg=results_Subtype["img"].squeeze(0), img_mk=results_Subtype["mask"][0,0,:,:], box=results_Subtype["box"][0])
    # Create html
    env = Environment(loader=FileSystemLoader('./'))
    template = env.get_template(os.path.join(dir_name, "temp.html"))
    with open(os.path.join(dir_name, "report.html"), "w+") as f:
        out = template.render(Age=clinical_info["age"],
                            Gender= clinical_info["gender"],
                            FamilyHistory=clinical_info["family"],
                            Smoke=clinical_info["smoke"],
                            Drink=clinical_info["drink"],
                            Menopause=clinical_info["menopause"],
                            BIRADS=clinical_info["BIRADS"],
                            risk_score_1=clinical_info["BIRADS_score"],
                            p_1=round(results_BIRADS["score"].max()*100,2),
                            Subtype=clinical_info["Subtype"],
                            risk_score_2=clinical_info["Subtype_score"],
                            p_2=round(results_Subtype["score"].max()*100,2),
                            Lymph=clinical_info["Lymph"],
                            risk_score_3=clinical_info["Lymph_score"],
                            p_3=round(results_Lymph["score"].max()*100,2),
                            features=clinical_info["feature"],
                            risk_score_4=clinical_info["features_score"],
                            total_risk_score=clinical_info["BIRADS_score"]+clinical_info["Subtype_score"]+clinical_info["Lymph_score"]+clinical_info["features_score"],
                            conclusion=clinical_info["conclusion"],
                            Date=clinical_info["Date"])
        f.write(out)
    os.remove(os.path.join(dir_name, "temp.html"))

if __name__ == "__main__":
    # Single patient
    clinical_info = {"age":52,
                    "gender": "Female",
                    "menopause":"No",
                    "family":"No",
                    "smoke":"No",
                    "drink":"No"}
    file_path = "dataset/test/img/BIC_0009.png"
    ABCGMS(clinical_info, file_path)