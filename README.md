# CVPR 2025 论文和开源项目合集(Papers with Code)

CVPR 2025 decisions are now available on OpenReview！22.1% = 2878 / 13008


> 注1：欢迎各位大佬提交issue，分享CVPR 2025论文和开源项目！
>
> 注2：关于往年CV顶会论文以及其他优质CV论文和大盘点，详见： https://github.com/amusi/daily-paper-computer-vision
>
> - [ICCV 2025](https://github.com/amusi/ICCV2025-Papers-with-Code)
> - [ECCV 2024](https://github.com/amusi/ECCV2024-Papers-with-Code)
> - [CVPR 2024](CVPR2024-Papers-with-Code.md)

欢迎扫码加入【CVer学术交流群】，可以获取CVPR 2025等最前沿工作！这是最大的计算机视觉AI知识星球！每日更新，第一时间分享最新最前沿的计算机视觉、AIGC、扩散模型、多模态、深度学习、自动驾驶、医疗影像和遥感等方向的学习资料，快加入学起来！

![](CVer学术交流群.png)

# 【CVPR 2025 论文开源目录】

- [3DGS(Gaussian Splatting)](#3DGS)
- [Agent)](#Agent)
- [Avatars](#Avatars)
- [Backbone](#Backbone)
- [CLIP](#CLIP)EVOS
- [Mamba](#Mamba)
- [Embodied AI](#Embodied-AI)
- [GAN](#GAN)
- [GNN](#GNN)
- [多模态大语言模型(MLLM)](#MLLM)
- [大语言模型(LLM)](#LLM)
- [NAS](#NAS)
- [OCR](#OCR)
- [NeRF](#NeRF)
- [DETR](#DETR)
- [扩散模型(Diffusion Models)](#Diffusion)
- [ReID(重识别)](#ReID)
- [长尾分布(Long-Tail)](#Long-Tail)
- [Vision Transformer](#Vision-Transformer)
- [视觉和语言(Vision-Language)](#VL)
- [自监督学习(Self-supervised Learning)](#SSL)
- [数据增强(Data Augmentation)](#DA)
- [目标检测(Object Detection)](#Object-Detection)
- [异常检测(Anomaly Detection)](#Anomaly-Detection)
- [目标跟踪(Visual Tracking)](#VT)
- [语义分割(Semantic Segmentation)](#Semantic-Segmentation)
- [实例分割(Instance Segmentation)](#Instance-Segmentation)
- [全景分割(Panoptic Segmentation)](#Panoptic-Segmentation)
- [医学图像(Medical Image)](#MI)
- [医学图像分割(Medical Image Segmentation)](#MIS)
- [视频目标分割(Video Object Segmentation)](#VOS)
- [视频实例分割(Video Instance Segmentation)](#VIS)
- [参考图像分割(Referring Image Segmentation)](#RIS)
- [图像抠图(Image Matting)](#Matting)
- [图像编辑(Image Editing)](#Image-Editing)
- [Low-level Vision](#LLV)
- [超分辨率(Super-Resolution)](#SR)
- [去噪(Denoising)](#Denoising)
- [去模糊(Deblur)](#Deblur)
- [自动驾驶(Autonomous Driving)](#Autonomous-Driving)
- [3D点云(3D Point Cloud)](#3D-Point-Cloud)
- [3D目标检测(3D Object Detection)](#3DOD)
- [3D语义分割(3D Semantic Segmentation)](#3DSS)
- [3D目标跟踪(3D Object Tracking)](#3D-Object-Tracking)
- [3D语义场景补全(3D Semantic Scene Completion)](#3DSSC)
- [3D配准(3D Registration)](#3D-Registration)
- [3D人体姿态估计(3D Human Pose Estimation)](#3D-Human-Pose-Estimation)
- [3D人体Mesh估计(3D Human Mesh Estimation)](#3D-Human-Pose-Estimation)
- [3D Visual Grounding(3D视觉定位)](#3DVG)
- [医学图像(Medical Image)](#Medical-Image)
- [图像生成(Image Generation)](#Image-Generation)
- [视频生成(Video Generation)](#Video-Generation)
- [3D生成(3D Generation)](#3D-Generation)
- [视频理解(Video Understanding)](#Video-Understanding)
- [行为检测(Action Detection)](#Action-Detection)
- [具身智能(Embodied AI)](#Embodied)
- [文本检测(Text Detection)](#Text-Detection)
- [知识蒸馏(Knowledge Distillation)](#KD)
- [模型剪枝(Model Pruning)](#Pruning)
- [图像压缩(Image Compression)](#IC)
- [三维重建(3D Reconstruction)](#3D-Reconstruction)
- [深度估计(Depth Estimation)](#Depth-Estimation)
- [轨迹预测(Trajectory Prediction)](#TP)
- [车道线检测(Lane Detection)](#Lane-Detection)
- [图像描述(Image Captioning)](#Image-Captioning)
- [视觉问答(Visual Question Answering)](#VQA)
- [手语识别(Sign Language Recognition)](#SLR)
- [视频预测(Video Prediction)](#Video-Prediction)
- [新视点合成(Novel View Synthesis)](#NVS)
- [Zero-Shot Learning(零样本学习)](#ZSL)
- [立体匹配(Stereo Matching)](#Stereo-Matching)
- [特征匹配(Feature Matching)](#Feature-Matching)
- [暗光图像增强(Low-light Image Enhancement)](#Low-light)
- [场景图生成(Scene Graph Generation)](#SGG)
- [风格迁移(Style Transfer)](#ST)
- [隐式神经表示(Implicit Neural Representations)](#INR)
- [图像质量评价(Image Quality Assessment)](#IQA)
- [视频质量评价(Video Quality Assessment)](#Video-Quality-Assessment)
- [压缩感知(Compressive Sensing)](#CS)
- [数据集(Datasets)](#Datasets)
- [新任务(New Tasks)](#New-Tasks)
- [其他(Others)](#Others)

<a name="3DGS"></a>

# 3DGS(Gaussian Splatting)


<a name="Agent"></a>

# Agent

**SpiritSight Agent: Advanced GUI Agent with One Look**

- Paper: https://arxiv.org/abs/2503.03196
- Code: https://hzhiyuan.github.io/SpiritSight-Agent


<a name="Avatars"></a>

# Avatars


# Backbone

**Building Vision Models upon Heat Conduction**

- Paper: https://arxiv.org/abs/2405.16555
- Code: https://github.com/MzeroMiko/vHeat

**LSNet: See Large, Focus Small**

- Paper: https://arxiv.org/abs/2503.23135
- Code: https://github.com/jameslahm/lsnet


<a name="CLIP"></a>

# CLIP



<a name="Mamba"></a>

# Mamba


**MambaVision: A Hybrid Mamba-Transformer Vision Backbone**

- Paper: https://arxiv.org/abs/2407.08083
- Code: https://github.com/NVlabs/MambaVision

**MobileMamba: Lightweight Multi-Receptive Visual Mamba Network**

- Paper: https://arxiv.org/abs/2411.15941
- Code: https://github.com/lewandofskee/MobileMamba

**MambaIC: State Space Models for High-Performance Learned Image Compression**

- Paper: https://arxiv.org/abs/2503.12461
- Code: https://arxiv.org/abs/2503.12461

<a name="Embodied-AI"></a>

# Embodied AI

**CityWalker: Learning Embodied Urban Navigation from Web-Scale Videos**

- Project: https://ai4ce.github.io/CityWalker/
- Paper: https://arxiv.org/abs/2411.17820
- Code: https://github.com/ai4ce/CityWalker


<a name="GAN"></a>

# GAN

<a name="OCR"></a>

# OCR


<a name="NeRF"></a>

# NeRF



<a name="DETR"></a>

# DETR

**Mr. DETR: Instructive Multi-Route Training for Detection Transformers**

- Paper: https://arxiv.org/abs/2412.10028
- Code: https://github.com/Visual-AI/Mr.DETR


<a name="Prompt"></a>

# Prompt

<a name="MLLM"></a>

# 多模态大语言模型(MLLM)

**LSceneLLM: Enhancing Large 3D Scene Understanding Using Adaptive Visual Preferences**

- Paper： https://arxiv.org/abs/2412.01292
- Code: https://github.com/Hoyyyaard/LSceneLLM


**DynRefer: Delving into Region-level Multimodal Tasks via Dynamic Resolution**

- Paper: https://arxiv.org/abs/2405.16071
- Code: https://github.com/callsys/DynRefer


**Retrieval-Augmented Personalization for Multimodal Large Language Models**

- Project Page: https://hoar012.github.io/RAP-Project/
- Paper: https://arxiv.org/abs/2410.13360
- Code: https://github.com/Hoar012/RAP-MLLM

**BiomedCoOp: Learning to Prompt for Biomedical Vision-Language Models**

- Paper: https://arxiv.org/abs/2411.15232
- Code: https://github.com/HealthX-Lab/BiomedCoOp

**FlashSloth: Lightning Multimodal Large Language Models via Embedded Visual Compression**

- Paper: https://arxiv.org/abs/2412.04317
- Code: https://github.com/codefanw/FlashSloth

**MMRL: Multi-Modal Representation Learning for Vision-Language Models**

- Paper: https://arxiv.org/abs/2503.08497
- Code: https://github.com/yunncheng/MMRL

**PAVE: Patching and Adapting Video Large Language Models**

- Paper: https://arxiv.org/abs/2503.19794
- Code: https://github.com/dragonlzm/PAVE

**AdaMMS: Model Merging for Heterogeneous Multimodal Large Language Models with Unsupervised Coefficient Optimization**

- Paper: https://arxiv.org/abs/2503.23733
- Code: https://github.com/THUNLP-MT/AdaMMS


<a name="LLM"></a>

# 大语言模型(LLM)




<a name="NAS"></a>

# NAS

<a name="ReID"></a>

# ReID(重识别)

**From Poses to Identity: Training-Free Person Re-Identification via Feature Centralization**

- Paper: https://arxiv.org/abs/2503.00938
- Code: https://github.com/yuanc3/Pose2ID


**AirRoom: Objects Matter in Room Reidentification**

- Project: https://sairlab.org/airroom/
- Paper: https://arxiv.org/abs/2503.01130


**IDEA: Inverted Text with Cooperative Deformable Aggregation for Multi-modal Object Re-Identification**

- Paper: https://arxiv.org/abs/2503.10324
- Code: https://github.com/924973292/IDEA



<a name="Diffusion"></a>

# 扩散模型(Diffusion Models)

**TinyFusion: Diffusion Transformers Learned Shallow**

- Paper: https://arxiv.org/abs/2412.01199
- Code: https://github.com/VainF/TinyFusion

**DKDM: Data-Free Knowledge Distillation for Diffusion Models with Any Architecture**

- Paper: https://arxiv.org/abs/2409.03550
- Code: https://github.com/qianlong0502/DKDM

**Tiled Diffusion**

- Homepage: https://madaror.github.io/tiled-diffusion.github.io/
- Paper: https://arxiv.org/abs/2412.15185
- Code: https://github.com/madaror/tiled-diffusion


<a name="Vision-Transformer"></a>

# Vision Transformer



<a name="VL"></a>

# 视觉和语言(Vision-Language)

**NLPrompt: Noise-Label Prompt Learning for Vision-Language Models**

- Paper: https://arxiv.org/abs/2412.01256
- Code: https://github.com/qunovo/NLPrompt

**PhysVLM: Enabling Visual Language Models to Understand Robotic Physical Reachability**

- Paper: https://arxiv.org/abs/2503.08481
- Code: https://github.com/unira-zwj/PhysVLM

**MMRL: Multi-Modal Representation Learning for Vision-Language Models**

- Paper: https://arxiv.org/abs/2503.08497
- Code: https://github.com/yunncheng/MMRL


<a name="Object-Detection"></a>

# 目标检测(Object Detection)


**LLMDet: Learning Strong Open-Vocabulary Object Detectors under the Supervision of Large Language Models**

- Paper: https://arxiv.org/abs/2501.18954
- Code：https://github.com/iSEE-Laboratory/LLMDet

**Mr. DETR: Instructive Multi-Route Training for Detection Transformers**

- Paper: https://arxiv.org/abs/2412.10028
- Code: https://github.com/Visual-AI/Mr.DETR


<a name="Anomaly-Detection"></a>

# 异常检测(Anomaly Detection)



<a name="VT"></a>

# 目标跟踪(Object Tracking)

**Multiple Object Tracking as ID Prediction**

- Paper：https://arxiv.org/abs/2403.16848
- Code: https://github.com/MCG-NJU/MOTIP

**Omnidirectional Multi-Object Tracking**

- Paper:https://arxiv.org/abs/2503.04565
- Code:https://github.com/xifen523/OmniTrack


<a name="MI"></a>

# 医学图像(Medical Image)


**BrainMVP: Multi-modal Vision Pre-training for Medical Image Analysis**

- Paper: https://arxiv.org/abs/2410.10604
- Code: https://github.com/shaohao011/BrainMVP


# 医学图像分割(Medical Image Segmentation)

**Test-Time Domain Generalization via Universe Learning: A Multi-Graph Matching Approach for Medical Image Segmentation**

- Paper: https://arxiv.org/abs/2503.13012
- Code: https://github.com/Yore0/TTDG-MGM


<a name="Autonomous-Driving"></a>

# 自动驾驶(Autonomous Driving)

**LiMoE: Mixture of LiDAR Representation Learners from Automotive Scenes**

- Project: https://ldkong.com/LiMoE
- Paper: https://arxiv.org/abs/2501.04004
- Code: https://github.com/Xiangxu-0103/LiMoE



# 3D点云(3D-Point-Cloud)

**Unlocking Generalization Power in LiDAR Point Cloud Registration**

- Paper: https://arxiv.org/abs/2503.10149
- Code: https://github.com/peakpang/UGP


<a name="3DOD"></a>

# 3D目标检测(3D Object Detection)



<a name="3DOD"></a>

# 3D语义分割(3D Semantic Segmentation)





<a name="LLV"></a>

# Low-level Vision



<a name="SR"></a>

# 超分辨率(Super-Resolution)

**AESOP: Auto-Encoded Supervision for Perceptual Image Super-Resolution**

- Paper: https://arxiv.org/abs/2412.00124
- Code: https://github.com/2minkyulee/AESOP-Auto-Encoded-Supervision-for-Perceptual-Image-Super-Resolution


<a name="Denoising"></a>

# 去噪(Denoising)

## 图像去噪(Image Denoising)

<a name="3D-Human-Pose-Estimation"></a>

# 3D人体姿态估计(3D Human Pose Estimation)

**Reconstructing Humans with a Biomechanically Accurate Skeleton**

- Homepage: https://isshikihugh.github.io/HSMR/
- Code: https://github.com/IsshikiHugh/HSMR

<a name="3DVG"></a>

#3D Visual Grounding(3D视觉定位)

**ProxyTransformation: Preshaping Point Cloud Manifold With Proxy Attention For 3D Visual Grounding**

- Homepage: https://pqh22.github.io/projects/ProxyTransformation/index.html

- Code: https://github.com/pqh22/ProxyTransformation

- Paper: https://arxiv.org/abs/2502.19247


<a name="Image-Generation"></a>

# 图像生成(Image Generation)

**Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models**

- Paper: https://arxiv.org/abs/2501.01423
- Code: https://github.com/hustvl/LightningDiT

**SleeperMark: Towards Robust Watermark against Fine-Tuning Text-to-image Diffusion Models**

- Paper: https://arxiv.org/abs/2412.04852
- Code: https://github.com/taco-group/SleeperMark


**TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation**

- Homepage: https://byteflow-ai.github.io/TokenFlow/
- Code: https://github.com/ByteFlow-AI/TokenFlow
- Paper:https://arxiv.org/abs/2412.03069

**PAR: Parallelized Autoregressive Visual Generation**

- Project: https://epiphqny.github.io/PAR-project/
- Paper: https://arxiv.org/abs/2412.15119
- Code: https://github.com/Epiphqny/PAR


**Generative Photography: Scene-Consistent Camera Control for Realistic Text-to-Image Synthesis**

- Project: https://generative-photography.github.io/project/
- Paper: https://arxiv.org/abs/2412.02168
- Code: https://github.com/pandayuanyu/generative-photography


**OpenING: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation**

- Project Page: https://opening-benchmark.github.io/
- Paper: https://arxiv.org/abs/2411.18499).
- Code: https://github.com/LanceZPF/OpenING




<a name="Video-Generation"></a>

# 视频生成(Video Generation)

**Identity-Preserving Text-to-Video Generation by Frequency Decomposition**

- Paper: https://arxiv.org/abs/2411.17440
- Code: https://github.com/PKU-YuanGroup/ConsisID


**Cinemo: Consistent and Controllable Image Animation with Motion Diffusion Models**

- Paper: https://arxiv.org/abs/2407.15642
- Code: https://github.com/maxin-cn/Cinemo

**X-Dyna: Expressive Dynamic Human Image Animation**

- Paper: https://arxiv.org/abs/2501.10021
- Code: https://github.com/bytedance/X-Dyna

**PhyT2V: LLM-Guided Iterative Self-Refinement for Physics-Grounded Text-to-Video Generation**

- Paper: https://arxiv.org/pdf/2412.00596
- Code: https://github.com/pittisl/PhyT2V


**Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model**

- Project: https://liewfeng.github.io/TeaCache/
- Paper: https://arxiv.org/abs/2411.19108
- Code: https://github.com/ali-vilab/TeaCache


**AR-Diffusion: Asynchronous Video Generation with Auto-Regressive Diffusion**

- Project: https://iva-mzsun.github.io/AR-Diffusion
- Paper: https://arxiv.org/abs/2503.07418
- Code: https://github.com/iva-mzsun/AR-Diffusion


<a name="Image-Editing"></a>

# 图像编辑(Image Editing)

**Edit Away and My Face Will not Stay: Personal Biometric Defense against Malicious Generative Editing**

- Paper: https://arxiv.org/abs/2411.16832
- Code: https://github.com/taco-group/FaceLock


**h-Edit: Effective and Flexible Diffusion-Based Editing via Doob’s h-Transform**

- Paper: https://arxiv.org/abs/2503.02187
- Code: https://github.com/nktoan/h-edit


<a name="Video-Editing"></a>

# 视频编辑(Video Editing)



<a name="3D-Generation"></a>

# 3D生成(3D Generation)


**Generative Gaussian Splatting for Unbounded 3D City Generation**

- Project: https://haozhexie.com/project/gaussian-city
- Paper: https://arxiv.org/abs/2406.06526
- Code: https://github.com/hzxie/GaussianCity

**StdGEN: Semantic-Decomposed 3D Character Generation from Single Images**

- Project: https://stdgen.github.io/
- Paper: https://arxiv.org/abs/2411.05738
- Code: https://github.com/hyz317/StdGEN


<a name="3D-Reconstruction"></a>

# 3D重建(3D Reconstruction)

**Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass**

- Project: https://fast3r-3d.github.io/
- Paper: https://arxiv.org/abs/2501.13928


<a name="HMG"></a>

# 人体运动生成(Human Motion Generation)

**SemGeoMo: Dynamic Contextual Human Motion Generation with Semantic and Geometric Guidance**

- Project: https://4dvlab.github.io/project_page/semgeomo/
- Paper: https://arxiv.org/abs/2503.01291
- https://github.com/4DVLab/SemGeoMo

<a name="Video-Understanding"></a>

# 视频理解(Video Understanding)

**Temporal Grounding Videos like Flipping Manga**

- Paper: https://arxiv.org/abs/2411.10332
- Code: https://github.com/yongliang-wu/NumPro

<a name="Embodied"></a>

# 具身智能(Embodied AI)

**Universal Actions for Enhanced Embodied Foundation Models**

- Project: https://2toinf.github.io/UniAct/
- Paper: https://arxiv.org/abs/2501.10105
- Code: https://github.com/2toinf/UniAct

**PhysVLM: Enabling Visual Language Models to Understand Robotic Physical Reachability**

- Paper: https://arxiv.org/abs/2503.08481
- Code: https://github.com/unira-zwj/PhysVLM


<a name="KD"></a>

# 知识蒸馏(Knowledge Distillation)

<a name="Depth-Estimation"></a>


# 深度估计(Depth Estimation)

**DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos**

- Project: https://depthcrafter.github.io
- Paper: https://arxiv.org/abs/2409.02095
- Code: https://github.com/Tencent/DepthCrafter


**MonSter: Marry Monodepth to Stereo Unleashes Power**

- Paper: https://arxiv.org/abs/2501.08643
- Code: https://github.com/Junda24/MonSter

**DEFOM-Stereo: Depth Foundation Model Based Stereo Matching**

- Project: https://insta360-research-team.github.io/DEFOM-Stereo/
- Paper: https://arxiv.org/abs/2501.09466
- Code: https://github.com/Insta360-Research-Team/DEFOM-Stereo


<a name="Stereo-Matching"></a>

# 立体匹配(Stereo Matching)

**MonSter: Marry Monodepth to Stereo Unleashes Power**

- Paper: https://arxiv.org/abs/2501.08643
- Code: https://github.com/Junda24/MonSter


<a name="Low-light"></a>

# 暗光图像增强(Low-light Image Enhancement)


**HVI: A New color space for Low-light Image Enhancement**

- Paper: https://arxiv.org/abs/2502.20272
- Code: https://github.com/Fediory/HVI-CIDNet
- Demo: https://huggingface.co/spaces/Fediory/HVI-CIDNet_Low-light-Image-Enhancement_

**ReDDiT: Efficient Diffusion as Low Light Enhancer**

- Paper: https://arxiv.org/abs/2410.12346
- Code: https://github.com/lgz-0713/ReDDiT



<a name="IC"></a>

# 图像压缩(Image Compression)](#IC)

**MambaIC: State Space Models for High-Performance Learned Image Compression**

- Paper: https://arxiv.org/abs/2503.12461
- Code: https://arxiv.org/abs/2503.12461


<a name="SGG"></a>

# 场景图生成(Scene Graph Generation)



<a name="ST"></a>

# 风格迁移(Style Transfer)

**StyleStudio: Text-Driven Style Transfer with Selective Control of Style Elements**

- Project: https://stylestudio-official.github.io/
- Paper: https://arxiv.org/abs/2412.08503
- Code: https://github.com/Westlake-AGI-Lab/StyleStudio


<a name="IQA"></a>

# 图像质量评价(Image Quality Assessment)

**Auto Cherry-Picker: Learning from High-quality Generative Data Driven by Language**

- Homepage: https://yichengchen24.github.io/projects/autocherrypicker
- Paper: https://arxiv.org/pdf/2406.20085
- Code: https://github.com/yichengchen24/ACP

<a name="Video-Quality-Assessment"></a>

# 视频质量评价(Video Quality Assessment)

<a name="CS"></a>

# 压缩感知(Compressive Sensing)

**Using Powerful Prior Knowledge of Diffusion Model in Deep Unfolding Networks for Image Compressive Sensing**

- Paper: https://arxiv.org/abs/2503.08429
- Code: https://github.com/FengodChen/DMP-DUN-CVPR2025


<a name="Datasets"></a>

# 数据集(Datasets)


**Objaverse++: Curated 3D Object Dataset with Quality Annotations**

- Paper: https://arxiv.org/abs/2504.07334
- Code: https://github.com/TCXX/ObjaversePlusPlus


<a name="Others"></a>

# 其他(Others)


**DTGBrepGen: A Novel B-rep Generative Model through Decoupling Topology and Geometry**

- Paper: https://arxiv.org/abs/2503.13110
- Code: https://github.com/jinli99/DTGBrepGen


**Analyzing the Synthetic-to-Real Domain Gap in 3D Hand Pose Estimation**

- Paper: https://arxiv.org/abs/2503.19307
- Code: https://github.com/delaprada/HandSynthesis.git

**EVOS: Efficient Implicit Neural Training via EVOlutionary Selector**

- Homepage: https://weixiang-zhang.github.io/proj-evos/
- Paper: https://arxiv.org/abs/2412.10153
- Code: https://github.com/zwx-open/EVOS-INR
  