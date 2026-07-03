# Good-Pickleball: AI 匹克球鹰眼系统 🏓

## 同系列项目

Good-Pickleball、Good-Tennis 和 Good-Badminton 是同一类计算机视觉运动视频分析项目，核心思路都围绕球员检测、球/球路追踪、球场坐标映射、轨迹统计和可视化输出展开，只是适配的球场模型、球检测目标和运动规则不同。

| 项目 | 方向 | Stars |
| --- | --- | --- |
| [Good-Pickleball](https://github.com/yo-WASSUP/Good-Pickleball) | 匹克球视频分析 | [![Good-Pickleball stars](https://img.shields.io/github/stars/yo-WASSUP/Good-Pickleball?style=social)](https://github.com/yo-WASSUP/Good-Pickleball/stargazers) |
| [Good-Tennis](https://github.com/yo-WASSUP/Good-Tennis) | 网球视频分析 | [![Good-Tennis stars](https://img.shields.io/github/stars/yo-WASSUP/Good-Tennis?style=social)](https://github.com/yo-WASSUP/Good-Tennis/stargazers) |
| [Good-Badminton](https://github.com/yo-WASSUP/Good-Badminton) | 羽毛球视频分析 | [![Good-Badminton stars](https://img.shields.io/github/stars/yo-WASSUP/Good-Badminton?style=social)](https://github.com/yo-WASSUP/Good-Badminton/stargazers) |

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/yo-WASSUP/Good-Pickleball?style=social)](https://github.com/yo-WASSUP/Good-Pickleball/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/yo-WASSUP/Good-Pickleball?style=social)](https://github.com/yo-WASSUP/Good-Pickleball/network/members)
[![GitHub license](https://img.shields.io/github/license/yo-WASSUP/Good-Pickleball)](https://github.com/yo-WASSUP/Good-Pickleball/blob/main/LICENSE)

**基于计算机视觉的匹克球比赛视频分析工具**

[中文](README.md) | [English](README_en.md)

</div>

### 🎬 视频分析结果

| YOLO26s 人体检测 | RTMPose 姿态检测 |
| --- | --- |
| ![YOLO26s 人体检测演示](assets/yolo26s_detect_demo.gif) | ![RTMPose 姿态检测演示](assets/rtmpose_detect_demo.gif) |

匹克球比赛远景里球员通常较小，目标检测一般比姿态估计更稳定。

## 📝 更新日志

- **2026-07-01**：从 Good-Tennis 迁移为匹克球版本，替换标准匹克球场地模型并整理中文 README。
- **当前版本**：支持球员检测、球检测、球场坐标映射、轨迹统计、回合检测、小地图、热力图/散点图和带标注视频输出。
- **实验功能**：自动球场外角点检测、匹克球检测和弹跳点检测仍在迭代中，适合研究和二次开发使用。

## 🗺️ 开发计划

- [x] 匹克球比赛视频逐帧分析
- [x] YOLO 人体检测和多姿态模型支持
- [x] YOLO 球检测模型接入
- [x] 手动/自动匹克球场标注与球场坐标映射
- [x] 球员移动轨迹、速度、距离和回合统计
- [x] 匹克球轨迹和弹跳点标注
- [x] 标准匹克球场小地图叠加
- [x] 中文 / 英文可视化文字
- [x] 热力图、散点图和检测数据导出
- [ ] 双打支持
- [ ] 更稳定的匹克球弹跳点识别
- [ ] 更精确的专用匹克球检测模型
- [ ] 更完整的击球点和技术动作统计
- [ ] 批量视频分析工作流

---

## ✨ 功能

- **球员检测** - 默认使用 YOLO 人体框检测，也可切换到 RTMPose、RTMO 或 Ultralytics YOLO Pose 姿态估计。
- **匹克球检测** - 默认模型是从 Good-Tennis 复制来的 `weights/tennis-ball.pt` 权重。并没有用匹克球的数据去训练，因为我看匹克球跟网球长得有点像，所以这个效果会比网球的差。
- **球场标注** - 默认尝试自动检测匹克球场四个外角点，失败后切换为手动点击四个外角点。
- **球场坐标映射** - 将图像坐标映射到标准匹克球场坐标，球场尺寸按 `6.096m x 13.4112m` 建模，非截击区距离球网 `2.1336m`。
- **球员位置追踪** - 记录球员球场坐标、移动轨迹、速度和距离。
- **回合检测** - 根据连续球场视图自动判断回合开始和结束，并在视频叠加层和检测数据中记录回合编号。
- **弹跳点检测** - 视频处理完成后，按整段球轨迹做离群点清理、插值、速度计算，默认使用规则评分；干净球轨迹和弹跳点会在主画面和小地图上显示。
- **小地图叠加** - 在输出视频中显示标准匹克球场小地图，标注球员、球和弹跳点位置。
- **位置图表** - 自动生成球员位置热力图和散点图。
- **中英文显示** - 可通过 `--language zh/en` 切换可视化文字。
- **本地运行** - 视频、模型和分析结果都保存在本地。

### 📊 球场与位置可视化

| 自动球场检测 | 球员位置热力图 | 球员位置散点图 |
| --- | --- | --- |
| ![自动球场检测](assets/auto.png) | ![球员位置热力图](assets/demo_heatmap.png) | ![球员位置散点图](assets/demo_scatter.png) |

## 🧩 系统要求

- Python 3.8+
- FFmpeg，并已加入系统 `PATH`
- OpenCV / PyTorch / Ultralytics / RTMLib / ONNX Runtime
- 推荐 NVIDIA GPU；CPU 可以运行，但视频分析速度会明显变慢

## ⚙️ 安装指南

### Windows

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### GPU 加速（Windows / NVIDIA）

默认依赖使用 CPU 版 PyTorch 和 ONNX Runtime。需要 GPU 加速时，先确认：

- 已安装 NVIDIA 显卡驱动，`nvidia-smi` 可以正常输出显卡信息。
- 推荐使用 CUDA 12.1 对应的 PyTorch wheel。
- 如果遇到 DLL 加载失败，先安装或修复 Microsoft Visual C++ Redistributable 2015-2022 x64。

PowerShell：

```bash
.\.venv\Scripts\activate

pip uninstall -y torch torchvision onnxruntime
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-gpu==1.20.1
```

切回 CPU 版：

```bash
pip install --force-reinstall -r requirements.txt
```

## 🧠 模型准备

首次运行前，请确认项目根目录下的 `weights/` 文件夹已有需要的权重文件：

```text
weights/tennis-ball.pt      # 从 Good-Tennis 复制来的球检测基线权重
weights/yolo26s.pt          # YOLO 人体检测模型
weights/yolo11s-pose.pt     # YOLO Pose 姿态模型
```

如果缺少默认权重，程序会在启动时提示对应文件不存在。可以从 Good-Tennis Release 页面下载基线权重，也可以把自己的模型放到 `weights/` 后通过 `--ball-model`、`--person-model`、`--yolo-pose-model` 指定路径。

```text
https://github.com/yo-WASSUP/Good-Tennis/releases/latest
```

默认球检测模型仍是网球检测权重，用来验证基线迁移效果；它不是专门训练过的匹克球模型。后续训练出专用匹克球检测模型后，建议替换 `weights/tennis-ball.pt` 或通过 `--ball-model` 指定新权重。

球员检测模型由 `--player-detector` 切换。默认 `yolo-person`，使用 YOLO 人体框检测，并取检测框底部中点作为球员位置。

本地 RTMPose / RTMO 文件不存在时，`rtmlib` 可能会尝试在线下载到用户缓存目录。

## 🚀 使用指南

### 第一次运行流程

1. 准备输入视频和对应的球场模板图，并确认 `weights/` 中已有模型权重。
2. 运行基础命令：

```bash
python main.py --video-path videos/demo.mp4 --template-path templates/demo.png
```

3. 程序会先尝试自动检测匹克球场四个外角点。
4. 检测到候选球场线时会显示预览窗口，并保存 `outputs/<视频文件名>/auto_court_preview.png` 供检查。
5. 按 `Enter`/`Y` 接受自动结果，按 `M`/`R`/`Esc` 切换到手动四角标注。
6. 手动标注时，按顺序点击左上、右上、右下、左下四个外角点。
7. 标注结果会保存到 `outputs/<视频文件名>/court_annotations.txt`。同一个输出目录下再次运行会复用这个文件。
8. 分析结束后，查看 `outputs/<视频文件名>/detect_<视频文件名>.mp4`、`detections.jsonl` 和 `position_visualizations/`。

如果换了视频视角、裁切方式或模板图，需要删除对应输出目录里的 `court_annotations.txt`，重新标注四点。

### 球员检测方式

默认使用 YOLO 人体框检测：

```bash
python main.py --video-path videos/demo.mp4 --template-path templates/demo.png --person-model weights/yolo26s.pt
```

启用 Ultralytics 内置多目标跟踪，减少球员框跨帧跳变：

```bash
python main.py --video-path videos/demo.mp4 --template-path templates/demo.png --person-tracker botsort
python main.py --video-path videos/demo.mp4 --template-path templates/demo.png --person-tracker bytetrack
```

跟踪器输出的 `track_id` 只作为球员框连续性的弱信号；球员身份仍以 `upper/lower` 半场和球场坐标连续性为准。

切换到姿态估计：

```bash
python main.py --video-path videos/demo.mp4 --template-path templates/demo.png --player-detector pose --pose-family rtmpose
```

使用 Ultralytics YOLO Pose：

```bash
python main.py --video-path videos/demo.mp4 --template-path templates/demo.png --player-detector pose --pose-family yolo-pose --yolo-pose-model weights/yolo11s-pose.pt
```

### 回合检测说明

程序会用球场模板图做比赛视图判断，并自动维护回合状态：

- 连续多帧匹配到球场视图时，判定新回合开始。
- 连续多帧没有匹配到球场视图时，判定当前回合结束。
- 回合编号会写入 `detections.jsonl`，并显示在输出视频的统计叠加层中。
- 每个回合开始时会重置该回合内的移动距离、速度等统计，整场统计继续累计。
- 这个逻辑依赖模板图和四点球场标注；如果模板图选得不准，回合切分也会不准。

### 常用参数

```text
--video-path                    输入视频路径，默认 videos/demo.mp4
--output-dir                    输出目录，默认 outputs/<视频文件名>
--ball-model                    YOLO 球检测模型路径，默认 weights/tennis-ball.pt
--pose-family                   姿态模型族：rtmpose、rtmo 或 yolo-pose
--pose-mode                     RTMPose / RTMO 档位：lightweight、balanced、performance
--yolo-pose-model               YOLO pose 模型路径或模型名，默认 weights/yolo11s-pose.pt
--player-detector               球员检测方式：yolo-person 或 pose，默认 yolo-person
--person-model                  YOLO 人体检测模型路径或模型名，默认 weights/yolo26s.pt
--person-tracker                YOLO 人体框跟踪器：none、botsort、bytetrack，默认 botsort
--player-detect-interval        球员检测间隔帧数，默认 2
--template-path                 球场模板图路径，默认 templates/demo.png
--court-detection               球场角点检测方式：manual、auto、auto-fallback，默认 auto-fallback
--pose-roi true|false           是否显示姿态检测 ROI 框，默认 true
--display true|false            是否显示 OpenCV 预览窗口，默认 true
--skeletons true|false          是否显示人体骨架，默认 true
--player-trajectories true|false 是否显示球员轨迹，默认 true
--court-trajectory true|false   是否显示球场轨迹叠加层，默认 true
--pickleball-trajectory true|false 是否显示匹克球轨迹，默认 true；旧参数 --tennis-ball-trajectory 仍兼容
--bounce-detection true|false   是否检测并标注匹克球弹跳点，默认 true
--bounce-classifier             可选弹跳分类器 pkl 路径；不传则使用规则评分
--mini-map true|false           是否显示球场小地图，默认 true
--player-stats true|false       是否显示球员统计信息，默认 true
--save-images                   保存处理后的每帧图像
--performance-stats             打印性能耗时
--visualize-positions true|false 是否生成热力图和散点图，默认 true
--audio true|false              是否保留原视频音频，默认 true
--language {zh,en}              选择界面语言，默认 en
```

## 📦 输出结果

默认输出到 `outputs/<视频文件名>/`：

- `metadata.json`：视频、模型、球场标注和输出文件元数据。
- `detections.jsonl`：逐帧检测记录，包含回合编号、球员、手部、球场坐标、速度、匹克球坐标和后处理弹跳点事件。
- `bounce_events.json`：整段轨迹后处理得到的弹跳点列表，包含帧号、图像坐标、球场坐标、置信度和诊断信息。
- `cleaned_ball_trajectory.json`：过滤和短缺失插值后的球轨迹，最终视频使用这份轨迹绘制。
- `detect_<视频文件名>.mp4`：带骨架、轨迹、统计信息、小地图和回合编号叠加层的输出视频。
- `court_annotations.txt`：球场标注坐标缓存。
- `auto_court_preview.png`：自动球场检测预览图，触发自动检测候选时生成。
- `position_visualizations/heatmaps/`：球员位置热力图。
- `position_visualizations/scatter_plots/`：球员位置散点图。
- `detect_images/`：使用 `--save-images` 时保存的逐帧图像。

## 🗂️ 项目结构

```text
main.py                    # 命令行入口和参数解析
requirements.txt           # 唯一依赖安装入口
pickleball_analysis/
├── system.py              # 视频分析主流程 PickleballAnalysisSystem
├── analysis/              # 弹跳点后处理
├── court/                 # 球场标注与坐标映射
├── data/                  # JSON / JSONL 输出
├── detection/             # 匹克球检测、球员检测和姿态检测
├── media/                 # 视频音频处理
├── tracking/              # 球员、球轨迹和回合追踪
└── visualization/         # 视频叠加层、统计图和位置图
```

## 🙏 致谢

Good-Pickleball 从 Good-Tennis 迁移而来，当前默认球检测权重也沿用 Good-Tennis 的基线模型。

感谢 RTMPose、RTMO 和 OpenMMLab 生态提供的姿态估计算法基础，以及 [Tau-J/rtmlib](https://github.com/Tau-J/rtmlib) 提供的轻量姿态估计运行库。

感谢 [Ultralytics](https://github.com/ultralytics/ultralytics) 提供的 YOLO 目标检测算法与工具链。

感谢 [yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet) 项目整理并公开网球数据集，为本项目的基线球检测与轨迹分析提供了重要参考。

## 许可证

本项目代码使用 Apache License 2.0。随项目使用的第三方模型权重许可证以其实际来源为准。
