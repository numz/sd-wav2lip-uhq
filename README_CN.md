# 🔉👄 Stable Diffusion WebUI Automatic1111 Wav2Lip UHQ 扩展插件 

## <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></b></div>

![Illustration](https://user-images.githubusercontent.com/800903/258130805-26d9732f-4d33-4c7e-974e-7af2f1261768.gif)

https://user-images.githubusercontent.com/800903/258139382-6594f243-b43d-46b9-89f1-7a9f8f47b764.mp4

## 💡 简介
本代码仓库是适用于Automatic1111的 Wav2Lip UHQ扩展插件。

本插件为一体化集成解决方案：只需要一段视频和一段口播音频文件（wav或者mp3），就可以生成一个嘴唇同步的视频。通过Stable Diffusion特别的后处理技术，本插件所生成视频的嘴唇同步效果相比于[Wav2Lip tool](https://github.com/Rudrabha/Wav2Lip)所生成的视频，有更好的质量。

![Illustration](https://user-images.githubusercontent.com/800903/261026004-871ed08b-c5b2-4de2-9cb2-41928d584396.png)

## 📖 快速索引
* [🚀 更新](#-更新)
* [🔗 必要环境](#-必要环境)
* [💻 安装说明](#-安装说明)
* [🐍 使用方法](#-使用方法)
* [📖 后台原理](#-后台原理)
* [💪 提高质量的小提示](#-提高质量的小提示)
* [⚠️需要注意的约束](#-需要注意的约束)
* [📝 即将上线](#-即将上线)
* [😎 贡献](#-贡献)
* [🙏 鸣谢](#-鸣谢)
* [📝 引用](#-引用)
* [📜 版权声明](#-版权声明)

## 🚀 更新
**2023.08.17**
- 🐛 修复嘴唇发紫的bug 

**2023.08.16**
- ⚡ 除了generated版本的视频，额外输出Wav2lip和enhanced版本的视频，你可以从中选择效果更好的一个版本。
- 🚢 用户界面更新：增加了对CodeFormer Fidelity的控制说明。
- 👄 删除了图片输入方式,因为 [SadTalker](https://github.com/OpenTalker/SadTalker) 的方法更好。
- 🐛 修复了输入和输出视频之间的差异导致遮罩蒙板位置不正确的bug。
- 💪 改进了处理流程，提高了效率。
- 🚫 如果程序在处理过程中，中段还会继续产出视频。

**2023.08.13**
- ⚡ 加快计算速度。
- 🚢 用户界面更新：添加了一些隐藏参数的设置。
- 👄 提供了“仅追踪嘴巴”的选项。
- 📰 控制debug。
- 🐛 修复了resize factor的bug。


## 🔗 必要环境

- 最新版本的Stable Diffusion WebUI Automatic1111 [Stable Diffusion Webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 。

## 💻 安装说明

1. 启动Automatic1111
2. 在扩展菜单里, 找到“从网址安装”标签，输入下方URL地址，点击“安装”：

![Illustration](https://user-images.githubusercontent.com/800903/258115646-22b4b363-c363-4fc8-b316-c162b61b5d15.png)

3. 来到“已安装”标签，点击“应用并重启用户界面”.

![Illustration](https://user-images.githubusercontent.com/800903/258115651-196a07bd-ee4b-4aaf-b11e-8e2d1ffaa42f.png)

4. 如果您仍然看不到"Wav2Lip UHQ"的菜单，尝试重启Automatic1111.

5. 🔥 十分重要: 必须要下载模型。从下方表格下载全部所需的模型。（要注意模型文件名，确保文件名正确无误，尤其是s3fd模型）。

|        模型        |                                    描述                                     |                                                                        地址                                                                         |                                       安装目录                                       |
|:-------------------:|:----------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------:|
|       Wav2Lip       |                              高精度的唇同步                              |        [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9jtMfbNDaQ?e=TBFBVW)         |                   extensions\sd-wav2lip-uhq\scripts\wav2lip\checkpoints\                   |
|    Wav2Lip + GAN    |               嘴唇同步稍差，但视觉质量更好                |        [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW)         |                   extensions\sd-wav2lip-uhq\scripts\wav2lip\checkpoints\                   |
|        s3fd         |                          人脸检测预训练模型                          |                                           [Link](https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth)                                           |      extensions\sd-wav2lip-uhq\scripts\wav2lip\face_detection\detection\sfd\s3fd.pth       |
| landmark predicator |        Dlib 68点人脸特征推测 (点击下载按钮)         |                              [Link](https://github.com/numz/wav2lip_uhq/blob/main/predicator/shape_predictor_68_face_landmarks.dat)                              | extensions\sd-wav2lip-uhq\scripts\wav2lip\predicator\shape_predictor_68_face_landmarks.dat |
| landmark predicator |              Dlib 68点人脸特征推测 (备用地址1)               | [Link](https://huggingface.co/spaces/asdasdasdasd/Face-forgery-detection/resolve/ccfc24642e0210d4d885bc7b3dbc9a68ed948ad6/shape_predictor_68_face_landmarks.dat) | extensions\sd-wav2lip-uhq\scripts\wav2lip\predicator\shape_predictor_68_face_landmarks.dat |
| landmark predicator | Dlib 68点人脸特征推测 (备用地址2，点击下载按钮) |                        [Link](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat)                         | extensions\sd-wav2lip-uhq\scripts\wav2lip\predicator\shape_predictor_68_face_landmarks.dat |


## 🐍 使用方法
1. 上传一个包含人脸的视频文件（avi格式或者mp4格式均可）。如果视频里没有人脸，哪怕只有一帧不包含人脸，会导致处理失败。请注意，如果你上传的是avi文件，在界面上你看不见它，但不用担心，插件会正常处理视频。
2. 上传一个口播音频文件。
3. 选择模型 (详见上方表格).
4. **Padding（填充）**: Wav2Lip用黑色边缘填充嘴巴边缘，这有助于在一定程度上防止面部检测时破坏嘴唇边缘，有时候可能你需要适当调整该参数，但默认值通常效果还不错。
5. **No Smooth（不要平滑）**: 当勾选该选项，将会保持原始嘴部形状不做平滑处理。
6. **Resize Factor（调整大小）**: 该选项会对视频的分辨率进行调整。默认值是1，如果你需要降低你的视频分辨率，你可以调整它。
7. **Only Mouth（仅追踪嘴巴）**: 选中该选项，将仅对嘴部进行追踪，这将会移除例如脸颊和下巴的动作。
8. **Mouth Mask Dilate（嘴部遮罩蒙板扩张）**: 该选项用于调整嘴巴覆盖区域，参数越大，覆盖面积越大，根据嘴巴的大小来作出调整。
9. **Face Mask Erode（面部遮罩蒙板侵蚀）**: 对脸部外延区域进行渗透侵蚀处理，根据脸型大小作出调整。
10. **Mask Blur（遮罩模糊）**: 通过对遮罩层进行模糊处理，使其变得更平滑，建议尽量使该参数小于等于 **Mouth Mask Dilate（嘴部遮罩蒙板扩张）** 参数.
11. **Code Former Fidelity（Code Former保真度）**: 
    1. 当该参数偏向0时，虽然有更高的画质，但可能会引起人物外观特征改变，以及画面闪烁。
    2. 当该参数偏向1时，虽然降低了画质，但是能更大程度的保留原来人物的外观特征，以及降低画面闪烁。
    3. 不建议该参数低于0.5。为了达到良好的效果，建议在0.75左右进行调整。
12. **Active debug（启用debug模式）**: 开启该选项，将会在debug目录里逐步执行来生成图片。
13. 点击“Generate”（生成）按钮。

## 📖 后台原理

本扩展分几个流程运行，以此达到提高Wav2Lip生成的视频的质量的效果：

1. **Generate a Wav2lip video（生成Wav2lip视频）**: 该脚本先使用输入的视频和音频生成低质量的Wav2Lip视频。
2. **Mask Creation（创建遮罩蒙板）**: 该脚本在嘴巴周围制作了一个遮罩蒙板，并试图保持其他面部动作，比如脸颊和下巴的动作。
3. **Video Quality Enhancement（视频质量增强）**: 将低质量的Wav2Lip视频和嘴部遮罩覆盖到高质量原始视频上。
4. **Face Enhancer（面部增强修复）**: 脚本接下来会将原始图像和低质量嘴部动作传递给Stable Diffusion，由Stable Diffusion面部增强进行处理，以生成高质量嘴脸图像。
5. **Video Generation（生成视频）***: 该脚本会获取高质量的嘴巴图像，并将其覆盖在由嘴部遮罩引导的原始图像上。
6. **Video Post Processing（后期合成）**: 该脚本会调用ffmpeg，生成最终版本的视频。

## 💪 提高质量的小提示
- 使用高质量的视频作为输入源
- 使用常见FPS（譬如24fps、25fps、30fps、60fps）的视频，如果不是常见的FPS，偶尔会出现一些问题，譬如面部遮罩蒙板处理。
- 使用高质量的音频源文件，不要有音乐，不要有背景白噪声。使用类似 [https://podcast.adobe.com/enhance](https://podcast.adobe.com/enhance) 的工具清除背景音乐。
- 尽量减少面部纹理。譬如，在输入到Wav2lip之前，你可以使用图生图里的“面部修复”功能对面部进行一次修复。
- 扩大嘴部遮罩蒙板范围。这将有助于模型保留一些面部动作，并盖住原来的嘴巴。
- “遮罩模糊”（Mask Blur）的最大值是“嘴部遮罩蒙板扩张”（Mouth Mask Dilate）值的两倍。如果要增加模糊度，请增加“嘴部遮罩蒙板扩张”的值，否则嘴巴将变得模糊，并且可以看到下面的嘴巴。
- 高清放大有利于提高质量，尤其是在嘴巴周围。但是，它将使处理时间变长。你可以参考Olivio Sarikas的教程来高清放大处理你的视频: [https://www.youtube.com/watch?v=3z4MKUqFEUk](https://www.youtube.com/watch?v=3z4MKUqFEUk). 确保去噪强度设置在0.0和0.05之间，选择“revAnimated”模型，并使用批处理模式。
- 确保视频的每一帧上都有一张脸。如果未检测到人脸，插件将停止运行。


## ⚠ 需要注意的约束
- 目前的模型对胡须并不友好.
- 如果初始化阶段过长，请考虑使用调整“Resize Factor”来减小视频分辨率的大小。
- 虽然对原始视频没有严格的大小限制，但较大的视频需要更多的处理时间。建议使用“调整大小因子”来最小化视频大小，然后在处理完成后升级视频。

## 📝 即将上线
- [ ] 增加文字转语音功能，可直接可以通过该功能生成语音作为语音输入。Suno/Bark文字转语音功能引擎 (参见 [bark](https://github.com/suno-ai/bark/))  
- [ ] 考虑实现视频生成时的暂停/恢复功能
- [ ] 本插件在Automatic1111里，将更名为"Wav2Lip Studio"
- [ ] 增加更多的样例和说明演示
- [ ] 将avi转mp4（目前avi文件在输入框不显示，但依然能正常工作）

## 😎 贡献

我们欢迎各位对本项目的贡献提交。提交合并提取请求（Pull requests）时，请提供更改的详细说明。 详参 [CONTRIBUTING](CONTRIBUTING.md) 。

## 🙏 鸣谢 
- [Wav2Lip](https://github.com/Rudrabha/Wav2Lip)
- [CodeFormer](https://github.com/sczhou/CodeFormer)

## 📝 引用
如果您在工作、发表文章、教程或演示中使用到了项目，我们非常鼓励您引用此项目。

如需引证本项目，请考虑使用如下BibTeX格式：

```
@misc{wav2lip_uhq,
  author = {numz},
  title = {Wav2Lip UHQ},
  year = {2023},
  howpublished = {GitHub repository},
  publisher = {numz},
  url = {https://github.com/numz/sd-wav2lip-uhq}
}
``` 

## 📜 版权声明
* 此代码仓库中的代码是根据MIT许可协议发布的。 [LICENSE file](LICENSE).
