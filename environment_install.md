<!--
 * @Author: PengJie pengjieb@mail.ustc.edu.cn
 * @Date: 2025-09-18 22:19:22
 * @LastEditors: PengJie pengjieb@mail.ustc.edu.cn
 * @LastEditTime: 2025-09-18 22:18:57
 * @FilePath: /imbalance_modality/environment_install.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->

```bash
conda create -n active_missing python=3.10 -y
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install einops opencv-python tqdm
```