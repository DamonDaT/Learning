#### 【说明】

***

开始接触微软的 DeepSpeed 框架，有些复杂，慢慢啃下



#### 【一】安装过程

***

> 参考 HuggingFace 给的例子：https://huggingface.co/docs/transformers/main/zh/main_classes/deepspeed

* 基本配置信息：
  * 显卡：NVIDIA GeForce RTX 4090（Ada 架构）
  * cuda：12.1（'sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90'）

* 源码安装 deepspeed

  ```shell
  git clone https://github.com/microsoft/DeepSpeed/
  cd DeepSpeed
  rm -rf build
  TORCH_CUDA_ARCH_LIST="8.9" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 \
  python setup.py build_ext -j8 bdist_wheel
  pip install dist/deepspeed-0.14.1+0529eac6-cp310-cp310-linux_x86_64.whl --proxy "http://127.0.0.1:7890"
  ```

* 源码安装 transformers（https://huggingface.co/docs/transformers/installation#install-from-source）

  ```shell
  pip install git+https://github.com/huggingface/transformers --proxy "http://127.0.0.1:7890"
  ```

