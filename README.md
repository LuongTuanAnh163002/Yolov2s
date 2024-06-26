<h1>YOLOV2: Build from scratch, Train YOLOV2 with Custom dataset</h1>
<div align="center" dir="auto">
<a href="https://github.com/LuongTuanAnh163002/Yolov2s/blob/main/LICENSE"><img src="https://camo.githubusercontent.com/00b6aa098f95cc8559f5f72a62f63261e44a1f09f0f560ca4c8ab25d4a631f05/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4c6963656e73652d4d49542d253343434f4c4f522533452e7376673f7374796c653d666f722d7468652d6261646765" alt="Generic badge" data-canonical-src="https://img.shields.io/badge/License-MIT-%3CCOLOR%3E.svg?style=for-the-badge" style="max-width: 100%;"></a>
<a href="https://pytorch.org/get-started/locally/" rel="nofollow"><img src="https://camo.githubusercontent.com/0add0c0b6ec6267b61016063796469feb03cc17c93d9f04201e25d0f12651de0/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5059544f5243482d312e31302b2d7265643f7374796c653d666f722d7468652d6261646765266c6f676f3d7079746f726368" alt="PyTorch - Version" data-canonical-src="https://img.shields.io/badge/PYTORCH-1.10+-red?style=for-the-badge&amp;logo=pytorch" style="max-width: 100%;"></a>
<a href="https://www.python.org/downloads/" rel="nofollow"><img src="https://camo.githubusercontent.com/c2623d41ae89703a8d56dab2e458028b95b87d8ce1897ff29930ef267e9e77e0/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f505954484f4e2d332e372b2d7265643f7374796c653d666f722d7468652d6261646765266c6f676f3d707974686f6e266c6f676f436f6c6f723d7768697465" alt="Python - Version" data-canonical-src="https://img.shields.io/badge/PYTHON-3.7+-red?style=for-the-badge&amp;logo=python&amp;logoColor=white" style="max-width: 100%;"></a>
<br></p>
</div>

<details open="">
  <summary>Table of Contents</summary>
  <ol dir="auto">
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#project-structure">Project Structure</a>
    </li>
    <li>
      <a href="#data-preparation">Data Preparation</a>
    </li>
    <li><a href="#custom-dataset">How to run repository with custom dataset</a></li>
    <li><a href="#colab">Try with example in google colab</a></li>
    <li><a href="#conclusion">Conclusion</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

<h2 tabindex="-1" id="user-content-about-the-project" dir="auto"><a class="heading-link" href="#about-the-project">About The Project<svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></h2>

<p dir="auto">In this project we will build YOLOV2 from scratch and training with all CUSTOM dataset</p>
<img width="100%" src="https://github.com/LuongTuanAnh163002/Yolov2s/blob/main/images/Yolov2_architechture.jpg" style="max-width: 100%;">
<h2 tabindex="-1" id="user-content-about-the-project" dir="auto"><a class="heading-link" href="#project-structure">Project Structure<svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></h2>
<div class="highlight highlight-source-shell notranslate position-relative overflow-auto" dir="auto">
  <pre>Yolov2s
  │   train.py                      <span class="pl-c"><span class="pl-c">#</span> Train script</span>
  │   detect.py                     <span class="pl-c"><span class="pl-c">#</span> Detect script inference</span>
  
  ├───model
  │       yolo.py               <span class="pl-c"><span class="pl-c">#</span>Define yolov2 model structure</span>
  │
  ├───data
  │       custom_dataset.yaml              <span class="pl-c"><span class="pl-c">#</span>Config data custom_dataset.yaml</span>
  │
  └───utils
      │   datasets.py               <span class="pl-c"><span class="pl-c">#</span>Processing datasets</span>
      │   metrics.py                <span class="pl-c"><span class="pl-c">#</span> Compute metrics</span>
      │   loss.py               <span class="pl-c"><span class="pl-c">#</span> Define loss function</span>
      │   general.py               <span class="pl-c"><span class="pl-c">#</span> Various helper functions</span>
      │   plots.py               <span class="pl-c"><span class="pl-c">#</span> Some plot function</span>
      
  </pre>

</div>
<h2 tabindex="-1" id="user-content-about-the-project" dir="auto"><a class="heading-link" href="#data-preparation">Data Preparation<svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></h2>

<pre>Yolov2s
└───datasets
    ├───images
    │   ├───train
          ├───file_name.jpg
          ├───..............
    │   └───valid
          ├───file_name.jpg
          ├───..............
        
    ├───labels
    │   ├───train
          ├───file_name.txt
          ├───.............
    │   └───valid
          ├───file_name.txt
          ├───.............
</pre>

<h2 tabindex="-1" id="user-content-about-the-project" dir="auto"><a class="heading-link" href="#custom-dataset">How to run repository with custom dataset<svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></h2>
  <h3>1.For training</h3>
  <p>+Step1: Install virtual environment, package</p>
  <pre>
  conda create --name yolo python=3.10.12
  git clone https://github.com/LuongTuanAnh163002/Yolov2s.git
  cd Yolov2s
  conda activate yolo
  pip install -r requirements.txt
  </pre>
  <p>+Step2: Dowload dataset</p>
  <pre>
  #for ubuntu/linux
  bash ./script/get_fruit.sh
  \
  #for window
  pip install gdown
  gdown 1btZfd9hFpY7J_UGDMHkUtia-2VggcLRP
  tar -xf fruit_dataset.zip
  del fruit_dataset.zip
  </pre>
  <p>+Step3: Go to "data" folder then create another file .yaml like custom_dataset.yaml</p>
  <p>+Step4: Run the command below to training for pretrain</p>
  <pre>python train.py --data data/file_name.yaml --epochs 20 --device [0, 1, 2..] --data_format yolo</pre>
  <p>After you run and done training, all results save in runs/train/exp/..., folder runs automatic create after training done:</p>

  <h3>2.For detect</h3>
  <pre>
  #for file
  python detect.py --source file_name.jpg --weight ../runs/train/../weights/__.pth --conf_thres 0.15 --device [0, 1, 2,..]
  #for folder
  python detect.py --source path_folder --weight ../runs/train/../weights/__.pth --conf_thres 0.15 --device [0, 1, 2,..]
  #for video
  python detect.py --source video.mp4 --weight ../runs/train/../weights/__.pth --conf_thres 0.15 --device [0, 1, 2,..]
  </pre>
  <h3>3.Launch tensorboard</h3>
  <pre>tensorboard --logdir ../runs/train/name_project --bind_all --port=2002</pre>

<h2 tabindex="-1" id="user-content-about-the-project" dir="auto"><a class="heading-link" href="#colab">Try with example in google colab<svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></h2>
<h3>1.For training with fruit dataset</h3>
<a href="https://colab.research.google.com/drive/16pBK6mbJOXZyMORBmEjHqHRe_cZGdbXN?usp=sharing" rel="nofollow"><img src="https://camo.githubusercontent.com/f5e0d0538a9c2972b5d413e0ace04cecd8efd828d133133933dfffec282a4e1b/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" style="max-width: 100%;"></a>

<h3>2.For inferrence with my model</h3>
<a href="https://colab.research.google.com/drive/1fMqkGKMdabVbd1-b2luW-hINI3A-4zJp?usp=sharing" rel="nofollow"><img src="https://camo.githubusercontent.com/f5e0d0538a9c2972b5d413e0ace04cecd8efd828d133133933dfffec282a4e1b/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" style="max-width: 100%;"></a>
<p>You can dowload weight here:</p>
<a href="https://drive.google.com/file/d/16EBb0VG1coVHHuxou5FHSPenFkquuXbc/view?usp=sharing"><code>last_tank.pth</code></a>
<a href="https://drive.google.com/file/d/1Rm8hyzPmcRicI4wgxC3w4WUN8D79oMjk/view?usp=sharing"><code>last_fruit.pth</code></a>

<h2 tabindex="-1" id="user-content-about-the-project" dir="auto"><a class="heading-link" href="#conclusion">Conclusion<svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></h2>

<p>We build complete yolov2 from scratch but we have some advantage and disadvantage:</p>
<p>Advantage</p>
<ul dir="auto">
<li>Simple</li>
<li>Speed</li>
<li>Train with many other dataset</li>
</ul>

<p>Disadvantage</p>
<ul dir="auto">
<li>Can only train with small dataset, if the amount of data is large, the data processing speed will be slow and model training will also take more time </li>
<li>Only jpg files images are supported during training, in the future we will improve to support more file types images.</li>
<li>Haven't exported model to onnx or tensorRT yet. In the near future we will update the conversion code for onnx and tensorRT.</li>
</ul>

<h2 tabindex="-1" id="user-content-about-the-project" dir="auto"><a class="heading-link" href="#license">License<svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></h2>
<p dir="auto">See <code>LICENSE</code> for more information.</p>

<h2 tabindex="-1" id="user-content-about-the-project" dir="auto"><a class="heading-link" href="#acknowledgements">Acknowledgements<svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></h2>

<ul dir="auto">
<li><a href="https://github.com/WongKinYiu/yolov7.git">https://github.com/WongKinYiu/yolov7.git</a></li>
<li><a href="https://arxiv.org/abs/1612.08242v1">https://arxiv.org/abs/1612.08242v1</a></li>
<li><a href="https://viblo.asia/p/yolo-series-p2-build-yolo-from-scratch-924lJGoz5PM">https://viblo.asia/p/yolo-series-p2-build-yolo-from-scratch-924lJGoz5PM</a></li>
</ul>
