<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h1 align="center">Face Recognition on Jetson Nano</h1>
</p>


<!-- TABLE OF CONTENTS -->
## Table of contents
- [About The Project](#about-the-project)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
- [Details](#details)
    - [Model](#model)
    - [Results](#results)
    - [Performance](#performance)
- [Challenges](#challenges)
- [License](#license)


<!-- ABOUT THE PROJECT -->
<h2 id="#about-the-project">About The Project</h2>
I deployment face recognition on jetson nano and build a database to load and
record the attendance of people. 

<!-- GETTING STARTED -->
<h2 id="#getting-started">Getting Started</h2>

To get a local copy up and running follow these simple steps.

<h3 id="#prerequisites"> Prerequisites </h3>

* [Use LXDE as your Desktop Environment](https://www.youtube.com/watch?v=9bACmWg0bvs&ab_channel=JetsonHacksJetsonHacks)
* [Extent swap partitions](https://www.jetsonhacks.com/2019/04/14/jetson-nano-use-more-memory/)
* Install [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)

<h3 id="#installation"> Installation </h3>

1. Clone the repo
   ```sh
   git clone https://github.com/phamvanhanh6720/Face-Recognition.git
   ```
2. Install requirements
   ```sh
   pip3 install -r requirments.txt
   ```



<!-- USAGE EXAMPLES -->
<h2 id="#details">Details</h2>
<h3 id="#model"> Model </h3>

* I use the [pre-trained model](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch#Model-Zoo) to extract embedding of a face image and
use cosine similarity between extracted embedding and other
stored embeddings in the system. After that, I choose a
threshold to identify the person.
  
<h3 id="#results"> Results </h3>

* My model has the ability to distinguish real
or fake faces based on 2D images. However, 
my model is sensitive when the user uses a color
image to fool the system.
* I have not evaluated my model on a large test set,
  but which achieves good accuracy on the small test set.
  
<h3 id="#performance"> Performance </h3>

* A full pipeline of process a frame contains face takes approximately 0.2s.
So my system achieves 5 FPS
<!-- ROADMAP -->

<h2 id="#challenges"> Challenges </h2>

* There are several challenges when you deploy on jetson nano:
    * Memory. If you run model with cuda, it takes so much memory
    * Installing few libraries is slow and hard
 
<!-- LICENSE -->
<h2 id="#license"> License </h2>

Distributed under the MIT License. See `LICENSE` for more information.






