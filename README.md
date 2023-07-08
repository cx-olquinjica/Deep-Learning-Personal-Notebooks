# Deep-Learning-Personal-Notebooks
This collection of notebooks is based on the [Dive into Deep Learning Book](https://d2l.ai/index.html). It was created with the intention of serving as a reference when working on future projects. All of the notes are written in [Pytorch](https://pytorch.org) and the [d2l library](https://github.com/d2l-ai/d2l-en/tree/master/d2l)

> Ralph Emerson Waldo:
"Nothing great was ever achieved without enthusiasm..." 
   
> Steven Wright: 
"Everywhere is walking distance if you have the time..."
   
> Lee Hanney:
"Exercise to stimulate, not to annihilate. The world wasn't formed in a day, and neither were we. set small goals and build upon them.."

> "The Man who loves walking will walk further than the man who loves the destination. When you fall in love with the journey, everything else takes care of itself. Trip, fall, pick yourself up. Get up, learn, do it over again..."


## Study Plan: 

1) Basics:  :white_check_mark:
    - Linear Neural Networks
    - Multilayer Perceptrons
    - Builder's guide
2) Convolutional Neural Networks   :white_check_mark:
    - LeNet ->> DenseNet
    - CNNs for Audio and Text (Maybe) 
3) Review Probability and Information Theory [Deep Learning: Adaptive Computation and Machine Learning Chapter III](hlsjlj) [another link](https://c.d2l.ai/berkeley-stat-157/units/probability.html) :white_check_mark:

        - Estimators, Bias and Variance 
        - Maximum Likelihood Estimation
        - Bayesian Statistics
        - Deep FeedForward Networks
4) [Deep Learning: Adaptive Computation and Machine Learning Chapter VII](dhhfh)  :white_check_mark:
    - Regularization for Deep Learning(apply to CNNs)
5) Optimization Algorithms [d2l.ai chapter 12](hfkh) :white_check_mark:
    - Companion: [Deep Learning: Adaptive Computation and Machine Learning Chapter VIII](dhhfh)
6) [Deep Learning: Adaptive Computation and Machine Learning Chapter IX](dhhfh) :white_check_mark:
    - Convolutional Neural Networks a Maths perspective
7) Computational Performance [d2l.ai chapter 13](hdhlh) :soon:
    - When talking about parallelization, do not forget to check the multiple implementation of GPU as shown on the AlexNet paper.
    - Implementation [cuda-convnet](https://code.google.com/archive/p/cuda-convnet/)
8) Computer Vision [d2l.ai chapter 14](hdoh) :white_check_mark:
9) Final Project :soon:
10) Recurrent Neural Networks [d2l.ai chapter 9-10](hdoh) :white_check_mark: 
    - Companion: [Deep Learning: Adaptive Computation and Machine Learning Chapter X](hodj)
11) Final Project :white_check_mark:
12) Attention Mechanisms and Transformers [d2l.ai chapter 11](hdoh) :white_check_mark:
13) Natural Language Processing: Pretraining [d2l.ai chapter 15](hdoh) :soon:
14) NLP: Applications [d2l.ai chapter 16](hdoh) :soon: 
    - RNN 
    - Transformers
15) Final Project (probably the biggest having into account how Large Language Models have become so popular), it will be divided in different sub-projects:
        - Machine Translation 
        - Document Summarization
        - Document Generation
    - Transformers in Computer Vision
        - Diffusion Models
        - Video Understanding
        - Nice time to reconsider GAN + Transformers!
16) __Hyperparameter Optimization__ [d2l.ai chapter 19](hdoh) :soon:

    Obs: The study of this topic can come after CNNs or before the CNNs final project, I want to start training large models from scratch, and I pretend to have on my arsenal all the most important trades of the field. The earlier I am exposed to them, the better. 
17) __Generative Adversarial Networks__ [d2l.ai chapter 20](hdoh)  :white_check_mark:
    - __This can come after the CNN final project__
    - Small project
    - [GANformer = GAN + Transformers](https://github.com/dorarad/gansformer)
    - [Video: NIPS 2016-Generative Adversarial Networks by Ian Goodfellow](https://www.youtube.com/watch?v=AJVyzd0rqdc)
18) __Recommender Systems__ [d2l.ai chapter 21](hdoh)
    - __This can come before the Transformer final project__
    - Small Project + Website
    - Checkout this paper LLMs + Recommender Systems [GPT4Rec: A Generative Framework for Personalized Recommendation and User Interests Interpretation](https://arxiv.org/pdf/2304.03879.pdf)
19) __Reinforcement Learning__ [d2l.ai chapter 17](hdoh)
    - Companion: [Course By David Silver from DeepMind](https://www.davidsilver.uk/teaching/)
    - Reinforcement Learning Recommender System: 
        - [Reinforcement Learning-based Recommender System](https://dl.acm.org/doi/pdf/10.1145/3543846)
        - [Off-Policy Actor-critic for Recommender Systems](https://dl.acm.org/doi/pdf/10.1145/3523227.3546758)
20) __Gaussian Processes__ [d2l.ai chapter 18](hdoh)
    - Why study Gaussian Processes??
        * They provide a function space perspective of modelling, which makes understanding a variety of model classes, including deep neural networks, much more approachable
        * They have an extraordinary range of applications where they are SOTA, including __active learning__, __hyperparameter learning__, __auto-ML__, and __spatiotemporal regression__
        * Over the last few years, algorithmic advances have made Gaussian processes increasingly scalable and relevant, harmonizing with deep learning through frameworks such as [GPyTorch](https://gpytorch.ai)

## Citation
```bibtex
@article{zhang2021dive,
    title={Dive into Deep Learning},
    author={Zhang, Aston and Lipton, Zachary C. and Li, Mu and Smola, Alexander J.},
    journal={arXiv preprint arXiv:2106.11342},
    year={2021}
}
```

## Important Note: 

Because of how relevant __Transformers__ have become in current machine learning research, the relative positions of the topics highlighted by __boldface__ notation will probably change, because I will dedicate a huge amount of time working with transformers, and don't intend to neglect these areas. 
