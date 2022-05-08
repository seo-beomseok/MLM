# Mixture of Linear Models(MLM) <br> Co-supervised by Deep Neural Network
<h4 class="author">Beomseok Seo</h4>
<h4 class="date">2022-05-01</h4>

MLM explains DNN by approximating it with a piecewise linear model.
<a href="https://arxiv.org/abs/2108.04035">Arxiv preprint can be obtained here.</a>
(accepted by Journal of Computational and Graphical Statistics on 4/6/2022)

---

---

<h1 class="title toc-ignore">Overview</h1>
<p>One natural idea of explaining a complex function is to view the function as a composite of multiple simple functions in different segments of input space.
<br>&rarr; How to find the segments?
<br>&rarr; Using DNN as a proxy of the optimal prediction function.</p>
<p align="center">
          <img src="files/img/example.png" alt="A toy example" width="600" />
</p>

<h1 class="title toc-ignore">Construction of MLM</h1>
<p align="center">
          <img src="files/img/steps_3.gif" width="500" />
</p>
<div id="step1" class="section level2">
<h2>Step1. Express DNN as a piecewise linear function</h2>
          <p> Because of its activation functions, DNN can be easily approximated by a piecewise linear function. If ReLU is used, DNN is acutally a piecewise linear.</p><br>
          <p align="center">
                    <img src="files/img/relu1.png" height="200" />
          </p>
          <p>To express a DNN as a piecewise linear function, we first find a piecewise linear function approximating a hidden layer in the DNN. A hidden layer can be easily expressed as a piecewise linear function by clustering its output. We call the clusters as the <em>layer-level clutsers</em>.</p><br>
          <p align="center">
                    <img src="files/img/layer_l-cell.png" height="230" /><br>
          </p>
          <p>Then, it is easy to find the piecewise linear expression of the DNN by applying Cartesian product to the layer-level clusters.</p><br>
          <p align="center">
                    <img src="files/img/cells.png" height="300" /><br>
          </p>
          <p> For each local cluster, a linear model is fitted to find the local linear models. In this process, the original data points and simulated data points are used together so that the local linear models approximate the underlying DNN. Perturbed data points are generated around the mean of each cluster, and their target points are obtained by the predicted values of the underlying DNN.</p><br>
          <p align="center">
                    <img src="files/img/localsamples.gif" width="580" /><br>
          </p>
</div>          
<div id="step2" class="section level2">
<h2>2. Merge cells into EPIC</h2>
          <p> To reduce the number of local clusters, we merge them into the smaller number of clusters. We define <em>mutual prediction disparity</em>, $d_{s,t}$, between the pair of local linear models <img src="https://render.githubusercontent.com/render/math?math=m_s(\mathbf{x})"> and <img src="https://render.githubusercontent.com/render/math?math=m_t(\mathbf{x})">, <img src="https://render.githubusercontent.com/render/math?math=s, t \in \{1,...,\widetilde{K}\}"> by <br>
                    <img src="https://render.githubusercontent.com/render/math?math=d_{s,t} = \frac{1}{n_s+n_t+2m} \left[\sum\limits_{i=1}^{n_s+m} \Big(m_s(\mathbf{v}^\prime_{s,i}) - m_t(\mathbf{v}^\prime_{s,i})\Big)^2+\sum\limits_{i=1}^{n_t+m} \Big(m_s(\mathbf{v}^\prime_{t,i}) - m_t(\mathbf{v}^\prime_{t,i})\Big)^2\right ]  \, ."><br>
                    We can take $d_{s,t}$ as a distance measure between the two local linear models. It is the average squared difference between the predicted values by the two models.</p><br>
          <p align="center">
          <img src="files/img/pairwise_prediction.png" width="560" /><br>
          </p>
          <p> using hierarchical clustering</p>
          <p align="center">
          <img src="files/img/epic.png" height="270" /><br>
          </p>
                    
</div>                    
<div id="step3" class="section level2">
<h2>3. Apply softweights to MLM</h2>
          <img src="files/img/softweights.png" width="320" />
