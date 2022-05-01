# Mixture of Linear Models(MLM) <br> Co-supervised by Deep Neural Network
<h4 class="author">Beomseok Seo</h4>
<h4 class="date">2022-05-01</h4>

MLM explains DNN by approximating it with a piecewise linear model.

<a href="https://arxiv.org/abs/2108.04035">Arxiv preprint can be obtained here.</a>


<h1 class="title toc-ignore">Overview</h1>
<p>One natural idea of explaining a complex function is to view the function as a composite of multiple simple functions in different segments of input space.
<br>&rarr; How to find the segments?
<br>&rarr; Using DNN as a proxy of the optimal prediction function.</p>
<img src="files/img/example.png" alt="A toy example" width="600" />


<h1 class="title toc-ignore">Construction of MLM</h1>
<img src="files/img/steps_3.gif" width="600" />

<div id="step1" class="section level2">
<h2>Step1. Express DNN as a piecewise linear function</h2>
          <p> Because of its activation functions, DNN can be easily approximated by a piecewise linear function. If ReLU is used, DNN is acutally a piecewise linear.  To express a DNN with a piecewise linear function, we first express a hidden layer of DNN as a piecewise linear function. A hidden layer can be easily expressed as a piecewise linear function by clustering its output. </p>
          <img src="files/img/relu1.png" height="230" /><br>
                    
          <p>Then, it is easy to find the piecewise linear expression of DNN by applying Cartesian product to the layer-level clusters.</p>
          <img src="files/img/layer_l-cell.png" height="230" /><br>
          <img src="files/img/cells.png" height="350" /><br>
          <img src="files/img/localsamples.gif" width="600" /><br>
</div>          
<div id="step2" class="section level2">
<h2>2. Merge cells into EPIC</h2>
          <img src="files/img/pairwise_prediction.png" width="600" /><br>
          <img src="files/img/epic.png" height="270" /><br>
                    
</div>                    
<div id="step3" class="section level2">
<h2>3. Apply softweights to MLM</h2>
          <img src="files/img/softweights.png" width="320" />
