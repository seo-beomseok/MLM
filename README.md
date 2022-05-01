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
<h2>1. Express DNN as a piecewise linear function</h2>
          <img src="files/img/layer_l-cell.png" height="230" />
          <img src="files/img/cells.png" height="350" />
          <img src="files/img/localsamples.gif" width="600" />
</div>          
<div id="step2" class="section level2">
<h2>2. Merge cells into EPIC</h2>
          <img src="files/img/epic.png" height="270" />
                    
</div>                    
<div id="step3" class="section level2">
<h2>3. Apply softweights to MLM</h2>
          <img src="files/img/softweights.png" width="320" />
