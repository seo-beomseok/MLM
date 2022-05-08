# Mixture of Linear Models(MLM) <br> Co-supervised by Deep Neural Network
<h4 class="author">Beomseok Seo</h4>
<h4 class="date">2022-05-01</h4>

- MLM explains DNN by approximating it with a piecewise linear model.
Arxiv preprint can be obtained <a href="https://arxiv.org/abs/2108.04035">here.</a>
(accepted by Journal of Computational and Graphical Statistics on 4/6/2022)

---

<h1 class="title toc-ignore">Install</h1>
            <pre><code>import mixturelinearmodel,
from mixturelinearmodel import MixtureLinearModel
from utils import plot_mosaic, plot_ci, explainable_tree, explainable_condition, explainable_dim, highest_explainable_dim, plot_id_1d, plot_id_2d, plot_id_3d
</code></pre>

<h1 class="title toc-ignore">Usage</h1>
<div>
<h2>1. Fit MLM-Cell</h2>
<pre><code>
MLM = MixtureLinearModel(dnn_model, verbose=True)
MLM.statsmodel = False
MLM.compute_CELL(X_train,K=100,random_seed=1)
'''
100%|██████████| 3/3 [03:02<00:00, 60.80s/it]
# of CELL:1712 / min size:1 / avg size:8.1 / max size:149 / # of singlton CELL:538
# fit MLM-cell
'''
            
MLM.fit_LocalModels(X_train, y_train, 
                    eps=0.01, num_noise_samp=100, 
                    classification=False, alpha=0, max_iter=10000,random_seed=1)
'''            
100%|██████████| 1712/1712 [00:50<00:00, 33.90it/s]
100%|██████████| 1712/1712 [00:00<00:00, 1726.29it/s]
'''
            
pred_lmm_train = MLM.predict(X_train, covariance_type='full', covariance_tied=True, uniform_prior=False,)
pred_lmm_test = MLM.predict(X_test, covariance_type='full', covariance_tied=True, uniform_prior=False,)
'''
100%|██████████| 1712/1712 [00:00<00:00, 15844.54it/s]
100%|██████████| 1712/1712 [00:05<00:00, 302.96it/s]
100%|██████████| 1712/1712 [00:00<00:00, 17756.26it/s]
100%|██████████| 1712/1712 [00:01<00:00, 1166.78it/s]
'''
            
print('MLM-CELL: Train RMSE:{:3.3f} / Test RMSE:{:3.3f}'.format(
            rmse(y_train,np.array(pred_lmm_train)),
            rmse(y_test,np.array(pred_lmm_test))))
MLM-CELL: Train RMSE:52.741 / Test RMSE:60.850
</code></pre>
</div>
<div>
<h2>2. Fit MLM-EPIC</h2>
<pre><code>
MLM.fit_MergedLocalModels(150, classification=False, alpha=0, max_iter=10000, random_seed=1)
'''
100%|██████████| 1712/1712 [04:20<00:00,  6.57it/s]
./src\mixturelinearmodel.py:392: ClusterWarning: scipy.cluster: The symmetric non-negative hollow observation matrix looks suspiciously like an uncondensed distance matrix
'''

LocalModelsTree = linkage(self.dist_mat_avg, 'ward')
'''
100%|██████████| 150/150 [00:00<00:00, 531.92it/s]
'''
            
pred_epic_train = MLM.predict(X_train,  merged=True, covariance_type='full', covariance_tied=True, uniform_prior=False)
pred_epic_test = MLM.predict(X_test, merged=True, covariance_type='full', covariance_tied=True, uniform_prior=False)
'''
100%|██████████| 1712/1712 [00:00<00:00, 15400.23it/s]
100%|██████████| 150/150 [00:05<00:00, 28.49it/s]
100%|██████████| 1712/1712 [00:00<00:00, 11435.41it/s]
100%|██████████| 150/150 [00:01<00:00, 91.79it/s]
'''
            
print('MLM-EPIC: Train RMSE:{:3.3f} / Test RMSE:{:3.3f}'.format(
            rmse(y_train,np.array(pred_epic_train)),
            rmse(y_test,np.array(pred_epic_test))))
MLM-EPIC: Train RMSE:62.777 / Test RMSE:66.684
MLM.save_dict('./output/mlm_bikesharing')
</code></pre>

<h1 class="title toc-ignore">Interpretation</h1>
 
<div id="step2" class="section level2">
<h2>1. Regression coefficients</h2>
<pre><code>            
import utils
from utils import plot_mosaic, plot_ci, explainable_tree, explainable_condition, explainable_dim, highest_explainable_dim, plot_id_1d, plot_id_2d, plot_id_3d

pred_epic_train = MLM.predict(X_train,  merged=True, covariance_type='full', covariance_tied=True, uniform_prior=False)
'''
100%|██████████| 1712/1712 [00:00<00:00, 19150.75it/s]
100%|██████████| 150/150 [00:03<00:00, 47.86it/s]
'''
            
MLM.feature_names = feature_names
plot_mosaic(MLM, epic_id=range(5), log_trans=False)

f, axs = plt.subplots(9,2,figsize=(10,15),dpi=300)
f.tight_layout()
for i in range(MLM.p+1):
    ax = plt.subplot(9,2,i+1)
    plot_ci(MLM,i,epic_id=range(5),ax=ax,title=True)            
</code></pre>
</div>
<div>            
<h2>2. Low Dimensional Subspace</h2>
<pre><code>   
explainable_tree(MLM, X_train, psi=0.8, epic_id=range(5))
'''            
EPIC 0(orig id 86): found tree with max pure size 727 and total pure size 883 for 883
EPIC 1(orig id 85): found tree with max pure size 377 and total pure size 488 for 488
EPIC 2(orig id 28): found tree with max pure size 166 and total pure size 456 for 456
EPIC 3(orig id 103): found tree with max pure size 182 and total pure size 370 for 370
EPIC 4(orig id 87): found tree with max pure size 66 and total pure size 307 for 307
'''
            
exp_cond = explainable_condition(MLM, X, xi=50, epic_id=range(5))
exp_cond
'''
[[array([['727', 'mnth > 3.5'],
         ['727', 'hr > 11.5'],
         ['727', 'hr <= 14.5'],
         ['727', 'workingday == 1']], dtype='<U21')],
 [array([['51', 'yr != 1'],
         ['51', 'mnth <= 2.5'],
         ['51', 'hr > 10.5'],
         ['51', 'hr <= 14.5'],
         ['51', 'workingday != 1'],
         ['51', 'season_0 != 1']], dtype='<U21'),
  array([['377', 'mnth <= 3.5'],
         ['377', 'hr <= 14.5'],
         ['377', 'hr > 9.5'],
         ['377', 'workingday == 1'],
         ['377', 'season_0 != 1']], dtype='<U21')],
 [array([['166', 'mnth > 3.5'],
         ['166', 'hr <= 21.5'],
         ['166', 'hr > 20.5'],
         ['166', 'workingday == 1'],
         ['166', 'season_2 != 1']], dtype='<U21')],
 [array([['88', 'hr <= 10.5'],
         ['88', 'hr > 9.5'],
         ['88', 'workingday == 1'],
         ['88', 'season_0 != 1'],
         ['88', 'season_1 == 1'],
         ['88', 'weathersit_1 != 1']], dtype='<U21'),
  array([['182', 'hr <= 11.5'],
         ['182', 'hr > 9.5'],
         ['182', 'workingday == 1'],
         ['182', 'season_0 == 1']], dtype='<U21')],
 [array([['66', 'yr != 1'],
         ['66', 'mnth > 3.5'],
         ['66', 'hr > 13.5'],
         ['66', 'hr <= 15.5'],
         ['66', 'holiday != 1'],
         ['66', 'workingday != 1'],
         ['66', 'temp > 0.35'],
         ['66', 'season_2 != 1']], dtype='<U21'),
  array([['59', 'hr <= 14.5'],
         ['59', 'hr > 11.5'],
         ['59', 'weekday <= 0.5'],
         ['59', 'workingday != 1'],
         ['59', 'season_2 == 1']], dtype='<U21')]]
'''
</code></pre>
</div>
<div>
<h2>3. Prominent Region</h2>         
<pre><code>
explainable_dim(MLM,X_train,epic_id=range(5),max_dim=p) 
</code></pre>
</div>
