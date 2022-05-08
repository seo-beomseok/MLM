import tensorflow as tf
import numpy as np
import pickle

from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.mixture import GaussianMixture

from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import cpu_count
from tqdm import tqdm
from warnings import simplefilter

from scipy.cluster.hierarchy import linkage, fcluster, ClusterWarning
from scipy.stats import multivariate_normal

import statsmodels.api as sm

class singleGaussModel:
    def __init__(self, covariance_type='full', dtype_=None):
        self.covariance_type = covariance_type
        self.dtype_ = dtype_
        
    def fit(self, x):
        self.p_ = x.shape[1]
        self.n_ = x.shape[0]
        self.means_ = np.sum(x, axis=0)/self.n_
        if self.covariance_type == 'diag':
            self.covariances_ = np.mean((x-self.means_)**2, axis=0, dtype=self.dtype_)
        elif self.covariance_type == 'spherical':
            self.covariances_ = np.repeat(np.mean((x-self.means_)**2, dtype=self.dtype_),self.p_)
        else:
            self.covariances_ = np.matmul((x-self.means_).T, (x-self.means_), dtype=self.dtype_)/self.n_

class marginalGauss:
    def __init__(self, gmm):
        self.mar_mean = gmm.means_

        if len(gmm.covariances_.shape)==1:
            self.mar_cov = np.diag(gmm.covariances_)
        else:
            self.mar_cov = gmm.covariances_

    def pdf(self, x):
        return multivariate_normal(self.mar_mean,self.mar_cov, allow_singular=True).pdf(x)

    def logpdf(self, x):
        return multivariate_normal(self.mar_mean,self.mar_cov, allow_singular=True).logpdf(x)
    
class marginalGaussMix:
    def __init__(self, gmm_list, clssize, dim=None):
        self.gmm_list = gmm_list
        self.clssize = clssize
        
        if dim is None:
            dim = range(len(gmm_list[0].means_))

        self.numgmm = len(gmm_list)
        self.mar_mean = []
        self.mar_cov = []
        self.mvn = []

        for i in range(self.numgmm):
            self.mar_mean.append(gmm_list[i].means_[dim])
            if len(gmm_list[i].covariances_.shape)==1:
                self.mar_cov.append(np.diag(gmm_list[i].covariances_)[np.ix_(dim,dim)])
            else:
                self.mar_cov.append(gmm_list[i].covariances_[np.ix_(dim,dim)])
            self.mvn.append(multivariate_normal(self.mar_mean[i],self.mar_cov[i], allow_singular=True))
        
    def pdf(self, x):
        prob = 0
        self.pi = self.clssize/np.sum(self.clssize)
        for i in range(self.numgmm):
            prob += self.pi[i]*self.mvn[i].pdf(x)
        return prob

    def logpdf(self, x):
        logprob = np.zeros((self.numgmm,x.shape[0]))
        self.pi = self.clssize/np.sum(self.clssize)
        for i in range(self.numgmm):
            logprob[i] = self.mvn[i].logpdf(x)
        return self.pi, logprob

#######################
# Main module for MLM #
#######################

class MixtureLinearModel:
    def __init__(self, nn_model, verbose=True):
        self.nn_model = nn_model
        self.numlayers = len(nn_model.layers)-1
        self.verbose = verbose
    
    # dissection of dnn models ------------------------------

    def SubModel(self, main_model, starting_layer_ix, ending_layer_ix):
        new_model = tf.keras.models.Sequential()
        input_layer = tf.keras.layers.InputLayer(input_shape=main_model.get_layer(index=starting_layer_ix-1).output.shape[1:])
        new_model.add(input_layer)
        for ix in range(starting_layer_ix, ending_layer_ix):
            curr_layer = main_model.get_layer(index=ix)
            new_model.add(curr_layer)
        return(new_model)
    
    def compute_CELL(self, X_train, K, verbose=None, random_seed=None):
        # num of layer l-cell
        self.K = K
        # sample size
        self.N = X_train.shape[0]
        # original dimension size
        self.p_orig = X_train.shape[1:]
        # flattened dimension size
        self.p = np.product(X_train.shape[1:])

        if verbose != None:
            self.verbose = verbose
        
        submodels = []
        submodels.append(tf.keras.models.Model(inputs=self.nn_model.input, outputs=self.nn_model.layers[0].output))
        for i in range(1,self.numlayers):
            submodels.append(self.SubModel(self.nn_model,i,i+1))
        
        extU = []    
        extX = []
        extU.append(submodels[0].predict(X_train))
        extX.append(extU[0].reshape((self.N,-1)))
        for i in range(1,len(submodels)):
            extU.append(submodels[i].predict(extU[i-1]))
            extX.append(extU[i].reshape(self.N,-1))

        extgmm = []
        if self.verbose:
            tasks = tqdm(range(len(submodels)), total=len(submodels), position=0, leave=True)
        else:
            tasks = range(len(submodels))

        for i in tasks:
            extgmm.append(GaussianMixture(n_components=self.K, random_state=random_seed).fit(extX[i]))
        
        # state sequence
        self.extcls = np.concatenate([[extgmm[i].predict(extX[i])] for i in range(len(extgmm))],axis=0).T

        # state sequence / semi_cls id / semi_cls size
        self.CELL =np.unique(self.extcls,axis=0,return_inverse=True,return_counts=True)

        # num of semi_cls
        self.Ktilde = self.CELL[0].shape[0]
        
        # semi_cls id
        self.CELL_id = self.CELL[1]

        # semi_cls size
        self.CELL_size = self.CELL[2]
        
        if self.verbose: print('# of CELL:{} / min size:{} / avg size:{:3.1f} / max size:{} / # of singlton CELL:{}'.format(self.Ktilde, self.CELL_size.min(), self.CELL_size.mean(), self.CELL_size.max(), np.sum(self.CELL_size==1)))
        
    # perturbed samples ------------------------------

    def compute_augmented_samp(self, X_train, y_train,
                              eps=0.001, num_noise_samp=100, classification=False, random_seed=None):
        self.eps = eps
        self.num_noise_samp = num_noise_samp
        
        X_train_ = np.reshape(X_train, (X_train.shape[0],-1))
        self.CELL_center = np.array([np.mean(X_train_[self.CELL_id==i],0) for i in range(self.Ktilde)])

        X_local = [X_train_[self.CELL_id==i] for i in range(self.Ktilde)]
        y_local = [y_train[self.CELL_id==i] for i in range(self.Ktilde)]
        
        np.random.RandomState(random_seed)
        rand = np.random.randint(100, size=self.Ktilde)

        X_perturb = np.zeros((self.Ktilde,num_noise_samp,self.p))
        y_perturb = np.zeros((self.Ktilde,num_noise_samp,1))
        
        cov = np.diag(np.zeros(self.p)+eps)

        def chol_sample(mean, cov, size=1):
            cholesky_cov = np.linalg.cholesky(cov)
            random_samp = np.array([mean + cholesky_cov @ np.random.standard_normal(mean.size) for i in range(size)])
            return random_samp

        if classification == False:
            def predict_perturb(i):
                np.random.seed(rand[i])
                X_perturb[i] = chol_sample(self.CELL_center[i],cov,num_noise_samp)
                y_perturb[i] = np.array(self.nn_model.predict(np.reshape(X_perturb[i], np.concatenate(((num_noise_samp,),self.p_orig)))))
        elif classification == True:
            def predict_perturb(i):
                np.random.seed(rand[i])
                X_perturb[i] = chol_sample(self.CELL_center[i],cov,num_noise_samp)
                y_perturb[i] = np.round(self.nn_model.predict(np.reshape(X_perturb[i], np.concatenate(((num_noise_samp,),self.p_orig)))))


        if self.verbose:
            tasks = tqdm(range(self.Ktilde), total=self.Ktilde, position=0, leave=True)
        else:
            tasks = range(self.Ktilde)

        for i in tasks:
            predict_perturb(i)

        
        self.X_augmented = []
        self.y_augmented = []
        for i in range(self.Ktilde):
            self.X_augmented.append(np.concatenate([X_perturb[i], X_local[i]], axis=0))
            self.y_augmented.append(np.concatenate([y_perturb[i].squeeze(), y_local[i]], axis=0))
        
        self.X_augmented = np.array(self.X_augmented, dtype=object)
        self.y_augmented = np.array(self.y_augmented, dtype=object)
        
    # train ------------------------------

    def fit_LocalModels(self, X_train, y_train, 
                        eps=0.001, num_noise_samp=100, 
                        classification=False, alpha=0, max_iter=1000,
                        verbose=None, random_seed=None,
                        compute_augmented_samp=True,
                        statsmodels=None,
                        method='default',
                        **kwargs):

        self.classification = classification
        self.alpha = alpha
        self.max_iter = max_iter

        if statsmodels == None:
            if 'statsmodels' not in self.__dict__.keys():
                self.statsmodels = True
        else:
            self.statsmodels = statsmodels

        if method == 'default':
            if classification == True:
                if alpha>0:
                    self.method = 'l1'
                else:
                    self.method = 'lbfgs'
            else:
                if alpha>0:
                    self.method = 'elastic_net'
                else:
                    self.method = 'pinv'
        else:
            self.method = method

        if verbose != None:
            self.verbose = verbose

        np.random.seed(random_seed)
        if compute_augmented_samp:
            self.compute_augmented_samp(X_train, y_train, eps, num_noise_samp, self.classification, random_seed)

        self.coef_CELL = np.zeros((self.Ktilde, self.p))
        self.intercept_CELL = np.zeros((self.Ktilde,1))
        self.local_models_CELL = []


        if self.verbose:
            tasks = tqdm(range(self.Ktilde), total=self.Ktilde, position=0, leave=True)
        else:
            tasks = range(self.Ktilde)

        if self.statsmodels:
            for i in tasks:
                if len(np.unique(self.y_augmented[i]))==1:
                    LocalLinearModel = sm.OLS(self.y_augmented[i],sm.add_constant(self.X_augmented[i])).fit(**kwargs) #LinearRegression()
                    LocalLinearModel.params[0] = self.y_augmented[i][0]
                    LocalLinearModel.params[1:] = np.zeros((self.p,))
                    self.coef_CELL[i] = LocalLinearModel.params[1:]                
                    self.intercept_CELL[i] = LocalLinearModel.params[0]
                    self.local_models_CELL.append(LocalLinearModel)
                    continue
                    
                if self.classification == True:
                    self.y_augmented[i] = np.round(self.y_augmented[i])
                    if self.alpha > 0:
                        LocalLinearModel = sm.Logit(self.y_augmented[i],sm.add_constant(self.X_augmented[i])).fit_regularized(method=self.method,alpha=self.alpha,maxiter=self.max_iter,disp=self.verbose,**kwargs)
                    else:
                        LocalLinearModel = sm.Logit(self.y_augmented[i],sm.add_constant(self.X_augmented[i])).fit(method=self.method,maxiter=self.max_iter,disp=self.verbose,**kwargs)
                elif self.alpha > 0:
                    LocalLinearModel = sm.OLS(self.y_augmented[i],sm.add_constant(self.X_augmented[i])).fit_regularized(method=self.method,L1_wt=1,alpha=self.alpha,maxiter=self.max_iter,**kwargs)
                else:
                    LocalLinearModel = sm.OLS(self.y_augmented[i],sm.add_constant(self.X_augmented[i])).fit(method=self.method,**kwargs)
            
                self.coef_CELL[i] = LocalLinearModel.params[1:]
                self.intercept_CELL[i] = LocalLinearModel.params[0]
                self.local_models_CELL.append(LocalLinearModel)
        else:
            for i in tasks:
                if len(np.unique(self.y_augmented[i]))==1:
                    LocalLinearModel = LinearRegression(**kwargs)
                    LocalLinearModel.intercept_ = self.y_augmented[i][0]
                    LocalLinearModel.coef_ = np.zeros((self.p,))
                    self.coef_CELL[i] = LocalLinearModel.coef_                
                    self.intercept_CELL[i] = LocalLinearModel.intercept_
                    self.local_models_CELL.append(LocalLinearModel)
                    continue
                    
                if self.classification == True:
                    self.y_augmented[i] = np.round(self.y_augmented[i])
                    if self.alpha > 0:
                        LocalLinearModel = LogisticRegression(max_iter=self.max_iter, solver='saga', penalty='l1', C=1/(alpha+1), **kwargs)
                    else:
                        LocalLinearModel = LogisticRegression(max_iter=self.max_iter, **kwargs)
                elif self.alpha > 0:
                    LocalLinearModel = Lasso(max_iter=self.max_iter,alpha=self.alpha, **kwargs)
                else:
                    LocalLinearModel = LinearRegression(**kwargs)

                LocalLinearModel.fit(self.X_augmented[i],self.y_augmented[i])

                self.coef_CELL[i] = LocalLinearModel.coef_
                self.intercept_CELL[i] = LocalLinearModel.intercept_
                self.local_models_CELL.append(LocalLinearModel)
    

    def compute_LocalModelsDist(self):
        dist_mat = np.zeros((self.Ktilde,self.Ktilde))
    
        if self.verbose:
            tasks = tqdm(range(self.Ktilde), total=self.Ktilde, position=0, leave=True)
        else:
            tasks = range(self.Ktilde)


        for i in tasks:
            if self.statsmodels:
                pred_i = self.local_models_CELL[i].predict(sm.add_constant(self.X_augmented[i]))
            else:
                pred_i = self.local_models_CELL[i].predict(self.X_augmented[i])
            for j in range(self.Ktilde):
                if i==j:
                    continue
                if self.statsmodels:
                    pred_j = self.local_models_CELL[j].predict(sm.add_constant(self.X_augmented[i]))
                else:
                    pred_j = self.local_models_CELL[j].predict(self.X_augmented[i])
                dist_mat[i,j] = np.mean(np.square(pred_i - pred_j))
                
        self.dist_mat_avg = (dist_mat+np.transpose(dist_mat))/2
        

    def fit_MergedLocalModels(self, Jtilde, dist_mat_avg=None,
                              classification=None, alpha=None,
                              max_iter=None, verbose=None, random_seed=None,
                              statsmodels=None,
                              method='default',
                              **kwargs):

        if classification != None:
            self.classification = classification
        if alpha != None:
            self.alpha = alpha
        if max_iter != None:
            self.max_iter = max_iter
        if verbose != None:
            self.verbose = verbose

        if statsmodels == None:
            if 'statsmodels' not in self.__dict__.keys():
                self.statsmodels = True
        else:
            self.statsmodels = statsmodels

        if method == 'default':
            if classification == True:
                if alpha>0:
                    self.method = 'l1'
                else:
                    self.method = 'lbfgs'
            else:
                if alpha>0:
                    self.method = 'elastic_net'
                else:
                    self.method = 'pinv'
        else:
            self.method = method

        if not self.verbose: simplefilter("ignore", ClusterWarning)

        np.random.seed(random_seed)

        self.Jtilde = Jtilde
        
        if dist_mat_avg is None:
            self.compute_LocalModelsDist()
        else:
            self.dist_mat_avg = dist_mat_avg

        LocalModelsTree = linkage(self.dist_mat_avg, 'ward')
        self.merge_map = fcluster(LocalModelsTree, Jtilde, criterion='maxclust')-1
        self.Jtilde = Jtilde = len(np.unique(self.merge_map))
        
        self.EPIC_id = self.merge_map[self.CELL_id]
        self.EPIC_size = np.array([np.sum(self.CELL_size[self.merge_map==i]) for i in range(Jtilde)])
        
        self.coef_EPIC = np.zeros((Jtilde,self.p))
        self.intercept_EPIC = np.zeros((Jtilde,1))
        self.local_models_EPIC = []
    
        sorted_merge_map = np.sort(np.unique(self.merge_map))

        if self.verbose:
            tasks = tqdm(range(len(sorted_merge_map)), total=len(sorted_merge_map), position=0, leave=True)
        else:
            tasks = range(len(sorted_merge_map))

        if self.statsmodels:
            for i in tasks:
                j = sorted_merge_map[i]
                X_merge = self.X_augmented[np.where(self.merge_map==j)]
                X_merge = np.concatenate(X_merge, axis=0)
                y_merge = self.y_augmented[np.where(self.merge_map==j)]
                y_merge = np.concatenate(y_merge, axis=0)    
            
    
                if self.classification == True:
                    if self.alpha > 0:
                        LocalLinearModel = sm.Logit(y_merge,sm.add_constant(X_merge)).fit_regularized(method=self.method,alpha=self.alpha,maxiter=self.max_iter,disp=self.verbose,**kwargs)
                    else:
                        LocalLinearModel = sm.Logit(y_merge,sm.add_constant(X_merge)).fit(method=self.method,maxiter=self.max_iter,disp=self.verbose,**kwargs)
                elif self.alpha > 0:
                    LocalLinearModel = sm.OLS(y_merge,sm.add_constant(X_merge)).fit_regularized(method=self.method,L1_wt=1,alpha=self.alpha,maxiter=self.max_iter,**kwargs)
                else:
                    LocalLinearModel = sm.OLS(y_merge,sm.add_constant(X_merge)).fit(method=self.method,**kwargs)

                self.coef_EPIC[i] = LocalLinearModel.params[1:]
                self.intercept_EPIC[i] = LocalLinearModel.params[0]
                self.local_models_EPIC.append(LocalLinearModel)
        else:
            for i in tasks:
                j = sorted_merge_map[i]
                X_merge = self.X_augmented[np.where(self.merge_map==j)]
                X_merge = np.concatenate(X_merge, axis=0)
                y_merge = self.y_augmented[np.where(self.merge_map==j)]
                y_merge = np.concatenate(y_merge, axis=0)    

                if self.classification == True:
                    if self.alpha > 0:
                        LocalLinearModel = LogisticRegression(max_iter=self.max_iter, solver='saga', penalty='l1', C=1/(alpha+1), **kwargs)
                    else:
                        LocalLinearModel = LogisticRegression(max_iter=self.max_iter, **kwargs)
                elif self.alpha > 0:
                    LocalLinearModel = Lasso(max_iter=self.max_iter,alpha=self.alpha, **kwargs)
                else:
                    LocalLinearModel = LinearRegression(**kwargs)

                LocalLinearModel.fit(X_merge,y_merge)

                self.coef_EPIC[i] = LocalLinearModel.coef_
                self.intercept_EPIC[i] = LocalLinearModel.intercept_
                self.local_models_EPIC.append(LocalLinearModel)

    
    def compute_GMM(self, X_augmented, covariance_type, dtype_):
        gmm = np.empty(len(X_augmented), dtype=object)
        gmm_cov = np.empty(len(X_augmented), dtype=object)

        def compute_single_gmm(i):
            singlegm = singleGaussModel(covariance_type, dtype_)
            singlegm.fit(X_augmented[i])
            gmm[i] = singlegm
            gmm_cov[i] = singlegm.covariances_

        num_cores = cpu_count()
        pool = Pool(processes=num_cores)

        if self.verbose:
            tasks = tqdm(pool.imap(compute_single_gmm, range(len(X_augmented))), total=len(X_augmented), position=0, leave=True)
        else:
            tasks = pool.imap(compute_single_gmm, range(len(X_augmented)))

        for _ in tasks:
            pass

        return gmm, gmm_cov            


    def compute_posterior(self, X, CELL_size=None, EPIC_size=None, merged=False,
                          covariance_type='full', covariance_tied=False, uniform_prior=False,
                          gmm=None,gmm_cov=None, dtype_=np.float64):
        if gmm is None:
            self.gmm, self.gmm_cov = self.compute_GMM(self.X_augmented,covariance_type,dtype_)
        else:
            self.gmm = gmm
            self.gmm_cov = gmm_cov

        if covariance_tied == True:
            tied_cov = np.sum([self.gmm_cov[i]*CELL_size[i] for i in range(len(CELL_size))],axis=0)/np.sum(CELL_size)
            for i in range(self.Ktilde):
                self.gmm[i].covariances_ = tied_cov
        
        num_cores = cpu_count()
        pool = Pool(processes=num_cores)

        if merged == False:
            prior = CELL_size/np.sum(CELL_size)
            post = np.zeros((self.Ktilde,X.shape[0]))

            def compute_marginal_gmm(i):
                logprob = marginalGauss(self.gmm[i]).logpdf(X)
                post[i] = logprob

            if self.verbose:
                tasks = tqdm(pool.imap(compute_marginal_gmm, range(self.Ktilde)), total=self.Ktilde, position=0, leave=True)
            else:
                tasks = pool.imap(compute_marginal_gmm, range(self.Ktilde))

            for _ in tasks:
                pass
            
            post = np.exp(post-np.max(post,axis=0))
        elif merged == True:
            prior = EPIC_size/np.sum(EPIC_size)
            post = np.zeros((self.Jtilde,X.shape[0]))
            post_ = []
            pi_ = []
            for i in range(self.Jtilde):
                numgmm = np.sum(self.merge_map==i)
                post_.append(np.zeros((numgmm,X.shape[0])))
                pi_.append(np.zeros((numgmm,)))

            def compute_marginal_gmm_mix(i):
                pi, logprob = marginalGaussMix(self.gmm[self.merge_map==i],CELL_size[self.merge_map==i]).logpdf(X)
                post_[i] = logprob
                pi_[i] = pi

            if self.verbose:
                tasks = tqdm(pool.imap(compute_marginal_gmm_mix, range(self.Jtilde)), total=self.Jtilde, position=0, leave=True)
            else:
                tasks = pool.imap(compute_marginal_gmm_mix, range(self.Jtilde))

            for _ in tasks:
                pass

            submax = np.zeros((self.Jtilde,X.shape[0]))
            for i in range(self.Jtilde):
                submax[i] = np.max(post_[i],axis=0)
            postmax = np.max(submax,axis=0)

            for i in range(self.Jtilde):
                post_[i] = np.exp(post_[i]-postmax)
                post[i] = np.matmul(pi_[i],post_[i])

        post = np.transpose(post)
        if uniform_prior == False:
            post = np.transpose([np.multiply(post[:,i],prior[i]) for i in range(len(prior))])
    
        colsum_post = np.sum(post,axis=1)
        colsum_post[colsum_post==0] = 1    
        loc_prob = post/np.reshape(colsum_post,(-1,1))
        
        return loc_prob

    # test ------------------------------

    def predict(self, X, merged=False, 
                covariance_type='full', covariance_tied=False, uniform_prior=False,
                gmm=None, gmm_cov=None, dtype_=np.float64):

        X_ = np.reshape(X, (X.shape[0],-1))

        if merged == False:
            self.loc_prob = self.compute_posterior(X, CELL_size=self.CELL_size, EPIC_size=None, merged=False,
                                              covariance_type=covariance_type, covariance_tied=covariance_tied, uniform_prior=uniform_prior,
                                              gmm=gmm, gmm_cov=gmm_cov, dtype_=dtype_)
            loc_models = self.local_models_CELL
        elif merged == True:
            self.loc_prob = self.compute_posterior(X, CELL_size=self.CELL_size, EPIC_size=self.EPIC_size, merged=True,
                                              covariance_type=covariance_type, covariance_tied=covariance_tied, uniform_prior=uniform_prior,
                                              gmm=gmm, gmm_cov=gmm_cov, dtype_=dtype_)
            loc_models = self.local_models_EPIC

        if self.statsmodels:
            loc_pred = np.transpose([loc_models[k].predict(sm.add_constant(X,has_constant='add')) for k in range(len(loc_models))])
        else:
            loc_pred = np.transpose([loc_models[k].predict(X) for k in range(len(loc_models))])
        pred = np.sum(np.multiply(loc_pred,self.loc_prob),axis=1)
        return pred


    # save & load ------------------------------

    def save_dict(self, name, remove_data=False, remove_loc_models=False):
        """save class as self.name.txt"""
        model_dict = dict([(key, value) for key, value in self.__dict__.items() if key not in ['nn_model','X_augmented','y_augmented','local_models_CELL','local_models_EPIC']])
        file = open(name+'.mlm','wb')
        pickle.dump(model_dict, file)
        file.close()

        if not remove_data:
            model_dict = dict([(key, value) for key, value in self.__dict__.items() if key in ['X_augmented','y_augmented']])
            file = open(name+'.augdat','wb')
            pickle.dump(model_dict, file)
            file.close()

        if not remove_loc_models and self.statsmodels:
            if self.__dict__.get('local_models_CELL') != None:
                i=0                
                for lm in self.__dict__.get('local_models_CELL'):
                    lm.save(name+str(i)+'.smc')
                    i=i+1
            if self.__dict__.get('local_models_EPIC') != None:
                i=0
                for lm in self.__dict__.get('local_models_EPIC'):
                    lm.save(name+str(i)+'.smp')
                    i=i+1
        elif not remove_loc_models and not self.statsmodels:
            model_dict = dict([(key, value) for key, value in self.__dict__.items() if key in ['local_models_CELL','local_models_EPIC']])
            file = open(name+'.skm','wb')
            pickle.dump(model_dict, file)
            file.close()

    def load_dict(self, name, add_data=True, add_loc_models=True):
        """try load self.name.txt"""
        with open(name+'.mlm', 'rb') as file:
            model_dict = pickle.load(file)
        for key, value in model_dict.items():
            self.__dict__[key] = value

        if add_data:
            with open(name+'.augdat', 'rb') as file:
                model_dict = pickle.load(file)
            for key, value in model_dict.items():
                self.__dict__[key] = value

        if self.statsmodels and add_loc_models:
            self.local_models_CELL = []
            for i in range(self.Ktilde):
                self.local_models_CELL.append(sm.load(name+str(i)+'.smc'))
            if 'Jtilde' in self.__dict__.keys():
                self.local_models_EPIC = []
                for i in range(self.Jtilde):
                    self.local_models_EPIC.append(sm.load(name+str(i)+'.smp'))
        elif not self.statsmodels and add_loc_models:
            with open(name+'.skm', 'rb') as file:
                model_dict = pickle.load(file)
            for key, value in model_dict.items():
                self.__dict__[key] = value
