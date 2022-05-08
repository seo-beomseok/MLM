from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from mixturelinearmodel import singleGaussModel, marginalGauss, marginalGaussMix

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import itertools
from tqdm import tqdm

# interpretation tools ------------------------------

def sort_EPIC(loc_prob):
    post = np.unique(np.argmax(loc_prob,axis=1),return_counts=True)
    sorted_postid = post[0][np.argsort(-post[1])]
    sorted_postsize = post[1][np.argsort(-post[1])]
    zero_postid = [k for k in range(loc_prob.shape[1]) if k not in sorted_postid]
    
    sorted_postid = np.concatenate((sorted_postid,np.array(zero_postid,dtype=int)))
    sorted_postsize = np.concatenate((sorted_postsize,np.zeros(len(zero_postid),dtype=int)))
    return sorted_postid, sorted_postsize

def plot_mosaic(MLM, epic_id=None, feature_names=None, intercept=True, sort=True, log_trans=True, ax=None):
    if ax == None:
        f, ax = plt.subplots(figsize=(7,3),dpi=300)
    
    cmap=plt.get_cmap('bwr')
    param = np.array([MLM.local_models_EPIC[i].params for i in range(MLM.Jtilde)])

    if epic_id == None:
        epic_id = range(param.shape[0])

    if log_trans:
        param = np.sign(param)*np.log(np.abs(param)+1)

    if sort:
        sorted_postid, sorted_postsize = sort_EPIC(MLM.loc_prob)
        param = param[sorted_postid[epic_id],:]
    else:
        param = param[epic_id,:]

    if feature_names == None:
        if 'feature_names' in MLM.__dict__.keys():
            feature_names = MLM.feature_names
        else:
            feature_names = ['v'+str(i) for i in range(param.shape[1])]
    if intercept:
        feature_names = np.concatenate([['intercept'],feature_names])

    vmax = np.max(np.abs(param))
    im = ax.imshow(param, cmap=cmap, vmin=-vmax, vmax=vmax)

    plt.xticks(ticks=range(0,len(feature_names)))
    ax.set_xticklabels(feature_names, rotation = 90)
    plt.yticks(ticks=range(0,len(epic_id)), labels = [i+1 for i in epic_id])
    plt.ylabel('EPIC')
    plt.xlabel('Variable')


    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

def plot_ci(MLM, feature_id, epic_id=None, feature_names=None, intercept=True, sort=True, log_trans=True, ax=None, yticklabels=True, title=True):
    if ax == None:
        f, ax = plt.subplots(1,1,figsize=(7,3),dpi=300)
    
    param = np.array([MLM.local_models_EPIC[i].params for i in range(MLM.Jtilde)])
    cilb = np.transpose([MLM.local_models_EPIC[i].conf_int() for i in range(MLM.Jtilde)])[0]
    ciub = np.transpose([MLM.local_models_EPIC[i].conf_int() for i in range(MLM.Jtilde)])[1]

    if epic_id == None:
        epic_id = range(param.shape[0])

    def log(x): return np.sign(x)*np.log(np.abs(x)+1)
    if log_trans:
        param = log(param)
        cilb = log(cilb)
        ciub = log(ciub)

    if sort:
        sorted_postid, sorted_postsize = sort_EPIC(MLM.loc_prob)
        param = param[sorted_postid[epic_id],:]
        cilb = cilb[:,sorted_postid[epic_id]]
        ciub = ciub[:,sorted_postid[epic_id]]
    else:
        param = param[epic_id,:]
        cilb = cilb[:,epic_id]
        ciub = ciub[:,epic_id]

    if feature_names == None:
        if 'feature_names' in MLM.__dict__.keys():
            feature_names = MLM.feature_names
        else:
            feature_names = ['v'+str(i) for i in range(param.shape[1])]
    if intercept:
        feature_names = np.concatenate([['intercept'],feature_names])

    data_dict = {}
    data_dict['category'] = ['EPIC'+str(i+1) for i in epic_id]
    data_dict['coef'] = param[:,feature_id]
    data_dict['lower'] = cilb[feature_id,:]
    data_dict['upper'] = ciub[feature_id,:]
    dataset = pd.DataFrame(data_dict)
    
    defcol = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for coef,lower,upper,y in zip(dataset['coef'],dataset['lower'],dataset['upper'],range(len(dataset))):
        ax.plot((lower,upper),(y,y),'-',color='#b642f5', linewidth=3, alpha=0.8)
        ax.scatter(coef,y,color='blue')
    ax.set_yticks(range(len(dataset)))
    ax.set_yticklabels(list(dataset['category']))
    ax.axvline(x=0,color='black',linewidth=1)
    if title:
        ax.set_title(feature_names[feature_id])

def compute_loc_prob(MLM, X, m, dim, uniform_prior=False):
    prior = np.array([np.sum(MLM.CELL_size[MLM.merge_map!=m]), np.sum(MLM.CELL_size[MLM.merge_map==m])]/np.sum(MLM.EPIC_size))
    post = np.zeros((2,X.shape[0]))
    post_ = []
    pi_ = []

    numgmm = np.sum(MLM.merge_map!=m)
    post_.append(np.zeros((numgmm,X.shape[0])))
    pi_.append(np.zeros((numgmm,)))

    numgmm = np.sum(MLM.merge_map==m)
    post_.append(np.zeros((numgmm,X.shape[0])))
    pi_.append(np.zeros((numgmm,)))
    
    
    pi, logprob = marginalGaussMix(MLM.gmm[MLM.merge_map!=m],MLM.CELL_size[MLM.merge_map!=m],list(dim)).logpdf(X[:,dim])
    post_[0] = logprob
    pi_[0] = pi
    
    pi, logprob = marginalGaussMix(MLM.gmm[MLM.merge_map==m],MLM.CELL_size[MLM.merge_map==m],list(dim)).logpdf(X[:,dim])
    post_[1] = logprob
    pi_[1] = pi
    

    submax = np.zeros((2,X.shape[0]))
    submax[0] = np.max(post_[0],axis=0)
    submax[1] = np.max(post_[1],axis=0)
    postmax = np.max(submax,axis=0)

    for i in range(2):
        post_[i] = np.exp(post_[i]-postmax)
        post[i] = np.matmul(pi_[i],post_[i])

    post = np.transpose(post)
    if uniform_prior == False:
        post = np.transpose([np.multiply(post[:,i],prior[i]) for i in range(len(prior))])
    
    colsum_post = np.sum(post,axis=1)
    colsum_post[colsum_post==0] = 1    
    loc_prob = post/np.reshape(colsum_post,(-1,1))
        
    return loc_prob


def explainable_tree(MLM, X, psi=1, epic_id=None,  sort=True):

    if epic_id == None:
        epic_id = range(MLM.Jtilde)

    if sort:
        sorted_postid, sorted_postsize = sort_EPIC(MLM.loc_prob)

    dtree = []
    binClass_list = []

    leave_node_set_list = []
    purity_list = []
    pure_size_list = []
    pure_leave_nodes_list = []

    def binary_classifier(k):
        return np.argmax(compute_loc_prob(MLM,X,k,dim=range(MLM.p)),axis=1)

    for m in epic_id:
        k = sorted_postid[m]
        binClass = binary_classifier(k)
        e = np.sum(binClass)

        dt0 = DecisionTreeClassifier(max_depth=None,random_state=0).fit(X,binClass)

        node_indicator = dt0.decision_path(X)
        print("Decision tree calculation has been done", end='\r')

        node_purity = np.zeros(dt0.tree_.node_count)
        node_ones = np.zeros(dt0.tree_.node_count)
        node_size = np.zeros(dt0.tree_.node_count)
        node_apply = []
    
        for i in range(dt0.tree_.node_count):
            node_apply.append(binClass[[np.where(node_indicator.indptr<=j)[0][-1] for j in np.where(node_indicator.indices==i)[0]]])
            node_ones[i] = sum(node_apply[i])
            node_size[i] = len(node_apply[i])
            node_purity[i] = node_ones[i]/node_size[i]
        print("Purity calculation has been done", end='\r')
    
        pure_nodes = np.where(node_purity>=psi)[0]
    
    
        st_index = np.where(node_indicator.indices==0)[0]

        pure_leave_nodes = []
    
        for i in pure_nodes:
            stop = False
    
            b = np.where(node_indicator.indices==i)[0][0]
            a = st_index[st_index<=b][-1]
            decision_path = node_indicator.indices[a:(b+1)]
    
            it=2
            l = len(decision_path)
            while it<=l:
                if (np.sum([decision_path[-it] == j for j in pure_nodes])) > 0:
                    stop = True
                    it = l
                it=it+1
            if stop:
                continue
            else:
                pure_leave_nodes.append(i)
        pure_leave_nodes = np.array(pure_leave_nodes)    
        print("Pure nodes calculation has been done", end='\r')
    
        leave_node_set = np.array(node_apply,dtype=object)[pure_leave_nodes]
            
        purity = [np.sum(i)/len(i)>=psi for i in leave_node_set]
        pure_size = np.multiply(purity,[np.sum(i) for i in leave_node_set])
        
    
        print('EPIC {m}(orig id {k}): found tree with max pure size {maxpuresize} and total pure size {puresize} for {totalsize}'.format(
                m=m,
                k=k,
                maxpuresize=np.max(pure_size),
                puresize=np.sum(pure_size),
                totalsize=np.sum(binClass)))
        dtree.append(dt0)
        binClass_list.append(binClass)
    
        leave_node_set_list.append(leave_node_set)
        purity_list.append(purity)
        pure_size_list.append(pure_size)
        pure_leave_nodes_list.append(pure_leave_nodes)
                
    MLM.dtree = dtree
    MLM.binClass_list = binClass_list

    MLM.leave_node_set_list = leave_node_set_list
    MLM.purity_list = purity_list
    MLM.pure_size_list = pure_size_list
    MLM.pure_leave_nodes_list = pure_leave_nodes_list
    

def explainable_condition(MLM, X, xi=50, epic_id=None,  feature_names=None, sort=True, origin_scale=False, export_condition=False):

    if not 'dtree' in MLM.__dict__.keys():
        print('Dtree is not found. Run explainable_tree(MLM, X) first.')
        return None

    if epic_id == None:
        epic_id = range(MLM.Jtilde)

    if sort:
        sorted_postid, sorted_postsize = sort_EPIC(MLM.loc_prob)

    if feature_names == None:
        if 'feature_names' in MLM.__dict__.keys():
            feature_names = MLM.feature_names
        else:
            feature_names = ['v'+str(i) for i in range(param.shape[1])]

    Xmax = np.min(X,axis=0)
    Xmin = np.max(X,axis=0)

    EPIC_desc = []
    EPIC_cond = []
    
    for k in epic_id:
        dt0 = MLM.dtree[k]
    
        n_nodes = dt0.tree_.node_count
        children_left = dt0.tree_.children_left
        children_right = dt0.tree_.children_right
        feature = dt0.tree_.feature
        threshold = dt0.tree_.threshold
    
    
        leave_node_set = MLM.leave_node_set_list[k]
        purity = np.array(MLM.purity_list[k])
        pure_size = np.array(MLM.pure_size_list[k])
        pure_leave_nodes = np.array(MLM.pure_leave_nodes_list[k])    
    
        node_description = []
        node_condition = []
    
        for i in range(np.sum(pure_size>xi)):
            crnt_node = np.array(pure_leave_nodes)[np.where(pure_size>xi)][i]
            crnt_size = pure_size[np.where(pure_size>xi)][i]
    
            node_description.append([])
            node_condition.append([])
            while crnt_node != 0:
                if np.any(children_left==crnt_node):
                    prev_node = np.where(children_left == crnt_node)[0][0]
                    if np.array_equal(np.unique(X[:,feature[prev_node]]),[0,1]):
                        sign = ' != '
                        sgid = 0
                        cond = '1'
                    else:
                        sign = ' <= '
                        sgid = -1
                        if origin_scale:
                            cond = np.round(threshold[prev_node]*(Xmax[feature[prev_node]]-Xmin[feature[prev_node]])+Xmin[feature[prev_node]],1)
                        else:
                            cond = np.round(threshold[prev_node],3)
                else:
                    prev_node = np.where(children_right == crnt_node)[0][0]
                    if np.array_equal(np.unique(X[:,feature[prev_node]]),[0,1]):
                        sign = ' == '
                        sgid = 0
                        cond = '1'
                    else:
                        sign = ' > '
                        sgid = 1
                        if origin_scale:
                            cond = np.round(threshold[prev_node]*(Xmax[feature[prev_node]]-Xmin[feature[prev_node]])+Xmin[feature[prev_node]],1)
                        else:
                            cond = np.round(threshold[prev_node],3)
                node_description[i].append((crnt_size,feature_names[feature[prev_node]] +sign+ str(cond)))
                node_condition[i].append([feature[prev_node],sgid,cond])
                crnt_node = prev_node
               
            
        node_cond0 = [np.array(x, dtype=float) for x in node_condition]
        node_cond1 = []
        node_desc0 = [np.array(x) for x in node_description]
        node_desc1 = []
    
        for i in range(len(node_cond0)):
            
            node_cond1.append([])
            node_cond1[i].append(node_cond0[i][0])
            node_desc1.append([])
            node_desc1[i].append(node_desc0[i][0])
            
            for j in range(1,len(node_cond0[i])):
                if np.any(np.prod(np.array(node_cond1[i])[:,0:2] == node_cond0[i][j,0:2],axis=1)):
                    where = np.where(np.prod(np.array(node_cond1[i])[:,0:2] == node_cond0[i][j,0:2],axis=1))[0][0]
                    if node_cond1[i][where][1] == 1.:
                        if node_cond1[i][where][2] < node_cond0[i][j,2]:
                            node_cond1[i][where][2] = node_cond0[i][j,2]
                            node_desc1[i][where] = node_desc0[i][j]
                    elif node_cond1[i][where][1] == -1.:
                        if node_cond1[i][where][2] > node_cond0[i][j,2]:
                            node_cond1[i][where][2] = node_cond0[i][j,2]
                            node_desc1[i][where] = node_desc0[i][j]
                else:
                    node_cond1[i].append(node_cond0[i][j])
                    node_desc1[i].append(node_desc0[i][j])
                
            node_cond1[i] = np.array(node_cond1[i])
            node_desc1[i] = np.array(node_desc1[i])
                
        for i in range(len(node_cond0)):
            node_desc1[i] = node_desc1[i][np.argsort(node_cond1[i][:,0])]
            node_cond1[i] = node_cond1[i][np.argsort(node_cond1[i][:,0])]
    
        EPIC_desc.append(node_desc1)
        EPIC_cond.append(node_cond1)

    if export_condition:
        return EPIC_cond
    else:
        return EPIC_desc


def explainable_dim(MLM, X, epic_id=None, threshold=0.8, max_dim = 2, greedy=True, sort=True, return_dim_score=False):

    if epic_id == None:
        epic_id = range(MLM.Jtilde)

    if sort:
        sorted_postid, sorted_postsize = sort_EPIC(MLM.loc_prob)

    dim_score = []
    explainable_dim = []
    EPIC_id_binary = []
    gmmrule_binary = []

    for m in epic_id:
        k = sorted_postid[m]
        EPIC_id_binary.append(1*np.array(np.argmax(MLM.loc_prob,axis=1)==k))
        gmmrule_binary.append([])
        dim_score.append([])
        for p_ in range(MLM.p):
            if greedy and p_!=0:
                dim_prev = tuple(dim_score[m][p_-1][np.argmax(dim_score[m][p_-1][:,4]),3])
                dim_cand = [dim_prev+(new,) for new in range(MLM.p) if new not in dim_prev]
            else:
                dim_cand = list(itertools.combinations(range(MLM.p),p_+1))
            mar_gmm_1 = []
            mar_gmm_2 = []
            gmmrule_binary[m].append([])
            dim_score[m].append([])
    
            for d in tqdm(range(len(dim_cand)), total=len(dim_cand), position=0, leave=True):
                dim = list(dim_cand[d])
            
                gmmrule_binary[m][p_].append(np.argmax(compute_loc_prob(MLM,X,k,dim),axis=1))
            
                acc = f1_score(gmmrule_binary[m][p_][d],EPIC_id_binary[m]) #f1_score
                
                dim_score[m][p_].append([k,p_,d,dim_cand[d],acc])
            
            dim_score[m][p_] = np.array(dim_score[m][p_], dtype=object)
            decision = dim_score[m][p_][:,-1] > threshold
            if np.sum(decision)>0:
                explainable_dim.append(dim_score[m][p_][decision,:])
                print(explainable_dim[m])
                break
            elif p_ == max_dim:
                explainable_dim.append(dim_score[m][p_])
                print(explainable_dim[m])
                break

    MLM.explainable_dim = explainable_dim

    if return_dim_score:
        return dim_score

def highest_explainable_dim(MLM, m, feature_names=None):
    if not 'explainable_dim' in MLM.__dict__.keys():
        print('Explainable_dim is not found. Run explainable_dim(MLM, X) first.')
        return None

    if feature_names == None:
        if 'feature_names' in MLM.__dict__.keys():
            feature_names = MLM.feature_names
        else:
            feature_names = ['v'+str(i) for i in range(param.shape[1])]

    k=np.argmax(np.array(MLM.explainable_dim[m])[:,4])
    return MLM.explainable_dim[m][k][0],k,MLM.explainable_dim[m][k][3],list(np.array(feature_names)[list(MLM.explainable_dim[m][k][3])]), np.round(MLM.explainable_dim[m][k][4],2)

def plot_id_1d(MLM, X, m, p_, d, dim, feature_names=None):
    if feature_names == None:
        if 'feature_names' in MLM.__dict__.keys():
            feature_names = MLM.feature_names
        else:
            feature_names = ['v'+str(i) for i in range(param.shape[1])]

    i = dim[0]
    fig, axs = plt.subplots(1,1, figsize=(5, 5), facecolor='w', edgecolor='k', dpi=300)
        
    a = (np.argmax(MLM.loc_prob,axis=1)==m)
    b = (np.argmax(MLM.loc_prob,axis=1)!=m)
    
    cmap=plt.get_cmap('viridis')
    axs.set_xlabel(feature_names[i])
    axs.set_yticks([])
    
    axs.scatter(X[b,i],np.zeros(sum(b)),s=100,c='lightgray',alpha=0.1, marker='|')
    axs.scatter(X[a,i],np.zeros(sum(a)),s=100,c=MLM.loc_prob[a,m],cmap=cmap,alpha=0.5, marker='|')
    
    plt.show()

def plot_id_2d(MLM, X, m, p_, d, dim, feature_names=None):
    if feature_names == None:
        if 'feature_names' in MLM.__dict__.keys():
            feature_names = MLM.feature_names
        else:
            feature_names = ['v'+str(i) for i in range(param.shape[1])]

    i,j = dim
    fig, axs = plt.subplots(1,1, figsize=(5, 5), facecolor='w', edgecolor='k', dpi=300)
    
    a = (np.argmax(MLM.loc_prob,axis=1)==m)
    b = (np.argmax(MLM.loc_prob,axis=1)!=m)
    
    cmap=plt.get_cmap('viridis')
    axs.set_ylabel(feature_names[j])
    axs.set_xlabel(feature_names[i])
    
    axs.scatter(X[b,i],X[b,j],s=5,c='gray',alpha=0.1)
    axs.scatter(X[a,i],X[a,j],s=5,c=MLM.loc_prob[a,m],cmap=cmap,alpha=0.5)
    
    plt.show()

def plot_id_3d(MLM, X, m, p_, d, dim, feature_names=None):
    if feature_names == None:
        if 'feature_names' in MLM.__dict__.keys():
            feature_names = MLM.feature_names
        else:
            feature_names = ['v'+str(i) for i in range(param.shape[1])]

    x,y,z = dim
    fig = plt.figure(figsize=(7, 7), dpi=300)
    ax0 = fig.add_subplot(111, projection='3d')
    
    a = (np.argmax(MLM.loc_prob,axis=1)==m)
    b = (np.argmax(MLM.loc_prob,axis=1)!=m)
    
    cmap=plt.get_cmap('viridis')
    ax0.scatter(X[b,x],X[b,y],X[b,z],s=3,c='gray',alpha=0.1)
    ax0.scatter(X[a,x],X[a,y],X[a,z],s=3,c=MLM.loc_prob[a,m],cmap=cmap,alpha=0.5)
    
    ax0.set_xlabel(feature_names[x])
    ax0.set_ylabel(feature_names[y])
    ax0.set_zlabel(feature_names[z])

    plt.show()



