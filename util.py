import os
import json
from sklearn_evaluation import plot as sklearnplot
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, roc_curve

def makedirs(d):

    if not os.path.exists(d):
        os.makedirs(d)
def ensure_dir(f):
    
    d = os.path.dirname(f)

    if not os.path.exists(d):
        try:
            os.makedirs(d)
        except:
            pass

def store_results(data, ofname):
    
    ensure_dir(ofname)
    with open(ofname, 'w') as fp:
        json.dump(data, fp)    

def plot_confusion_matrix(y, preds, classes, ofname, title='Confusion matrix', figsize=(10,10), cmap=cm.Blues, logscale=False, verbose=0):
    
    if verbose > 0:
        print("Plotting confusion matrix ...")
    fig = plt.figure(tight_layout=True, figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    sklearnplot.confusion_matrix(y, preds, target_names=classes, cmap=cmap, ax=ax)  

    ensure_dir(ofname)
    plt.savefig(ofname)
    plt.close()

def plot_image(img, ofname, titles=None, figsize=(10,5)):
    """
    """
        
    fig = plt.figure(tight_layout=True, figsize=figsize)
    
    for i in range(img.shape[0]):
        
        ax = fig.add_subplot(1, img.shape[0], i + 1)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        if titles is not None:
            ax.set_title(titles[i])
        im = ax.imshow(img[i,:,:], interpolation='none')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax,  format=ticker.FuncFormatter(_colorbar_fmt))
        
    plt.savefig(ofname, bbox_inches='tight')
    plt.close()            


def plot_misclassifications(y, preds, X, orig_indices, odir, titles=None, verbose=0):
    
    orig_indices = numpy.array(orig_indices)
    
    misclassifications = numpy.array(range(len(y)))
    misclassifications = misclassifications[y != preds]
    misclassifications_indices = orig_indices[y != preds]
    
    if verbose > 0:
        print("Number of test elements: %i" % len(y))
        print("Misclassifications: %s" % str(misclassifications_indices))
        print("Plotting misclassifications ...")
        
    for i in xrange(len(misclassifications)):
        
        index = misclassifications[i]
        orig_index = misclassifications_indices[i]
        ofname = os.path.join(odir, str(y[index]), str(orig_index) + ".png")
        ensure_dir(ofname)
        plot_image(X[index], ofname, titles=titles, mode=mode) 

def assess_classification_performance(preds, 
                                      preds_proba, 
                                      y_test, 
                                      X_test, 
                                      indices_test, 
                                      model, 
                                      odir, 
                                      image_labels, 
                                      clabels,  
                                      verbose=0, 
                                      plot_misses=True):

    mcc = matthews_corrcoef(y_test, preds)
    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)

    fpr, tpr, thres = roc_curve(y_test, preds_proba[:,1], pos_label=1)
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.savefig(os.path.join(odir, 'roc.png'))
    plt.close()

    if verbose > 0:
        print("----------------------------------------------------------------")
        print("Matthews Correlation Coefficient: \t" + str(mcc))
        print("Accuracy: \t\t\t\t" + str(acc))
        print("Precision: \t\t\t\t" + str(precision))
        print("Recall: \t\t\t\t" + str(recall))        
    
    if plot_misses:
        plot_misclassifications(y_test,
                                preds,
                                X_test,
                                indices_test,
                                odir,
                                titles=image_labels,
                                verbose=verbose)
    
    plot_confusion_matrix(y_test, preds, clabels, os.path.join(odir, "confusion.png"), verbose=verbose)
    store_results({'Matthews Correlation Coefficient':mcc, 
                       'Accuracy':acc,  
                       'Precision':precision, 
                       'Recall':recall}, 
                    os.path.join(odir, "results.txt"))    
    
