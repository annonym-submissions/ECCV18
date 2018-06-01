import numpy as np
######################
def distortion_calculator(F_in,F_hat,rate=None):
    """Calculates the reconstruction distortion and AUC of D-R curve for each sample.

    F_hat can be a multi-layer successive tensor where at each layer
    new details help in estimation of F_in.
    Also the area-under-curve of the rate-distortion curve is
    calculated for each sample for which the rate can optinally be specified."""
    n,N,L = np.shape(F_hat)
    distortion_per_sample = np.divide(np.linalg.norm(F_in,axis=0)**2,n)
    # F_chapeau will be the cumulative sum of F_hat.
    F_chapeau = np.zeros(np.shape(F_in))
    for l in range(L):
        F_chapeau += F_hat[:,:,l]
        distortion_per_sample = np.vstack((distortion_per_sample,
                                          np.divide(np.linalg.norm(F_in-F_chapeau,
                                                                   axis=0)**2,n)))
    distortion = np.mean(distortion_per_sample,axis=1)
    if rate is None:
        DR_AUC = np.trapz(distortion_per_sample.T).reshape((1,N))
    elif isinstance(rate, np.ndarray):
        rate.reshape((1,L+1))
        DR_AUC = np.trapz(distortion_per_sample.T,rate).reshape((1,N))

    return distortion_per_sample,distortion,DR_AUC
#################################################################
def distortion_affine_normalizer(F_in,F_hat):
    """
    Affinely normalizes F_hat, to minimize the distortion w.r.t. F_in
    """
    n, N, L = np.shape(F_hat)
    F_in -= np.mean(F_in)
    F_chapeau = np.zeros(np.shape(F_in))
    distortion = np.divide(np.linalg.norm(F_in) ** 2, n*N)
    for l in range(L):
        F_chapeau += F_hat[:, :, l] - np.mean(F_hat[:, :, l])

        coeff_nume = 0.5*(np.linalg.norm(F_in)**2 + np.linalg.norm(F_chapeau)**2
        -np.linalg.norm(F_in - F_chapeau)**2)
        coeff_deno = np.linalg.norm(F_chapeau)**2
        reWeight_coeff = np.divide(coeff_nume,coeff_deno)
        #
        F_chapeau *= reWeight_coeff
        #
        distortion_temp = np.divide(np.linalg.norm(F_in - F_chapeau)**2,n*N)
        distortion = np.hstack((distortion,distortion_temp))
        #
    return distortion, reWeight_coeff
#############################################################
def R_Recall_at_T(Ground, Estimate, R=None, T=None):
    """
    Calculates R-recall@T measure btw. two lists
    """
    if R is None:
        R = np.shape(Ground)[0]
    if T is None:
        T = np.shape(Estimate)[0]
    ##
    Recall = 0
    for i in range(np.shape(Ground)[1]):
        Recall += len(set(Ground[:R, i]).intersection(set(Estimate[:T, i])))
    Recall /= np.shape(Ground)[1]
    return Recall


########################################################################
def mAP_at_k(Ground, Estimate, k=10):
    """
    Calculates the mean average precision @ k for ANN search evaluation.

    This is based on the code from (https://github.com/benhamner/Metrics/tree/master/Python/ml_metrics)
    and is simply interface-adapted to my own setup.

    Ground and Estimate are both 2D numpy arrays where columns represent queries and rows represent
    the indices to items from some database.
    """

    def apk(ground, estimate):
        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(estimate):
            if p in ground and p not in estimate[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        return score / min(ground.size, estimate.size)

    ######
    mAP = 0
    for i in range(np.shape(Ground)[1]):
        mAP += apk(Ground[:, i], Estimate[:k, i])
    mAP /= np.shape(Ground)[1]
    return mAP