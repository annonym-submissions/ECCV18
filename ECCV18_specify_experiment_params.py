# # #  Database selection:
# Database = 'MNIST'
# Database = 'GIST1M'
Database = 'SIFT1M'
######################################################################################
# Hyper-parameter setting:
L = 9
k = [20]*3 + [4]*(L-3)
#m = [10]*2 + [100] * (L-2)
# k = 20
m = 128
numSB = 1
vote_layer_dismiss = 4
Learner = 'SuccessivePCA'
initial_list_size = 4096
nu = 10
nu_prime = 20
rate_calc_exct = False
####################################################################################

