#!usr/bin/env python
#-*- coding: utf-8 -*-

from __future__ import division
import itertools
from collections import defaultdict
import numpy as np
from numpy.random import uniform,dirichlet,multinomial,normal,multivariate_normal,beta,binomial
import random
from numpy import exp,log

"""list of branches representing the following Indo-Aryan tree:
Proto-Indo-Aryan
>>>>Prakrit
>>>>Pali
>>>>Central_Indo-Aryan
>>>>>>>>Khari_Boli
>>>>>>>>>>>>Hindi
>>>>>>>>>>>>Urdu
>>>>>>>>Punjabi
>>>>>>>>Romani
>>>>>>>>>>>>Romani_(Arli)
>>>>>>>>>>>>Romani_(Sepečides)
>>>>>>>>>>>>Northern_Romani
>>>>>>>>>>>>>>>>Angloromani
>>>>>>>>>>>>>>>>Scandoromani
>>>>>>>>>>>>>>>>Romani_(Burgenland)
>>>>>>>>>>>>>>>>Romani_(Kale)
>>>>>>>>>>>>>>>>Romani_(Sinte)
>>>>>>>>>>>>Vlax_Romani
>>>>>>>>>>>>>>>>Romani_(Kelderash)
>>>>>>>>>>>>>>>>Romani_(Lovara)
>>>>Eastern_Indo-Aryan
>>>>>>>>Assamese
>>>>>>>>Bengali
>>>>>>>>Maithili
>>>>>>>>Oriya
>>>>Nepali
>>>>Northwestern_Indo-Aryan
>>>>>>>>Kashmiri
>>>>>>>>Khowar
>>>>>>>>Sindhi
>>>>Sinhalese-Maldivian
>>>>>>>>Sinhalese
>>>>>>>>Maldivian
>>>>Southern_Indo-Aryan
>>>>>>>>Konkani
>>>>>>>>Marathi
>>>>Gujarati
Each list within the list has the following structure: [PARENT NODE, CHILD NODE, BRANCH LENGTH]
(This tree is not meant to be realistic, just for instructional purposes.)
"""

branches = [['Southern_Indo-Aryan', 'Konkani', 1700], ['Southern_Indo-Aryan', 'Marathi', 1700], ['Eastern_Indo-Aryan', 'Assamese', 1700], ['Eastern_Indo-Aryan', 'Bengali', 1700], ['Eastern_Indo-Aryan', 'Maithili', 1700], ['Eastern_Indo-Aryan', 'Oriya', 1700], ['Khari_Boli', 'Hindi', 800], ['Khari_Boli', 'Urdu', 800], ['Northwestern_Indo-Aryan', 'Kashmiri', 1700], ['Northwestern_Indo-Aryan', 'Khowar', 1700], ['Northwestern_Indo-Aryan', 'Sindhi', 1700], ['Northern_Romani', 'Angloromani', 500], ['Northern_Romani', 'Scandoromani', 500], ['Northern_Romani', 'Romani_(Burgenland)', 500], ['Northern_Romani', 'Romani_(Kale)', 500], ['Northern_Romani', 'Romani_(Sinte)', 500], ['Vlax_Romani', 'Romani_(Kelderash)', 500], ['Vlax_Romani', 'Romani_(Lovara)', 500], ['Sinhalese-Maldivian', 'Sinhalese', 1900], ['Sinhalese-Maldivian', 'Maldivian', 1900], ['Romani', 'Romani_(Arli)', 1100], ['Romani', 'Romani_(Sepeides)', 1100], ['Romani', 'Northern_Romani', 600], ['Romani', 'Vlax_Romani', 600], ['Central_Indo-Aryan', 'Khari_Boli', 900], ['Central_Indo-Aryan', 'Punjabi', 1700], ['Central_Indo-Aryan', 'Romani', 600], ['Proto-Indo-Aryan', 'Prakrit', 700], ['Proto-Indo-Aryan', 'Pali', 600], ['Proto-Indo-Aryan', 'Central_Indo-Aryan', 1500], ['Proto-Indo-Aryan', 'Eastern_Indo-Aryan', 1500], ['Proto-Indo-Aryan', 'Nepali', 3200], ['Proto-Indo-Aryan', 'Northwestern_Indo-Aryan', 1500], ['Proto-Indo-Aryan', 'Sinhalese-Maldivian', 1300], ['Proto-Indo-Aryan', 'Southern_Indo-Aryan', 1500], ['Proto-Indo-Aryan', 'Gujarati', 3200]]

"""list of nodes to be visited by pruning algorithm, in post-traversal order"""

pruneorder = ['Southern_Indo-Aryan', 'Eastern_Indo-Aryan', 'Khari_Boli', 'Northwestern_Indo-Aryan', 'Northern_Romani', 'Vlax_Romani', 'Sinhalese-Maldivian', 'Romani', 'Central_Indo-Aryan', 'Proto-Indo-Aryan']

root = pruneorder[-1] #root of tree is 

mother = {}
daughters = defaultdict(list)
brlen = {}

for b in branches:
    mother[b[1]]=b[0]
    daughters[b[0]].append(b[1])
    brlen[b[1]]=b[2]/1000


featdict = {'Prakrit': (0, 1), 'Kashmiri': (1, 0), 'Romani_(Burgenland)': (1, 0), 'Marathi': (0, 1), 'Oriya': (1, 0), 'Gujarati': (0, 1), 'Romani_(Sinte)': (1, 0), 'Khowar': (1, 0), 'Hindi': (1, 0), 'Romani_(Kelderash)': (1, 0), 'Konkani': (0, 1), 'Sindhi': (1, 0), 'Romani_(Kale)': (1, 0), 'Sinhalese': (1, 0), 'Bengali': (1, 0), 'Maithili': (1, 0), 'Scandoromani': (0, 1), 'Assamese': (1, 0), 'Maldivian': (1, 0), 'Nepali': (1, 0), 'Punjabi': (1, 0), 'Romani_(Arli)': (1, 0), 'Urdu': (1, 0), 'Angloromani': (1, 0), 'Romani_(Lovara)': (1, 0), 'Pali': (0, 1), 'Romani_(Sepeides)': (1, 0)}


"""the following function calculates the CTMC transition probability for a branch of length t under rates a and b"""

def makemat(a,b,t): #generate transitional matrix from infinitesimal rates (faster than scipy.linalg, I think)
    m = [[0,0],[0,0]]
    m[0][0] = (b/(a+b))+((a/(a+b))*exp(-1*(a+b)*t))
    m[0][1] = (a/(a+b))-((a/(a+b))*exp(-1*(a+b)*t))
    m[1][0] = (b/(a+b))-((b/(a+b))*exp(-1*(a+b)*t))
    m[1][1] = (a/(a+b))+((b/(a+b))*exp(-1*(a+b)*t))
    return m

"""the following function implements the pruning algorithm"""

def prune(a,b,treelik=True): #compute tree likelihood under current gain and loss rates using pruning algorithm
    for n in pruneorder: #visit each node in post-traversal order
        pi = [0,0] 
        for k in daughters[n]:
            pi_k = makemat(a,b,brlen[k])
            pi_k[0][0] *= featdict[k][0]
            pi_k[1][0] *= featdict[k][0]
            pi_k[0][1] *= featdict[k][1]
            pi_k[1][1] *= featdict[k][1]
            pi[0] += log(sum(pi_k[0]))   #probability that node n == 0
            pi[1] += log(sum(pi_k[1]))   #probability that node n == 1
        featdict[n] = tuple(exp(pi))     #store probabilities
    if treelik==True: #if computing whole tree likelihood rather than computing ancestral state likelihoods for reconstruction
        phi = log(np.dot(featdict[root],[b/(a+b),a/(a+b)]))  #log overall likelihood of tree
        return phi


posterior=defaultdict()

"""MCMC inference for rates a,b"""

def inference(chains=3,iters=100000):
    global posterior
    burnin=int(iters/2) #discard 1st half of samples
    thin=int(iters/100) #store only a 100th of samples
    acc_prop={} #data structure for chain acceptance probability (the chain should accept around 23% of proposed states)
    for c in range(chains):
        posterior[c]=defaultdict(list)
        a = normal(0,10) 
        b = normal(0,10)
        a_step = .5           #initialize step sizes for rate proposals
        b_step = .5
        accepted = []
        lp_curr = prune(exp(a),exp(b))
        for t in range(iters):
            a_prime = normal(a,a_step)
            b_prime = normal(b,b_step)
            lp_prime = prune(exp(a_prime),exp(b_prime))
            acc = uniform(0,1)
            if min(1,exp(lp_prime-lp_curr)) > acc:
                a,b,lp_curr=a_prime,b_prime,lp_prime
#                    a,b=a_prime,b_prime
                accepted.append(1)
            else:
                accepted.append(0)
#                print t,i,a,b,lp_curr
            if t in range(thin,burnin,thin):
                if sum(accepted)/len(accepted) < .22:
                    a_step *= exp(-.5)
                    b_step *= exp(-.5)
                if sum(accepted)/len(accepted) > .25:
                    a_step *= exp(.5)
                    b_step *= exp(.5) 
            if t in range(burnin,iters,thin):
                posterior[c]['a'].append(a)
                posterior[c]['b'].append(b)
        acc_prop[c]=sum(accepted)/len(accepted)


#random.seed(0) #set seeds for duplicability
#np.random.seed(0)


#a_step = .5
#b_step = .5
acc_prop={} #data structure for chain acceptance probability (the chain should accept around 23% of proposed states)
def inference(chains=3,iters=10000):
    global posterior
    burnin=int(iters/2) #discard 1st half of samples
    thin=int(iters/100) #store only a 100th of samples
    for c in range(chains):
        posterior[c]=defaultdict(list)
        a = uniform(0,.5) #initialize gain rate
        while a == 0: #make sure gain rate > 0
            a = uniform(0,.5)
        b = uniform(0,.5) #initialize loss rate
        while b == 0: #make sure loss rate > 0
            b = uniform(0,.5)
        a_step = .5           #initialize step sizes for rate proposals
        b_step = .5
        accepted = []
        lp_curr = prune(a,b) #compute likelihood of tree under current a,b
        for t in range(iters):
            a_prime = normal(a,a_step) #propose new value for a
            while a_prime <= 0: #or a_prime >= 10:        #make sure proposed value > 0
                a_prime = normal(a,a_step)
            b_prime = normal(b,b_step) #propose new value for b
            while b_prime <= 0: #or b_prime >= 10:
                b_prime = normal(b,b_step)
            lp_prime = prune(a_prime,b_prime) #compute likelihood of tree under proposed new values for a,b
            acc = uniform(0,1)                #draw random acceptance probability
            if min(1,exp(lp_prime-lp_curr)) > acc:
                a,b,lp_curr=a_prime,b_prime,lp_prime
#                    a,b=a_prime,b_prime
                accepted.append(1)
            else:
                accepted.append(0)
            print t,a,b,lp_curr
#            if t in range(thin,burnin,thin):
#                if sum(accepted)/len(accepted) < .22:
#                    a_step *= exp(-.5)
#                    b_step *= exp(-.5)
#                if sum(accepted)/len(accepted) > .25:
#                    a_step *= exp(.5)
#                    b_step *= exp(.5)
#                print sum(accepted)/len(accepted),a_step,b_step
            if t in range(burnin,iters,thin):
                posterior[c]['a'].append(a)
                posterior[c]['b'].append(b)
        acc_prop[c]=sum(accepted)/len(accepted)



"""monitor for convergence via Gelman-Rubin R_hat (values below 1.1 indicate convergence)"""
def gelmandiag():
    return [(np.var(posterior[0]['a']+posterior[1]['a']+posterior[2]['a'])/((np.var(posterior[0]['a'])+np.var(posterior[1]['a'])+np.var(posterior[2]['a']))/3))**.5,(np.var(posterior[0]['b']+posterior[1]['b']+posterior[2]['b'])/((np.var(posterior[0]['b'])+np.var(posterior[1]['b'])+np.var(posterior[2]['b']))/3))**.5]


inference(iters=5000)
gelmandiag()


for k in posterior.keys():
    print 'chain',k,'alpha mean',np.mean(posterior[k]['a']),'std. dev.',np.var(posterior[k]['a'])**.5
    print 'chain',k,'beta mean',np.mean(posterior[k]['b']),'std. dev.',np.var(posterior[k]['b'])**.5



"""ancestral state simulation using rates inferred in previous procedure"""


states = defaultdict(list) #data structure for sampled ancestral states


def ancstate(a,b): #draw ancestral state probabilities, given gain and loss rates
    prune(a,b,treelik=False) #use pruning algorithm to get likelihoods under a,b
    state = {} #data structure for currently sampled states
    preorder = pruneorder[::-1] #list of nodes in pre-traversal order
    root = preorder[0]
    pi_root = (a/(a+b))*featdict[root][1]/((b/(a+b))*featdict[root][0]+(a/(a+b))*featdict[root][1]) #p(root=1)
    state[root] = binomial(1,pi_root)
    states[root].append(state[root])
    for n in preorder[1:]: #for all non-root internal nodes
        s_p = state[mother[n]] #collect currently sampled parent state
        pi = makemat(a,b,brlen[n]) #transition probabilities
        pi_n = (featdict[n][1]*pi[s_p][1])/((featdict[n][1]*pi[s_p][1])+(featdict[n][1]*pi[s_p][0])) #p(n=1)
        state[n] = binomial(1,pi_n)
        states[n].append(state[n])

"""iteratively sample ancestral states based on posterior rates"""

"""flatten posterior samples for alpha and beta"""
post_a = posterior[0]['a']+posterior[1]['a']+posterior[2]['a']
post_b = posterior[0]['b']+posterior[1]['b']+posterior[2]['b']



for t in range(10000):
    a = random.sample(post_a,1)[0] #sample from posterior distribution with replacement
    b = random.sample(post_b,1)[0]
    ancstate(a,b)


"""show averages of samples for each interior node"""
for k in states.keys():
    print k,np.mean(states[k])
