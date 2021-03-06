#install.packages('phytools','ggplot2')
require(phytools)
require(ggplot2)

#data from Dunn et al., 2017.

#read csv file of attested data into data frame
gmc.data <- data.frame(read.csv('https://raw.githubusercontent.com/evoling/dative-sickness/master/data/GmcArgStrucStates.csv', sep='\t',header=F,row.names=1))

#newick tree form https://raw.githubusercontent.com/evoling/dative-sickness/master/data/germanic-tree-formatted.tre
#gmc.newick <- "(Gothic:850.0,(((Old_Norse:50.0,(Faroese:750.0,Icelandic:750.0,Norwegian:750.0):50.0):300.0,(Old_Swedish:50.0,Swedish:650.0):450.0):1000.0,(((Old_High_German:50.0,(Middle_High_German:50.0,German:750.0):400.0):350.0,(Old_Saxon:100.0,(Middle_Dutch:50.0,Dutch:700.0):450.0):350.0):350.0,(Old_English:50.0,(Middle_English:50.0,English:700.0):400.0):750.0):250.0):350.0);"



#tweaked version of newick tree form https://raw.githubusercontent.com/evoling/dative-sickness/master/data/germanic-tree-formatted.tre
gmc.newick <- "(Gothic:850.0,(((Old_Norse:1.0,(Faroese:750.0,Icelandic:750.0,Norwegian:750.0):50.0):349.0,(Old_Swedish:1.0,Swedish:650.0):499.0):1000.0,(((Old_High_German:1.0,(Middle_High_German:1.0,German:750.0):449.0):399.0,(Old_Saxon:1.0,(Middle_Dutch:1.0,Dutch:700.0):499.0):399.0):350.0,(Old_English:1.0,(Middle_English:1.0,English:700.0):449.0):799.0):250.0):350.0);"

#convert newick string to phylo object
gmc.tree <- read.newick(textConnection(gmc.newick))

#make sure that order of languages in data frame and in tree match
gmc.data <- gmc.data[gmc.tree$tip.label,]

#define a function returning nodes of tree in post-order traversal
make.prune.order <- function(tr) {
    new.order <- c(1:length(tr$tip.label))
    old.order <- unique(tr$edge[,1])
    while (length(old.order)!=0) {
        for (i in length(old.order):1) {
            subtree <- tr$edge[tr$edge[,1]==old.order[i],]
            if (subtree[1,2] %in% new.order && subtree[2,2] %in% new.order) {
                new.order<- c(new.order,old.order[i])
                old.order <- old.order[-i]
        }
        }
    }
    return(new.order)
}

#generate post-traversal order
prune.order <- make.prune.order(gmc.tree)

#make a data frame of (1) parent node (2) child node and (3) length of branch connecting them, for each branch in tree
gmc.edge <- data.frame(gmc.tree$edge,gmc.tree$edge.length)
colnames(gmc.edge) <- c('p','c','l')

#scale branch lengths to avoid underflow
gmc.edge$l <- gmc.edge$l/1000

#number of tips in tree
T <- length(gmc.tree$tip.label)

#number of total nodes in tree
N <- T + gmc.tree$Nnode 

#function to generate matrix of likelihoods for languages in column i of data frame, as well as compute likelihoods of internal nodes
make.lik.mat <- function(i) {
    lik.mat <- matrix(1,nrow=N,ncol=3)
    allstates <- c('A','D','N')
    for (j in 1:T) {
        if (gmc.data[j,i] != '-' && nchar(as.character(gmc.data[j,i])) < 3) {  #if the character state is not '-' or some combination of all three states
            states = unlist(strsplit(as.character(gmc.data[j,i]),split=''))
            for (k in 1:3) {
                if (!(allstates[k] %in% states)) {
                    lik.mat[j,k] <- 0                            #set likelihoods for unattested states to zero
                }
            }
        }
    }
    return(lik.mat)
}

#compute likelihood of tree from matrix of likelihoods for a particular feature, plus rate matrix Q
prune.lik <- function(lmat,Q) {
    curr.lik.mat <- lmat
    for (i in (T+1):N) {
        n <- prune.order[i]
        subtree <- gmc.edge[gmc.edge$p==n,]
        lambda <- c(1,1,1)
        for (j in 1:nrow(subtree)) {
            child <- subtree[j,]$c
            brlen <- subtree[j,]$l
            P <- expm(Q*brlen)
            phi <- c(0,0,0)
            for (k in 1:3) {
#                phi <- phi + P[k,]*curr.lik.mat[child,]
                for (l in 1:3) {
                    phi[k] <- phi[k] + P[k,l]*curr.lik.mat[child,l]
                }
            }
            lambda <- lambda*phi
        }
#        print(lambda)
        curr.lik.mat[n,] <- lambda
    }
    rho <- curr.lik.mat[nrow(curr.lik.mat),] #root likelihood
#    pi <- eigen(Q)$vectors[,3]/sum(eigen(Q)$vectors[,3]) #prior of state
    pi <- c(1,1,1)/3
#    print(rho)
    return(sum(rho*pi))
}

#make rate matrix from rates
gen.Q <- function(qAD,qAN,qDA,qDN,qNA,qND) {
    qAD<-exp(qAD); qAN<-exp(qAN); qDA<-exp(qDA); qDN<-exp(qDN); qNA<-exp(qNA); qND<-exp(qND)
    Q <- matrix(nrow=3,ncol=3)
    Q[1,] <- c(-(qAD+qAN),qAD,qAN)
    Q[2,] <- c(qDA,-(qDA+qDN),qDN)
    Q[3,] <- c(qNA,qND,-(qNA+qND))
    return(Q)
}

verbs <- c('hunger','thirst','like','lack','dream','avail','lust','long','wonder','think/seem','suffice','fail')


infer.rates <- function(i) {
    verb <- c()
    rate <- c()
    value <- c()
    iters <- 10000                    #number of iterations
    burnin <- iters/2                 #burn-in (below which all samples are discarded)
    thin <- 10                        #thinning interval
    #make likelihood matrix for feature i
    feat.mat <- make.lik.mat(i)       #make likelihood matrix
        for (c in 1:3) {              #for each chain
        accepted <- 0                 #number of times proposal is accepted
        #initialize rates
        qAD <- runif(1,0,1)
        while (qAD==0) {
            qAD <- runif(1,0,1)
        }
        qAN <- runif(1,0,1)
        while (qAN==0) {
            qAN <- runif(1,0,1)
        }
        qDA <- runif(1,0,1)
        while (qDA==0) {
            qDA <- runif(1,0,1)
        }
        qDN <- runif(1,0,1)
        while (qDN==0) {
            qDN <- runif(1,0,1)
        }
        qNA <- runif(1,0,1)
        while (qNA==0) {
            qNA <- runif(1,0,1)
        }
        qND <- runif(1,0,1)
        while (qND==0) {
            qND <- runif(1,0,1)
        }
        #generate matrix of current rates
        Q <- gen.Q(qAD,qAN,qDA,qDN,qNA,qND)      
        #compute likelihood under current rates
        lik.curr <- prune.lik(feat.mat,Q)
        delta <- .5 #step size
        for (t in 1:iters) {                     #for each iteration
            #propose new rates
            qAD.p <- qAD+rnorm(1,0,delta)
            while (qAD.p >= 2) {
                qAD.p <- qAD+rnorm(1,0,delta)
            }
            qAN.p <- qAN+rnorm(1,0,delta)
            while (qAN.p >= 2) {
                qAN.p <- qAN+rnorm(1,0,delta)
            }
            qDA.p <- qDA+rnorm(1,0,delta)
            while (qDA.p >= 2) {
                qDA.p <- qDA+rnorm(1,0,delta)
            }
            qDN.p <- qDN+rnorm(1,0,delta)
            while (qDN.p >= 2) {
                qDN.p <- qDN+rnorm(1,0,delta)
            }
            qNA.p <- qNA+rnorm(1,0,delta)
            while (qNA.p >= 2) {
                qNA.p <- qNA+rnorm(1,0,delta)
            }
            qND.p <- qND+rnorm(1,0,delta)
            while (qND.p >= 2) {
                qND.p <- qND+rnorm(1,0,delta)
            }
            #generate matrix of proposed rates
            Q.prime <- gen.Q(qAD.p,qAN.p,qDA.p,qDN.p,qNA.p,qND.p)
            #compute likelihood under proposed rates
            lik.prop <- prune.lik(feat.mat,Q.prime)
            a <- runif(1)   #acceptance probability for Metropolis-Hastings
            #accept new rates if the following condition is met
            if (min(1,exp(log(lik.prop)-log(lik.curr))) > a) {
                qAD <- qAD.p
                qAN <- qAN.p
                qDA <- qDA.p
                qDN <- qDN.p
                qNA <- qNA.p
                qND <- qND.p
                lik.curr <- lik.prop
            }
#            print(c(qAD,qAN,qDA,qDN,qNA,qND,lik.curr))
#            if (t %in% seq(0,burnin,thin*10)) {
#                #tune step size parameter during burnin period
#                if (accepted/t < .15) {
#                    delta <- delta*exp(.5)
#                }
#                if (accepted/t > .35) {
#                    delta <- delta*exp(-.5)
#                }
#            }
            if (t %in% seq(burnin,iters,thin)) {
                #store samples to posterior
                verb <- c(verb,rep(verbs[i],6))
                rate <- c(rate,c('qAD','qAN','qDA','qDN','qNA','qND'))
                value <- c(value,c(qAD,qAN,qDA,qDN,qNA,qND))
            }
        }
    }
    posterior <- data.frame(verb,rate,value)
    return(posterior)
}


#post<-infer.rates(1)

set.seed(1)
post <- data.frame()
for (i in 1:12) {
    post <- rbind(post,infer.rates(i))
}

levels(post$rate)<-c('qAD','qAN','qDN','qDA','qNA','qND')
sumpost<-aggregate(value ~ rate+verb,post,FUN='median')
#sumpost$sick <- rep(NA,nrow(sumpost))
#sumpost[sumpost$rate=='qDA'|sumpost$rate=='qAD'|sumpost$rate=='qDN'|sumpost$rate=='qAN',]$sick <- 'S'
#sumpost[sumpost$rate=='qND'|sumpost$rate=='qNA',]$sick <- 'W'
#aggregate(value~sick,sumpost,FUN='mean')
#wilcox.test(value~sick,sumpost)

mean(sumpost[sumpost$rate=='qAD'|sumpost$rate=='qDN'|sumpost$rate=='qAN',]$value)
mean(sumpost[sumpost$rate=='qND'|sumpost$rate=='qNA'|sumpost$rate=='qDA',]$value)
wilcox.test(sumpost[sumpost$rate=='qAD'|sumpost$rate=='qDN'|sumpost$rate=='qAN',]$value,sumpost[sumpost$rate=='qND'|sumpost$rate=='qNA'|sumpost$rate=='qDA',]$value)
wilcox.test(sumpost[sumpost$rate=='qAD'|sumpost$rate=='qDN'|sumpost$rate=='qAN'|sumpost$rate=='qDA',]$value,sumpost[sumpost$rate=='qND'|sumpost$rate=='qNA',]$value)
wilcox.test(sumpost[sumpost$rate=='qAD',]$value,sumpost[sumpost$rate=='qDA',]$value)

mean(sumpost[sumpost$rate=='qAD',]$value)
mean(sumpost[sumpost$rate=='qDA',]$value)
wilcox.test(sumpost[sumpost$rate=='qAD',]$value,sumpost[sumpost$rate=='qDA',]$value)

mean(sumpost[sumpost$rate=='qDN',]$value)
mean(sumpost[sumpost$rate=='qND',]$value)
wilcox.test(sumpost[sumpost$rate=='qND',]$value,sumpost[sumpost$rate=='qDN',]$value)

mean(sumpost[sumpost$rate=='qAN',]$value)
mean(sumpost[sumpost$rate=='qNA',]$value)
wilcox.test(sumpost[sumpost$rate=='qNA',]$value,sumpost[sumpost$rate=='qAN',]$value)




#see also https://gist.github.com/anonymous/1b0a2b78664b7a33c4e1dce53ba5224b
