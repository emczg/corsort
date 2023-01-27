"""Main module."""
import numpy as np
from numpy import random



##

# mo = set of mothers
# da = set of daughters
# an_ = set of ancestors
# de_ = set of descendants
# x.downheight = max card(chain of descendants of x)
# x.upheight = max card(chain of ascendants of x)
# downlen = number of daughters
# u = uncertainty




class Node():

    def __init__(self,val,n,theta,eta):
        self.v=val
        self.mo=set() # set of mothers
        self.da=set() # set of daughters
        self.an=set() # set of ancestors
        self.de=set() # set of descendants
        self.downheight=0
        self.upheight=0
        self.rank=None # downheight - upheight
        self.altrank = None # rank + some alteration
        self.downlen=None # number of daughters
        self.uplen=None # number of mothers
        self.u=None # uncertainty
        self.refreshnode(n,theta,eta)

    def __gt__(self,other):
        return self.v>other.v


    def __repr__(self):
        return "Node("+str(self.v)+")"

#    def __str__(self):
#        return "toto str"

    def refreshnode(self,n,theta,eta):
        self.u=n-1-len(self.an)-len(self.de)
        self.downlen=len(self.da)
        self.uplen = len(self.mo)
        self.rank=self.downheight-self.upheight
        self.altrank=self.rank + theta*self.downlen - eta*self.uplen





class Corsort():
    def __init__(self,P):
        self.poset=P #[Node(val,n_) for val in random.sample( n_)]
        self.n=len(P)
        self.rank=[x.altrank for x in self.poset]
        self.currentsort=self.poset
        self.currentrank=None
        self.refreshrank()

    def refreshrank(self):
        self.currentrank=[x.altrank for x in self.currentsort]




    def bt(self,index1,index2):
        currentsort=self.currentsort
        node1=currentsort[index1]
        node2=currentsort[index2]
        node1.da.add(node2)
        node1.de_.add(node2)
        node1.de_ = node1.de_ | node2.de_
        node2.mo.add(node1)
        node2.an_.add(node1)
        node2.an_ = node2.an_ | node1.an_
        self.updatedown(index1,index2)
        self.updateup(index1,index2)
        self.updatesort(index1,index2)


    def updatedown(self,index1,index2):
        currentsort=self.currentsort
        node1=currentsort[index1]
        node2=currentsort[index2]
        for ancestor in node1.an_:
            ancestor.downheight=max(ancestor.downheight,ancestor.downheight- node1.downheight + node2.downheight+1)
            ancestor.refreshnode(self.n)
        node1.downheight=max(node1.downheight,node2.downheight+1)
        node1.refreshnode(self.n)
        node2.refreshnode(self.n)
        self.refreshrank()

    def updateup(self,index1,index2):
        currentsort=self.currentsort
        node1=currentsort[index2]
        node2=currentsort[index1]
        for descendant in node1.an_:
            descendant.upheight=max(descendant.upheight,descendant.upheight- node1.upheight + node2.upheight+1)
            descendant.refreshnode(self.n)
        node1.upheight=max(node1.upheight,node2.upheight+1)
        node1.refreshnode(self.n)
        node2.refreshnode(self.n)
        self.refreshrank()


    def updatesort(self,index1,index2):
        currentrank=self.currentrank
        currentsort = self.currentsort
        node1=currentsort[index1]
        node2=currentsort[index2]
        k=index1
        l=index2
        if k>l:
            while node1.rank>currentsort[k-1].rank and k-1>0:
                currentsort[k-1],currentsort[k]=currentsort[k],currentsort[k-1]
                k-=1
            while node2.rank>currentsort[l+1].rank and l<self.n-1:
                currentsort[l+1],currentsort[l]=currentsort[l],currentsort[l+1]
                l+=1
        else:
            currentsort[k-1], currentsort[k] = currentsort[k], currentsort[k-1]
            k-=1
            l+=1
            while node1.rank>currentsort[l-1].rank and l-1>0:
                currentsort[l-1],currentsort[l]=currentsort[l],currentsort[l-1]
                l-=1
            while node2.rank>currentsort[k+1].rank and k<self.n-1:
                currentsort[k+1],currentsort[k]=currentsort[k],currentsort[k+1]
                k+=1
        self.refreshrank()


    def nextcomp(self):
        currentrank=self.currentrank
        diff=[abs(currentrank[i]-currentrank[i+1]) for i in range(self.n-1)]
        mini=0
        for k in range(self.n-1):
            if diff[k]<diff[mini]:
                mini=k
        return (mini,mini+1)


    def corsort(self,P):
        self.__init__(P)
        currentsort=self.poset
        compteur=0
        while sum([node.u for node in currentsort])>0:
            index1,index2=self.nextcomp()
            node1=currentsort[index1]
            node2=currentsort[index2]
            if node1>node2:
                self.bt(index1,index2)
            else:
                self.bt(index2,index1)
            #self.__gt__(index1,index2)
            compteur+=1
            print(compteur)
            print(self.currentsort)
        return(self.poset,self.currentsort,compteur)

