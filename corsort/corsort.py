"""Main module."""
import numpy as np
from numpy import random



##

# mo = set of mothers
# da = set of daughters
# an = set of ancestors
# de = set of descendants
# x.downheight = max card(chain of descendants of x)
# x.upheight = max card(chain of ascendants of x)
# downlen = number of daughters
# u = uncertainty




class Node():

    def __init__(self,val,n):
        self.v=val
        self.mo=set() # set of mothers
        self.da=set() # set of daughters
        self.an=set() # set of ancestors
        self.de=set() # set of descendants
        self.downheight=0
        self.upheight=0
        self.rank=None # downheight - upheight
        self.downlen=None # number of daughters
        self.u=None # uncertainty
        self.refreshnode(n)

    def __gt__(self,other):
        return self.v>other.v


    def __repr__(self):
        return "Node("+str(self.v)+")"

#    def __str__(self):
#        return "toto str"

    def refreshnode(self,n):
        self.u=n-1-len(self.an)-len(self.de)
        self.downlen=len(self.da)
        self.rank=self.downheight-self.upheight





class Corsort():
    def __init__(self,P):
        self.poset=P #[Node(val,n) for val in random.sample( n)]
        self.n=len(P)
        self.rank=[x.rank for x in self.poset]
        self.currentsort=self.poset
        self.currentrank=None
        self.refreshrank()

    def refreshrank(self):
        self.currentrank=[x.rank for x in self.currentsort]




    def bt(self,index1,index2):
        currentsort=self.currentsort
        node1=currentsort[index1]
        node2=currentsort[index2]
        node1.da.add(node2)
        node1.de.add(node2)
        node1.de = node1.de | node2.de
        node2.mo.add(node1)
        node2.an.add(node1)
        node2.an = node2.an | node1.an
        self.updatedown(index1,index2)
        self.updateup(index1,index2)
        self.updatesort(index1,index2)


    def updatedown(self,index1,index2):
        currentsort=self.currentsort
        node1=currentsort[index1]
        node2=currentsort[index2]
        for ancestor in node1.an:
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
        for descendant in node1.an:
            descendant.upheight=max(descendant.upheight,descendant.upheight- node1.upheight + node2.upheight+1)
            descendant.refreshnode(self.n)
        node1.upheight=max(node1.upheight,node2.upheight+1)
        node1.refreshnode(self.n)
        node2.refreshnode(self.n)
        self.refreshrank()


    def updatesort(self,index1,index2):
        currentrank=self.currentrank
        currentsort=self.currentsort
        node1=currentsort[index1]
        node2=currentsort[index2]
        k=index1
        l=index2
        while node1.rank>currentsort[k-1].rank:
            currentsort[k-1],currentsort[k]=currentsort[k],currentsort[k-1]
            k-=1
        while node2.rank>currentsort[l+1].rank:
            currentsort[l+1],currentsort[l]=currentsort[l],currentsort[l+1]
            l+=1
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

####


n1=Node(val=10,n=20)

Node.refreshnode(n1,20)
print(repr(n1))

print(n1.u)

####

#Creer une classe qui fait la liste toute seule

P1=Corsort(P=[Node(1,3),Node(0,3),Node(2,3)])

print(Corsort.corsort(P1,P=[Node(1,3),Node(0,3),Node(2,3)]))

#print(Posort.__gt__(P1,0,1))

###

random.sample(10)

