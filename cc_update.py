
                   ##----------------------------------------------------------------------------------------------------------------##
                                   
                                            # Routine to Symmetrize the two body residue of ground state CC #
                                                # Author: Soumi Tribedi, Anish Chakraborty, Rahul Maitra #
                                                                  # Date - 10th Dec, 2019 # 

                   ##----------------------------------------------------------------------------------------------------------------##

##--------------------------------------------------##
          #Import important modules#
##--------------------------------------------------##

import gc
import numpy as np
import copy as cp
import MP2
import inp
import amplitude
import intermediates

##--------------------------------------------------##
          #import important parameters#
##--------------------------------------------------##

D1 = MP2.D1
D2=MP2.D2
So = MP2.So
Do=MP2.Do
Sv = MP2.Sv
Dv=MP2.Dv
conv = 10**(-inp.conv)
eta = inp.eta
occ=MP2.occ
virt=MP2.virt

##--------------------------------------------------##
              #compute new t2#
##--------------------------------------------------##

def update_t2(R_ijab,t2):
    ntmax = 0
    eps_t = 100
    #if eps_t >= conv:
    del_t=np.divide(R_ijab,D2)
    t2= np.add(np.divide(R_ijab,D2),t2)
    eps_t = np.average(np.abs(R_ijab))
    return eps_t, t2, R_ijab

##--------------------------------------------------##
              #compute new t1 and t2#
##--------------------------------------------------##

def update_t1t2(R_ia,R_ijab,t1,t2):
    #ntmax = 0
    #eps = 100
    delt2 = np.divide(R_ijab,(D2+eta))
    delt1 = np.divide(R_ia,(D1+eta))
    t1 = t1 + delt1
    t2 = t2 + delt2
    ntmax = np.size(t1)+np.size(t2)
    eps = float(np.sum(abs(R_ia))+np.sum(abs(R_ijab)))/ntmax
    print('eps',eps)
    return eps, t1, t2

def update_t1t2_reduced(R_ia, R_ijab,t1,t2,ls2,ls4):
    #delt2=np.divide(R_ijab,D2)
    #delt1=np.divide(R_ia,D1) 
    order_params=[]
    res=[]
    res=np.array(res)
    order_params=np.array(order_params)
    if ls2.size>1:
        t1_red = np.array(t1[ls2[:,0],ls2[:,1]])
        R_ia_red = np.array(R_ia[ls2[:,0],ls2[:,1]])
        res=np.concatenate((res,R_ia_red))
        D1_red = np.array(D1[ls2[:,0],ls2[:,1]])
        t1_red=np.add(t1_red,np.divide(R_ia_red,D1_red))
        order_params=np.concatenate((order_params,t1_red))
    if ls4.size>1:
        t2_red = np.array(t2[ls4[:,0],ls4[:,1],ls4[:,2],ls4[:,3]])
        R_ijab_red = np.array(R_ijab[ls4[:,0],ls4[:,1],ls4[:,2],ls4[:,3]])
        res=np.concatenate((res,R_ijab_red))
        D2_red = np.array(D2[ls4[:,0],ls4[:,1],ls4[:,2],ls4[:,3]])
        t2_red=np.add(t2_red,np.divide(R_ijab_red,D2_red))
        order_params=np.concatenate((order_params,t2_red))
    
    #t1=np.add(np.multiply(t1,R_ia!=0),np.multiply(delt1,R_ia!=0))
    #t2=np.add(np.multiply(t2,R_ijab!=0),np.multiply(delt2,R_ijab!=0))
    #if len(
    #t2=np.add(t2,delt2)
    #t1=np.add(t1,delt1)
    #t2=np.add(t1,delt2)
    eps=np.average(np.abs(res))
    #print(res)
    print('eps',eps)
    #print('eps',eps)
    #print(eps)
    return eps,order_params
##--------------------------------------------------##
                #compute new So#
##--------------------------------------------------##

def update_So(R_ijav,So):
  ntmax = 0
  eps_So = 100
  if eps_So >= conv:
    delSo = np.divide(R_ijav,(Do+eta))
    So = So + delSo
  ntmax = np.size(So)
  eps_So = float(np.sum(abs(R_ijav))/ntmax)
  return eps_So, So

##--------------------------------------------------##
                #compute new Sv#
##--------------------------------------------------##

def update_Sv(R_iuab,Sv):
  ntmax = 0
  eps_Sv = 100
  if eps_Sv >= conv:
    delSv = np.divide(R_iuab,(Dv+eta))
    Sv = Sv + delSv
  ntmax = np.size(Sv)
  eps_Sv = float(np.sum(abs(R_iuab))/ntmax)
  return eps_Sv, Sv

                          ##-----------------------------------------------------------------------------------------------------------------------##
                                                                                    #THE END#
                          ##-----------------------------------------------------------------------------------------------------------------------##
