
                   ##----------------------------------------------------------------------------------------------------------------##
                                   
                              # Calculate the different t and s diagrams associated with the ground state energy calculations #
                                           # Author: Soumi Tribedi, Anish Chakraborty, Valay Agarawal, Rahul Maitra #
                                                           # Date - 10th Dec, 2019 # 

                   ##----------------------------------------------------------------------------------------------------------------##

##--------------------------------------------## 
          #Import important modules#
##--------------------------------------------## 

import gc
import numpy as np
import copy as cp
import trans_mo
import MP2
import inp
import intermediates

##------------------------------------------------##
           #Import important parameters#
##------------------------------------------------##

nao = MP2.nao
twoelecint_mo = MP2.twoelecint_mo 
Fock_mo = MP2.Fock_mo
D1 = MP2.D1
D2=MP2.D2
So = MP2.So
Do=MP2.Do
Sv = MP2.Sv
Dv=MP2.Dv
occ = MP2.occ
virt = MP2.virt
conv = 10**(-inp.conv)

##-------------------------------------------------------------##
          #Active orbital imported from input file#
##-------------------------------------------------------------##

o_act = inp.o_act
v_act = inp.v_act
act = o_act + v_act

##--------------------------------------------------------------------##
                  #t1 and t2 contributing to R_ia#
##--------------------------------------------------------------------##

def singles(I1,I2,I_oo,I_vv,tau,t1,t2):
  R_ia = cp.deepcopy(Fock_mo[:occ,occ:nao])
  R_ia += -np.einsum('ik,ka->ia',I_oo,t1)                                          #diagrams 1,l,j,m,n
  R_ia += np.einsum('ca,ic->ia',I_vv,t1)                                           #diagrams 2,k,i
  R_ia += -2*np.einsum('ibkj,kjab->ia',twoelecint_mo[:occ,occ:nao,:occ,:occ],tau)     #diagrams 5 and a
  R_ia += np.einsum('ibkj,jkab->ia',twoelecint_mo[:occ,occ:nao,:occ,:occ],tau)     #diagrams 6 and b
  R_ia += 2*np.einsum('cdak,ikcd->ia',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],tau) #diagrams 7 and c
  R_ia += -np.einsum('cdak,ikdc->ia',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],tau) #diagrams 8 and d
  R_ia += 2*np.einsum('bj,ijab->ia',I1,t2) - np.einsum('bj,ijba->ia',I1,t2)     #diagrams e,f
  R_ia += 2*np.einsum('bj,ijab->ia',I2,t2) - np.einsum('bj,ijba->ia',I2,t2)     #diagrams g,h
  R_ia += 2*np.einsum('icak,kc->ia',twoelecint_mo[:occ,occ:nao,occ:nao,:occ],t1)           #diagram 3
  R_ia += -np.einsum('icka,kc->ia',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],t1)           #diagram 4
  return R_ia

  R_ia = None
  I_oo = None
  I_vv = None
  I1 = None
  I2 = None
  gc.collect()


##--------------------------------------------------------------------------##
                  #t2 and tau contributing to R_ijab#
##--------------------------------------------------------------------------##

def doubles(I_oo,I_vv,Ivvvv,Ioooo,Iovvo,Iovvo_2,Iovov,Iovov_2,tau,t2):
  #print " "
  R_ijab = 0.5*cp.deepcopy(twoelecint_mo[:occ,:occ,occ:nao,occ:nao])
  #R_ijab=np.zeros((occ,occ,virt,virt))
  R_ijab += -np.einsum('ik,kjab->ijab',I_oo,t2)        #diagrams linear 1 and non-linear 25,27,5,8,35,38'
  R_ijab += np.einsum('ca,ijcb->ijab',I_vv,t2)         #diagrams linear 2 and non-linear 24,26,34',6,7
  R_ijab += 0.5*np.einsum('cdab,ijcd->ijab',Ivvvv,tau) #diagrams linear 5 and non-linear 2
  R_ijab += 0.5*np.einsum('ijkl,klab->ijab',Ioooo,tau)  #diagrams linear 9 and non-linear 1,22,38
  R_ijab += 2*np.einsum('jcbk,kica->ijab',Iovvo,t2)    #diagrams linear 6 and non-linear 19,28,20
  R_ijab += - np.einsum('jcbk,ikca->ijab',Iovvo_2,t2)  #diagrams linear 8 and non-linear 21,29 
  R_ijab += - np.einsum('ickb,kjac->ijab',Iovov,t2)    #diagrams linear 10 and non-linear 23
  R_ijab += -np.einsum('icka,kjcb->ijab',Iovov_2,t2)   #diagram linear 7
  return R_ijab

  R_ijab = None
  I_oo = None
  I_vv = None
  Ivvvv = None
  Ioooo = None
  Iovvo = None
  Iovvo_2 = None
  Iovov = None
  Iovov_2 = None
  gc.collect()


##-----------------------------------------------------------------##
                #t1 terms contributing to R_ijab#
##-----------------------------------------------------------------##

def singles_n_doubles(t1,I_oovo,I_vovv):
  R_ijab = -np.einsum('ijak,kb->ijab',I_oovo,t1)       #diagrams 11,12,13,15,17
  R_ijab += np.einsum('cjab,ic->ijab',I_vovv,t1)       #diagrams 9,10,14,16,18
  R_ijab += -np.einsum('ijkb,ka->ijab',twoelecint_mo[:occ,:occ,:occ,occ:nao],t1)            #diagram 3
  R_ijab += np.einsum('cjab,ic->ijab',twoelecint_mo[occ:nao,:occ,occ:nao,occ:nao],t1)       #diagram 4
  R_ijab += -np.einsum('ickb,ka,jc->ijab',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],t1,t1)   #diagrams non-linear 3
  R_ijab += -np.einsum('icak,jc,kb->ijab',twoelecint_mo[:occ,occ:nao,occ:nao,:occ],t1,t1)   #diagrams non-linear 4
  return R_ijab

  R_ijab = None
  I_oovo = None
  I_vovv = None
  gc.collect() 


##------------------------------------------------------------------##
           #Higher orders of t1 contributing to R_ijab#
##------------------------------------------------------------------##

def higher_order(t1,t2,Iovov_3,Iovvo_3,Iooov,I3,Ioooo_2,I_voov):
  R_ijab = -np.einsum('ickb,jc,ka->ijab',Iovov_3,t1,t1)       #diagrams 36
  R_ijab += -np.einsum('jcbk,ic,ka->ijab',Iovvo_3,t1,t1)      #diagrams 32,33,31,30
  R_ijab += -np.einsum('ijlb,la->ijab',Iooov,t1)      #diagram 34
  R_ijab += -0.5*np.einsum('idal,jd,lb->ijab',I3,t1,t1)      #diagram 40
  R_ijab += np.einsum('ijkl,klab->ijab',Ioooo_2,t2)      #diagram 37
  R_ijab += -np.einsum('cjlb,ic,la->ijab',I_voov,t1,t1)      #diagram 39
  return R_ijab

  R_ijab = None 
  Iovov_3 = None 
  Iovvo_3 = None 
  Iooov = None 
  I3 = None 
  Ioooo_2 = None 
  I_voov = None
  gc.collect()

               ##---------------------------------------------------------------------------------------------------------------------------------------##
                                                                         #THE END#
               ##---------------------------------------------------------------------------------------------------------------------------------------##
