
                   ##----------------------------------------------------------------------------------------------------------------##
                                   
                                             # Routine to calculate the ground state Coupled cluster energy #
                                                # Author: Soumi Tribedi, Anish Chakraborty, Rahul Maitra #
                                                                  # Date - 10th Dec, 2019 # 

                   ##----------------------------------------------------------------------------------------------------------------##

##--------------------------------------------------##
          #Import important modules#
##--------------------------------------------------##

import numpy as np
import copy as cp
import trans_mo
import MP2
import inp
import intermediates
#import intermediates_reduced
import amplitude
#import amplitude_reduced
import diis
import cc_symmetrize
import cc_update
#import cc_update_reduced
import time
#import pandas as pd
import diagrams_reduced_with_tau
from MVM2 import reg
import os
mol = inp.mol

##------------------------------------------------------------------------##
        #Obtain the number of atomic orbitals in the basis set#
##------------------------------------------------------------------------##

nao = MP2.nao

##--------------------------------------------------##
          #Import important parameters#
##--------------------------------------------------##

E_hf = trans_mo.E_hf
Fock_mo = MP2.Fock_mo
twoelecint_mo = MP2.twoelecint_mo 
t1 = MP2.t1 
D1 = MP2.D1
t2 = MP2.t2
D2=MP2.D2
So = MP2.So
Do=MP2.Do
Sv = MP2.Sv
Dv=MP2.Dv
occ = MP2.occ
virt = MP2.virt
o=occ
v=virt
tei=twoelecint_mo
E_old = MP2.E_mp2_tot
n_iter = inp.n_iter
calc = inp.calc
conv = 10**(-inp.conv)
max_diis = inp.max_diis
nfo = MP2.nfo
nfv = MP2.nfv
alpha=inp.alpha
thres=inp.thres
discard=inp.discard
basis=mol.basis
train_size=inp.train_size
print('occ',occ,'virt',virt,'nao',nao)
#ls=inp.ls
#eta=inp.eta
##----------------------------------------------------------------------------------------##
                   #calculation of CCSD, iCCSDn and iCCSDn-PT energies#
##----------------------------------------------------------------------------------------##

def energy_ccd(t2):
  E_ccd = 2*np.einsum('ijab,ijab',t2,twoelecint_mo[:occ,:occ,occ:nao,occ:nao]) - np.einsum('ijab,ijba',t2,twoelecint_mo[:occ,:occ,occ:nao,occ:nao])
  return E_ccd

def energy_ccsd(t1,t2):
  E_ccd = energy_ccd(t2)
  E_ccd += 2*np.einsum('ijab,ia,jb',twoelecint_mo[:occ,:occ,occ:nao,occ:nao],t1,t1) - np.einsum('ijab,ib,ja',twoelecint_mo[:occ,:occ,occ:nao,occ:nao],t1,t1)
  return E_ccd

##----------------------------------------------------------------------------------------##
                  #Convergence check for the ground state iteration#
##----------------------------------------------------------------------------------------##


def convergence(E_ccd,E_old,eps):
    del_E = E_ccd - E_old
    if abs(eps) <= conv and abs(del_E) <= conv:
        print ("ccd converged!!!")
        #print ("Total energy is : "+str(E_hf + E_ccd))
        print (str(E_hf + E_ccd))
        return True
    else:
        #print ("cycle number : "+str(x+1))
        print ("change in t1 and t2 : "+str(eps))
        print ("energy difference : "+str(del_E))
        print ("energy : "+str(E_hf + E_ccd))
        return False

##---------------------------------------------------------------------------##
                            #Setup DIIS#
##---------------------------------------------------------------------------##

if inp.diis == True:
  diis_vals_t2, diis_errors_t2 = diis.DIIS_ini(t2)
  diis_errors = []
  if calc == 'CCSD' or calc == 'ICCSD' or calc == 'ICCSD-PT':
    diis_vals_t1, diis_errors_t1 = diis.DIIS_ini(t1)
  if calc == 'ICCD' or calc == 'ILCCD' or calc == 'ICCSD':
    diis_vals_So, diis_errors_So = diis.DIIS_ini(So)
    diis_vals_Sv, diis_errors_Sv = diis.DIIS_ini(Sv)

##---------------------------------------------------------------------------##
                    #Calculation for CCD method#
##---------------------------------------------------------------------------##

##---------------------------------------------------------------------------##
                    #Calculation for CCSD method#
##---------------------------------------------------------------------------##
def CCSD_full(i,t1,t2,E_old):
    
    #Same as main.py, running Exact CCSD iterations for the first few training dataset generation
    print("-----------CCSD_full, iteration number "+str(i)+" ------------")
    tau=np.add(t2,np.einsum('ia,jb->ijab',t1,t1))
    I_vv, I_oo, Ivvvv, Ioooo, Iovvo, Iovvo_2, Iovov,Iovov_2 = intermediates.initialize()
    I_oo,I_vv,Ioooo,Iovvo,Iovvo_2,Iovov = intermediates.update_int(tau,t2,I_vv,I_oo,Ioooo,Iovvo,Iovvo_2,Iovov)
    I1, I2 = intermediates.R_ia_intermediates(t1)
    R_ia = amplitude.singles(I1,I2,I_oo,I_vv,tau,t1,t2)
    I_oo,I_vv,I_oovo,I_vovv,Ioooo_2,I_voov,Iovov_3,Iovvo_3,Iooov,I3=intermediates.singles_intermediates(t1,t2,I_oo,I_vv,I2)
    R_ijab = amplitude.doubles(I_oo,I_vv,Ivvvv,Ioooo,Iovvo,Iovvo_2,Iovov,Iovov_2,tau,t2)
    R_ijab += amplitude.singles_n_doubles(t1,I_oovo,I_vovv)
    R_ijab += amplitude.higher_order(t1,t2,Iovov_3,Iovvo_3,Iooov,I3,Ioooo_2,I_voov)
    R_ijab = cc_symmetrize.symmetrize(R_ijab)
    oldt2 = t2.copy()
    oldt1 = t1.copy()
    eps_t, t1, t2 = cc_update.update_t1t2(R_ia,R_ijab,t1,t2)
    E_ccd = energy_ccsd(t1,t2)
    #print(eps_t)
    return t1,t2,E_ccd,eps_t
def CCSD_reduced(i,t1,t2,E_old,ls2, ls4):
    print ("-----------CCSD_reduced, iteration number "+str(i)+" ------------")
    tau=np.add(t2,np.einsum('ia,jb->ijab',t1,t1))
    

    #Singles residues construction, is done without intermediates
    R_ia=diagrams_reduced_with_tau.R_ia_red(t1,t2,tau,ls2,ls4)
    #Doubles residues construction, is done without intermediates
    R_ijab=diagrams_reduced_with_tau.R_ijab_red(t1,t2,tau,ls2,ls4)
    
    #Symmetrize
    R_ijab = cc_symmetrize.symmetrize(R_ijab)
    
    #updating LS t1 and t2
    eps_t,order_params=cc_update.update_t1t2_reduced(R_ia,R_ijab,t1,t2,ls2,ls4)
    
    #predict remaining
    t1,t2=reg.reg_predict(order_params)
    if ls2.size>1:
        t1[ls2[:,0],ls2[:,1]]=order_params[:len(ls2)]
    if ls4.size>1:
        t2[ls4[:,0],ls4[:,1],ls4[:,2],ls4[:,3]]=order_params[-len(ls4):]
    
    #energy calculation
    E_ccsd=energy_ccsd(t1,t2)
    return t1,t2,E_ccsd,eps_t
if calc=='CCSD':
    energy_vs_iter=[]
    start = time.time()
    #remove previous models 
    try:
        os.remove("temp_train_data_Amp.npy")
        os.remove("temp_train_data_Order.npy")
    except Exception:
        pass
    
    #run few iterations to train and discard
    for i in range(discard+train_size):
        timx=time.time()
        t1,t2,E_ccsd,eps_t=CCSD_full(i,t1,t2,E_old)
        #energy_vs_iter.append(E_ccsd)
        timy=time.time()
        print('ccsd_full',timy-timx)
        #After discarded 
        if i>discard-1:
            tim1=time.clock()
            reg.reg_train(t1,t2,thres)
            tim2=time.clock()
            print('model_train',tim2-tim1)
    ls2=np.load('ls2.npy')
    ls4=np.load('ls4.npy')
    for i in range(100):  #100 is basically a large number,
        timx=time.time()
        t1,t2,E_ccsd,eps_t= CCSD_reduced(i+1,t1,t2,E_old,ls2,ls4)
        energy_vs_iter.append(E_ccsd)
        timy=time.time()
        print('reduced ccsd',timy-timx)
        if convergence(E_ccsd,E_old,eps_t):
            break
        else:
            E_old=E_ccsd

end = time.time()
print("{:.6f}".format(end-start))
##----------------------------------------------------------------------------------------------------------------------------------------##
                                                                                           #THE END#
                    ##----------------------------------------------------------------------------------------------------------------------------------------##
