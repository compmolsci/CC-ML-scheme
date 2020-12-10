
                   ##----------------------------------------------------------------------------------------------------------------##
                                   
                                             # Routine to calculate the ground state Coupled cluster energy #
                                                # Author: Soumi Tribedi, Anish Chakraborty, Valay Agarawal, Rahul Maitra #
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
import amplitude
#import amplitude_reduced
import diis
import cc_symmetrize
import cc_update
import time
import pandas as pd
#import reducing_cc
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
E_old = MP2.E_mp2_tot
n_iter = inp.n_iter
calc = inp.calc
conv = 10**(-inp.conv)
max_diis = inp.max_diis
nfo = MP2.nfo
nfv = MP2.nfv
basis=mol.basis
start = time.time()
eta=inp.eta
ls=np.load('ls4.npy')
##----------------------------------------------------------------------------------------##
                   #calculation of CCSD energies#
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
    print ("Total energy is : "+str(E_hf + E_ccd))
    return True
  else:
    print ("cycle number : "+str(x+1))
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
                            #Begin iteration#
##---------------------------------------------------------------------------##
#resfile=np.zeros([n_iter])
#print ('running for eta = '+str(eta)+'and number of iterations = '+str(n_iter))
for x in range(0,n_iter):

##---------------------------------------------------------------------------##
                    #Calculation for CCSD method#
##---------------------------------------------------------------------------##

  if calc == 'CCSD':
    print ("-----------CCSD------------")
    timx=time.clock()
    tau = cp.deepcopy(t2)
    tau += np.einsum('ia,jb->ijab',t1,t1) 
    
    #----------------------------------------------#
      ## See  intermediates.py file, construction of intermediates
    #----------------------------------------------#
    
    I_vv, I_oo, Ivvvv, Ioooo, Iovvo, Iovvo_2, Iovov,Iovov_2 = intermediates.initialize()
    I_oo,I_vv,Ioooo,Iovvo,Iovvo_2,Iovov = intermediates.update_int(tau,t2,I_vv,I_oo,Ioooo,Iovvo,Iovvo_2,Iovov)
    I1, I2 = intermediates.R_ia_intermediates(t1) 

    #---------------------------------------------#
      ## See  amplitudes.py file, construction of residues for singles excitation 
    #----------------------------------------------#
    R_ia = amplitude.singles(I1,I2,I_oo,I_vv,tau,t1,t2)
    
    #----------------------------------------------#
      ## See  intermediates.py file, constructon of intermediates
    #----------------------------------------------#
        
    I_oo,I_vv,I_oovo,I_vovv,Ioooo_2,I_voov,Iovov_3,Iovvo_3,Iooov,I3=intermediates.singles_intermediates(t1,t2,I_oo,I_vv,I2)


    #---------------------------------------------#
      ## See  amplitudes.py file, constuction of resides for doubles excitation
    #----------------------------------------------#
    
    R_ijab = amplitude.doubles(I_oo,I_vv,Ivvvv,Ioooo,Iovvo,Iovvo_2,Iovov,Iovov_2,tau,t2)
    R_ijab += amplitude.singles_n_doubles(t1,I_oovo,I_vovv)
    R_ijab += amplitude.higher_order(t1,t2,Iovov_3,Iovvo_3,Iooov,I3,Ioooo_2,I_voov)
    R_ijab = cc_symmetrize.symmetrize(R_ijab)

    
    #---------------------------------------------#
      ## Updating t1 and t2 with residues
    #----------------------------------------------#
    
    oldt2 = t2.copy()
    oldt1 = t1.copy()
    eps_t, t1, t2 = cc_update.update_t1t2(R_ia,R_ijab,t1,t2)
    

    ## DIIS   ##
    if inp.diis == True:
      if x+1>max_diis:
        # Limit size of DIIS vector
        if (len(diis_vals_t1) > max_diis):
          del diis_vals_t1[0]
          del diis_vals_t2[0]
          del diis_errors[0]
        diis_size = len(diis_vals_t1) - 1

        # Build error matrix B, [Pulay:1980:393], Eqn. 6, LHS
        ci = diis.error_matrix(diis_size,diis_errors)
    
        # Calculate new amplitudes
        if (x+1) % max_diis == 0:
          t1 = diis.new_amp(t1,diis_size,ci,diis_vals_t1)
          t2 = diis.new_amp(t2,diis_size,ci,diis_vals_t2)
        # End DIIS amplitude update
    # Energy calculation
    E_ccd = energy_ccsd(t1,t2)
    
    #Convergence check
    val = convergence(E_ccd,E_old,eps_t)     
    if val == True :
      break
    else:  
      E_old = E_ccd
    
    
    #    DIIS update
    if inp.diis == True: 
      # Add DIIS vectors
      error_t1, diis_vals_t1 = diis.errors(t1,oldt1,diis_vals_t1)
      error_t2, diis_vals_t2 = diis.errors(t2,oldt2,diis_vals_t2)
      # Build new error vector
      diis_errors.append(np.concatenate((error_t1,error_t2)))

end = time.time()
print ("serial code timing", (end-start))
                    ##----------------------------------------------------------------------------------------------------------------------------------------##
                                                                                           #THE END#
                    ##----------------------------------------------------------------------------------------------------------------------------------------##
