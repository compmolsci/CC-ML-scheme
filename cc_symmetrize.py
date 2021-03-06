
                   ##----------------------------------------------------------------------------------------------------------------##
                                   
                                            # Routine to Symmetrize the two body residue of ground state CC #
                                                # Author: Valay Agarawal, Anish Chakraborty, Dipanjali Halder(?) Rahul Maitra #
                                                                  # Date - 7th July, 2020 # 

                   ##----------------------------------------------------------------------------------------------------------------##

##--------------------------------------------------##
          #Import important modules#
##--------------------------------------------------##

import gc
import numpy as np
import copy as cp
import MP2

##-------------------------------------------------------------##
       #Import number of occupied and virtual orbitals#
##-------------------------------------------------------------##

occ = MP2.occ
virt = MP2.virt

##-------------------------------------------------------------##
            #Symmetrize Residue i.e. R_ijab
##-------------------------------------------------------------##

def symmetrize(R_ijab):
    #R_ijab_new = np.zeros((occ,occ,virt,virt))
    return np.add(R_ijab, np.swapaxes(np.swapaxes(R_ijab,0,1), 2,3))
    R_ijab = None
    gc.collect()

                          ##-----------------------------------------------------------------------------------------------------------------------##
                                                                                    #THE END#
                          ##-----------------------------------------------------------------------------------------------------------------------##
