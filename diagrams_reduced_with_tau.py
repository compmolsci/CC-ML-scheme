## File to construct the reduced residues of the coupled cluster equation
## Author: Valay Agarawal, Anish Chakraborty, Rahul Maitra
## Data: 15th Sept, 2020

import numpy as np
import copy as cp
import trans_mo
import MP2
import inp

#importing stuff

mapping=inp.mapping
occ=MP2.occ
virt=MP2.virt
o=occ
v=virt
twoelecint=MP2.twoelecint_mo
Fock=MP2.Fock_mo
#print(Fock.shape)
mappping='f2l'#inp.mapping
import time
#l2=np.load('ls2.npy')  # Largest subset in sinlges excitation
#l4=np.load('ls4.npy')  # Largest subset in doubles excitation


#Doubles residue construction. All done in one go, no intermediates. Some diagrams are clubbed together. 

def R_ia_red(t1,t2,tau,l2,l4):
    #Linear Terms
    R_ia=np.zeros((o,v))
    if l2.size>1:
        if mapping=='f2l':
            for tt2 in l2:
                i=tt2[0];a=tt2[1]
                R_ia[i,a]=Fock[i,o+a]
                R_ia[i,a]+= -np.einsum('k,k',Fock[i,:o],t1[:,a]) ## Diagram 1, ik,ka->ia
                R_ia[i,a]+= np.einsum('c,c',Fock[-v:,o+a],t1[i,:]) ## Diagram 2, ca,ic->ia
                R_ia[i,a]+= 2*np.einsum('ck,kc',twoelecint[i,-v:,o+a,:o],t1) ##Diagram 3, icak,kc->ia 
                R_ia[i,a]+= -np.einsum('ck,kc',twoelecint[i,-v:,:o,o+a],t1) ##Diagram 4, icka,kc->ia
                #Involving T2
                R_ia[i,a]+= -2*np.einsum('bkj,kjb',twoelecint[i,-v:,:o,:o],tau[:,:,a,:]) ##Diagram (5,(a)), ickl,klac->ia
                R_ia[i,a]+=  np.einsum('bkj,jkb',twoelecint[i,-v:,:o,:o],tau[:,:,a,:])##Diagram 6, ickl,klca->ia
                R_ia[i,a]+=  2*np.einsum('cdk,kcd',twoelecint[-v:,-v:,o+a,:o],tau[i,:,:,:])##Diagram (7,(c)), dcka,ikcd->ia
                R_ia[i,a]+=  -np.einsum('cdk,kdc',twoelecint[-v:,-v:,o+a,:o],tau[i,:,:,:])##Diagram (8,(d)), dcak,ikcd->ia
                #Involving T1, T1
                #R_ia[i,a]+=  -2*np.einsum('ckl,k,lc',twoelecint[i,-v:,:o,:o],t1[:,a],t1)##Diagram (a) ickl,ka,lc->ia
                #R_ia[i,a]+=  np.einsum('ckl,kc,l',twoelecint[i,-v:,:o,:o],t1,t1[:,a])##Diagram (b) ickl,kc,la
                #R_ia[i,a]+=  2*np.einsum('cdk,c,kd',twoelecint[-v:,-v:,o+a,:o],t1[i,:],t1)##Diagram (c) cdak,ic,kd
                #R_ia[i,a]+=  -np.einsum('cdk,c,kd',twoelecint[-v:,-v:,:o,o+a],t1[i,:],t1)##Diagram (d) cdka,ic,kd->ijab 
                #Involving T1, T2
                R_ia[i,a]+= 4*np.einsum('cbkj,kc,jb',twoelecint[-v:,-v:,:o,:o],t1,t2[i,:,a,:]) ##Diagram (e) cdkl,ld,kica->ia 
                R_ia[i,a]+= -2*np.einsum('cbkj,kc,jb',twoelecint[-v:,-v:,:o,:o],t1,t2[i,:,:,a]) ##Diagram (f) cdkl,ld,ikca->ia
                R_ia[i,a]+= np.einsum('cbjk,kc,jb',twoelecint[-v:,-v:,:o,:o],t1,t2[i,:,:,a]) ##Diagram (g) cdlk,ld,ikac
                R_ia[i,a]+= -2*np.einsum('cbjk,kc,jb',twoelecint[-v:, -v:,:o,:o],t1,t2[i,:,a,:]) ##Diagram (h) cdkl,kd,ilca
                R_ia[i,a]+= np.einsum('cdkl,kld,c',twoelecint[-v:,-v:,:o,:o],t2[:,:,:,a],t1[i,:]) ##Diagram (i) cdkl, ic,klda->ia
                #R_ia[i,a]+= np.einsum('cdkl,l,kcd',twoelecint[-v:,-v:,:o,:o],t1[:,a],t2[i,:,:,:]) ##Diagram (j) cdkl, la,ikcd->ia
                R_ia[i,a]+= np.einsum('dckl,ldc,k',twoelecint[-v:,-v:,:o,:o],tau[:,i,:,:],t1[:,a]) ## Diagram ((j),(m)) cdkl, la,ikcd->ia
                R_ia[i,a]+= -2*np.einsum('cdkl,kld,c',twoelecint[-v:,-v:,:o,:o],t2[:,:,a,:],t1[i,:]) ##Diagram (k) cdkl,ic,klad->ia
                #R_ia[i,a]+= -2*np.einsum('cdkl,k,ldc',twoelecint[-v:,-v:,:o,:o],t1[:,a],t2[:,i,:,:]) ##Diagram (l) cdkl,ka,lidc->ia
                R_ia[i,a]+= -2*np.einsum('cdkl,lcd,k',twoelecint[-v:,-v:,:o,:o],tau[i,:,:,:],t1[:,a]) ##Diagram ((l),(n)) cdkl,ka,lidc->ia
                #Involving T1, T1, T1
                #R_ia[i,a]=  np.einsum('cdkl,c,kd,l',twoelecint[-v:,-v:,:o,:o],t1[i,:],t1,t1[:,a]) ##Diagram (m) cdkl,ic,kd,la->ia
                #R_ia[i,a]=  -2*np.einsum('cdkl,ld,c,k',twoelecint[-v:,-v:,:o,:o],t1,t1[i,:],t1[:,a]) ##Diagram (n) cdkl,ld,ic,ka->ia

            return R_ia
        elif mapping=='l2l':
            pass
    else:
        return R_ia




#Doubles residue construction. All done in one go, no intermediates. Some diagrams are clubbed together. 

def R_ijab_red(t1,t2,tau,l2,l4):
    R_ijab=np.zeros((o,o,v,v))
    if l4.size>1:
        if mapping=='f2l':
            for t4 in l4:
                i=t4[0];j=t4[1];a=t4[2];b=t4[3];
                R_ijab[i,j,a,b]+=0.5*twoelecint[i,j,o+a,o+b]
                #Involving T1
                R_ijab[i,j,a,b] += -np.einsum('k,k',twoelecint[i,j,:o,o+b],t1[:,a])## Diagram 3, ijkb,ka->ijab #singles_n_doubles
                R_ijab[i,j,a,b] +=  np.einsum('c,c',twoelecint[-v:,j,o+a,o+b],t1[i,:])## Diagram 4, cjab,ic->ijab #singles_n_doublees
                # Involving T2
                R_ijab[i,j,a,b] += -np.einsum('k,k',Fock[i,:o],t2[:,j,a,b])## Diagram 1, ik,kjab->ijab #doubles
                #print(R_ijab[i,j,a,b])
                R_ijab[i,j,a,b] +=  np.einsum('c,c',Fock[-v:,o+a],t2[i,j,:,b])## Diagram 2, ca,ijcb->ijab #doubles
                #R_ijab[i,j,a,b] +=  0.5*np.einsum('cd,cd',twoelecint[-v:,-v:,o+a,o+b],tau[i,j,:,:])## Diagram 5, cdab,ijcd->ijab #doubles
                R_ijab[i,j,a,b] +=  0.5*np.einsum('cd,cd',twoelecint[-v:,-v:,o+a,o+b],tau[i,j,:,:])## Diagram (L5,NL2), cdab,ijcd->ijab #doubles  
                R_ijab[i,j,a,b] += 2*np.einsum('ck,kc',twoelecint[j,-v:,o+b,:o],t2[:,i,:,a])#diagrams 6  jcbk,kica->ijab linear 6 
                R_ijab[i,j,a,b] +=  -np.einsum('ck,kc',twoelecint[i,-v:,:o,o+a],t2[:,j,:,b])## Diagram 7, icka,kjcb->ijab #doubles
                R_ijab[i,j,a,b] +=  -np.einsum('ck,kc',twoelecint[j,-v:,o+b,:o],t2[i,:,:,a])## Diagram 8, jckb,ikca->ijab #doubles
                #R_ijab[i,j,a,b] +=  -np.einsum('ck,kc',twoelecint[i,-v:,o+a,:o],tau[j,:,:,b])## Diagram (L8,NL4), icak,jkcb->ijab #doubles 
                #R_ijab[i,j,a,b] +=  0.5*np.einsum('kl,kl',twoelecint[i,j,:o,:o],t2[:,:,a,b])## Diagram 9, ijkl,klab->ijab #doubles
                R_ijab[i,j,a,b] +=  0.5*np.einsum('kl,kl',twoelecint[i,j,:o,:o],tau[:,:,a,b])## Diagram (L9,NL1), ijkl,klab->ijab 
                R_ijab[i,j,a,b] +=  -np.einsum('ck,kc',twoelecint[i,-v:,:o,o+b],t2[:,j,a,:])## Diagram 10, ickb,kjac->ijab #doubles
                #R_ijab[i,j,a,b] +=  -np.einsum('ck,kc',twoelecint[i,-v:,:o,o+b],tau[:,j,a,:])## Diagram (L10,NL3), ickb,kjac->ijab
                #Involving T1, T1
                #R_ijab[i,j,a,b]+= 0.5*np.einsum('kl,k,l', twoelecint[i,j,:o,:o],t1[:,a],t1[:,b])#Diagram NL1 ijkl,ka,lb->ijab #doubles
                #R_ijab[i,j,a,b]+= 0.5*np.einsum('cd,c,d', twoelecint[-v:,-v:,o+a,o+b],t1[i,:],t1[j,:])#Diagram NL2 cdab,ic,jd->ijab #doubles
                R_ijab[i,j,a,b]+= -np.einsum('ck,k,c', twoelecint[i,-v:,:o,o+b],t1[:,a],t1[j,:])#Diagram NL3 icka,ka,jb->ijab #singles_n_doubles
                R_ijab[i,j,a,b]+= -np.einsum('ck,c,k', twoelecint[i,-v:,o+a,:o],t1[j,:],t1[:,b])#Diagram NL4 icak,jc,kb->ijab #singles_n_doubles
                #Involving T1, T2
                R_ijab[i,j,a,b]+= -2*np.einsum('clk,kc,l', twoelecint[i,-v:,:o,:o],t1,t2[:,j,a,b])#Diagram NL5 iclk,kc,ljab->ijab  #doubles 
                #print(R_ijab[i,j,a,b])
                R_ijab[i,j,a,b]+= 2*np.einsum('dck,kc,d', twoelecint[-v:,-v:,o+a,:o],t1,t2[i,j,:,b])#Diagram NL6 dcak,kc,ijdb->ijab  #doubles
                R_ijab[i,j,a,b]+= -np.einsum('dcl,ld,c', twoelecint[-v:,-v:,o+a,:o],t1,t2[i,j,:,b])#Diagram NL7 dcal,ld,ijcb->ijab  #doubles
                R_ijab[i,j,a,b]+= np.einsum('ckl,kc,l', twoelecint[i,-v:,:o,:o],t1,t2[:,j,a,b])#Diagram NL8 ickl,kc,ljab->ijab #doubles
                #print(R_ijab[i,j,a,b])
                R_ijab[i,j,a,b]+= np.einsum('ckl,kl,c', twoelecint[-v:,j,:o,:o],t2[:,:,a,b],t1[i,:])#Diagram NL9 cjkl,klab,ic->ijab  #singles_n_doubles
                R_ijab[i,j,a,b]+= -np.einsum('cdl,ld,c', twoelecint[-v:,-v:,:o,o+b],t2[:,j,a,:],t1[i,:])#Diagram NL10 cdlb,ljad,ic->ijab  #singles_n_doubles
                #R_ijab[i,j,a,b]+= np.einsum('ckl,lc,k', twoelecint[-v:,i,:o,:o],t2[j,:,:,a],t1[:,b])#Diagram NL11 cikl,jlca,kb->ijab  #singles_n_doubles
                R_ijab[i,j,a,b]+= np.einsum('ckl,lc,k', twoelecint[-v:,i,:o,:o],tau[j,:,:,a],t1[:,b])#Diagram NL11 cikl,jlca,kb->ijab  #singles_n_doubles
                R_ijab[i,j,a,b]+= -np.einsum('cdk,cd,k', twoelecint[-v:,-v:,:o,o+a],t2[j,i,:,:],t1[:,b])#Diagram NL12 cdka,jicd,kb->ijab  #singles_n_doubles
                R_ijab[i,j,a,b]+= np.einsum('clk,lc,k', twoelecint[j,-v:,:o,:o],t2[:,i,:,a],t1[:,b])#Diagram NL13 jclk,lica,kb->ijab  #singles_n_doubles
                R_ijab[i,j,a,b]+= -np.einsum('cdk,kd,c', twoelecint[-v:,-v:,:o,o+a],t2[:,j,:,b],t1[i,:])#Diagram NL14 cdka,kjdb,ic->ijab  #singles_n_doubles
                R_ijab[i,j,a,b]+= -2*np.einsum('ckl,lc,k', twoelecint[j,-v:,:o,:o],t2[i,:,a,:],t1[:,b])#Diagram NL15 jckl,ilac,kb->ijab  #singles_n_doubles
                R_ijab[i,j,a,b]+= 2*np.einsum('cdl,ld,c', twoelecint[-v:,-v:,o+a,:o],t2[:,j,:,b],t1[i,:])#Diagram NL16 cdal,ljdb,ic->ijab  #singles_n_doubles
                R_ijab[i,j,a,b]+= np.einsum('ckl,lc,k', twoelecint[j,-v:,:o,:o],t2[i,:,:,a],t1[:,b])#Diagram NL17 jckl,ilca,kb->ijab  #singles_n_doubles
                #R_ijab[i,j,a,b]+= np.einsum('ckl,lc,k', twoelecint[j,-v:,:o,:o],tau[i,:,:,a],t1[:,b])#Diagram NL17 jckl,ilca,kb->ijab  #singles_n_doubles
                #R_ijab[i,j,a,b]+= -np.einsum('cdl,ld,c', twoelecint[-v:,-v:,o+a,:o],t2[j,:,:,b],t1[i,:])#Diagram NL18 cdal,jldb,ic->ijab  #singles_n_doubles
                R_ijab[i,j,a,b]+= -np.einsum('cdl,ld,c', twoelecint[-v:,-v:,o+a,:o],tau[j,:,:,b],t1[i,:])#Diagram NL18,NL31 cdal,jldb,ic->ijab  #singles_n_doubles
                #Involving T2, T2
                R_ijab[i,j,a,b]+= 2*np.einsum('dclk,ld,kc', twoelecint[-v:,-v:,:o,:o],t2[j,:,b,:],t2[:,i,:,a])#Diagram NL19 dclk,jlbd,kica->ijab #doubles
                R_ijab[i,j,a,b]+= -1*np.einsum('cdlk,ld,kc', twoelecint[-v:,-v:,:o,:o],t2[j,:,b,:],t2[:,i,:,a])#Diagram NL20 cdlk,jlbd,kica->ijab #doubles
                R_ijab[i,j,a,b]+= 0.5*np.einsum('cdkl,kc,ld', twoelecint[-v:,-v:,:o,:o],t2[i,:,:,a],t2[j,:,:,b])#Diagram NL21 cdkl,ikca,jldb->ijab #doubles
                R_ijab[i,j,a,b]+= 0.5*np.einsum('cdkl,cd,kl', twoelecint[-v:,-v:,:o,:o],t2[i,j,:,:],tau[:,:,a,b])#Diagram NL22 cdkl,ijcd,klab->ijab #doubles
                R_ijab[i,j,a,b]+= 0.5*np.einsum('dckl,ld,kc', twoelecint[-v:,-v:,:o,:o],t2[i,:,:,b],t2[:,j,a,:])#Diagram NL23 dckl,ildb,kjac->ijab #doubles
                #R_ijab[i,j,a,b]+= -2*np.einsum('cdkl,kld,c', twoelecint[-v:,-v:,:o,:o],t2[:,:,a,:], t2[i,j,:,b])#Diagram NL24 cdlk,klad,ijcb->ijab #doubles
                R_ijab[i,j,a,b]+= -2*np.einsum('cdkl,kld,c', twoelecint[-v:,-v:,:o,:o],tau[:,:,a,:],t2[i,j,:,b])#Diagram (NL24,NL34') dclk,ijad,lkbc->ijab
                R_ijab[i,j,a,b]+= -2*np.einsum('cdkl,lcd,k', twoelecint[-v:,-v:,:o,:o],tau[i,:,:,:],t2[:,j,a,b])#Diagram NL25,NL38' cdkl,ilcd,kjab->ijab #doubles
                #R_ijab[i,j,a,b]+= np.einsum('cdkl,kld,c', twoelecint[-v:,-v:,:o,:o],t2[:,:,:,a],t2[i,j,:,b])#Diagram NL26 cdkl,klda,ijcb->ijab #doubles
                R_ijab[i,j,a,b]+= np.einsum('cdkl,kld,c', twoelecint[-v:,-v:,:o,:o],tau[:,:,:,a],t2[i,j,:,b])#Diagram NL26 cdkl,klda,ijcb->ijab #doubles
                R_ijab[i,j,a,b]+= np.einsum('dckl,ldc,k', twoelecint[-v:,-v:,:o,:o],tau[:,i,:,:],t2[:,j,a,b])# Diagram NL27,NL35 (dckl,lidc,kjab->ijab
                #R_ijab[i,j,a,b]+= -2*np.einsum('dclk,ld,kc', twoelecint[-v:,-v:,:o,:o],t2[j,:,:,b],t2[:,i,:,a])#Diagram NL28 dclk,jldb,kica->ijab #doubles
                R_ijab[i,j,a,b]+= -2*np.einsum('dclk,ld,kc', twoelecint[-v:,-v:,:o,:o],tau[i,:,:,a],t2[j,:,b,:])#Diagram (NL28,NL32) dclk,ilda,jkbc->ijab
                #R_ijab[i,j,a,b]+= np.einsum('cdkl,lc,kd', twoelecint[-v:,-v:,:o,:o],t2[i,:,:,a],t2[j,:,b,:])#Diagram NL29 cdkl,ilca,jkbd->ijab #doubles
                R_ijab[i,j,a,b]+= np.einsum('cdkl,lc,kd', twoelecint[-v:,-v:,:o,:o],tau[i,:,:,a],t2[j,:,b,:])#Diagram (NL29,NL39) cdkl,ilca,jkbd->ijab
                #Involving T1, T1, T1
                #R_ijab[i,j,a,b]+= np.einsum('clk,l,c,k', twoelecint[j,-v:,:o,:o],t1[:,b],t1[i,:],t1[:,a])#Diagram NL30 jclk,lb,ic,ka->ijab #higher_order
                #R_ijab[i,j,a,b]+= -np.einsum('dck,d,c,k', twoelecint[-v:,-v:,o+b,:o],t1[j,:],t1[i,:],t1[:,a])#Diagram NL31 dcbk,jd,ic,ka->ijab #higher_order      
                #Involving T1, T1, T2
                #R_ijab[i,j,a,b]+= -2*np.einsum('dclk,ld,c,k', twoelecint[-v:,-v:,:o,:o],t2[j,:,b,:], t1[i,:],t1[:,a])#Diagram NL32 dclk,jlbd,ic,ka->ijab #higher_order 
                R_ijab[i,j,a,b]+= np.einsum('dclk,ld,c,k', twoelecint[-v:,-v:,:o,:o],t2[j,:,:,b],t1[i,:],t1[:,a])#Diagram NL33 dclk,jldb,lc,ka->ijab #higher_order 
                #R_ijab[i,j,a,b]+= np.einsum('cdlk,kc,d,l', twoelecint[-v:,-v:,:o,:o],t1,t2[i,j,:,b],t1[:,a])#Diagram NL34 cdlk,kc,ijdb,la->ijab #higher_order 
                #R_ijab[i,j,a,b]+= np.einsum('dckl,ldc,k', twoelecint[-v:,-v:,:o,:o],t1[j,:],t1,t2[:,i,b,a])#Diagram NL35 cdlk,jc,ld,kiba->ijab #doubles 
                #print(R_ijab[i,j,a,b])
                R_ijab[i,j,a,b]+= np.einsum('dckl,ld,c,k', twoelecint[-v:,-v:,:o,:o],t2[i,:,:,b],t1[j,:],t1[:,a])#Diagram NL36 dckl,ildb,jc,ka->ijab #higher_order
                R_ijab[i,j,a,b]+= 0.5*np.einsum('cdkl,c,d,kl', twoelecint[-v:,-v:,:o,:o],t1[i,:],t1[j,:],tau[:,:,a,b])#Diagram NL37 cdkl,ic,jd,lkab->ijab #higher_order 
                #R_ijab[i,j,a,b]+= 0.5*np.einsum('cdkl,k,l,cd', twoelecint[-v:,-v:,:o,:o],t1[:,a],t1[:,b],t2[i,j,:,:])#Diagram NL38 cdkl,ka,lb,ijcd->ijab #doubles
                #R_ijab[i,j,a,b]+= np.einsum('cdkl,kd,c,l', twoelecint[-v:,-v:,:o,:o],t2[:,j,:,b],t1[i,:],t1[:,a])#Diagram NL39 cdkl,kjdb,ic,la->ijab #higher_order 
                #R_ijab[i,j,a,b]+= -2*np.einsum('cdkl,ld,c,k', twoelecint[-v:,-v:,:o,:o],t1,t1[i,:],t2[j,:,b,a])#Diagram NL38' cdkl,ld,ic,jkba->ijab #doubles
                #print(R_ijab[i,j,a,b])
                #R_ijab[i,j,a,b]+= -2*np.einsum('dclk,ld,k,c', twoelecint[-v:,-v:,:o,:o],t1,t1[:,a],t2[i,j,:,b])#Diagram NL34' #dclk,ld,ka,ijcb doubles
                #Involving T1, T1, T1, T1
                #R_ijab[i,j,a,b]+= 0.5*np.einsum('cdkl,c,k,d,l', twoelecint[-v:,-v:,:o,:o],t1[i,:],t1[:,a],t1[j,:],t1[:,b])#Diagram NL40 #higher_order
            return R_ijab
        elif mapping=='l2l':
            pass
    else:
        return R_ijab
