class reg(object):  
    from sklearn.kernel_ridge import KernelRidge
    import numpy as np
    import inp
    alpha = inp.alpha
    global np
    import time
    global model 
    model = KernelRidge(alpha = alpha)
    global shape1 ,shape2 , size1 , size2
    global i 
    global Amp1
    i = 0
    
    # RidgeRegression for multioutput regression
    
    
    
    def get_order_parameters_and_indices(t1,t2,thres):
        # Get shapes, will help later in unravelling
        global shape1 ,shape2 , size1 , size2
        shape1=t1.shape;shape2=t2.shape
        
        #for recontruscting the output tensor
        size1 = t1.size
        size2 = t2.size
        
        #Flatten the array for further processing, stuff works better with linear arrays
        t1=t1.flatten();t2=t2.flatten()
    
        #Get the number of order parameters in each amplitude matrix
        l1=len(t1[abs(t1)>thres]);l2=len(t2[abs(t2)>thres])
        #print('l1,l2',l1,l2)
         
        #If the number is zero, no contribution from the arrays, else get the exact indices, and value of order parameters
        if l1==0:
            ls1=[0];op1=[]
        else:
            #ls1=[0]
            #op1=[]
            op1,ls1 = reg.largest_indices(t1,l1,shape1)
        if l2==0:
            ls2=[0];op2=[]
        else:
            op2,ls2= reg.largest_indices(t2,l2,shape2)
        return np.concatenate((op1,op2)),np.array(ls1),np.array(ls2)
    
    def largest_indices(arr,n,s):
        #Get the highest absolutes values 
        indices=np.argpartition(abs(arr),-n)[-n:]
        
        #Arrange them in decreasing format, can be bypassed
        indices=indices[np.argsort(-abs(arr[indices]))]
        return arr[indices],np.transpose(np.array(np.unravel_index(indices,s)))
      
      
      
      
    def reg_train(t1 , t2 , thres):
        import time
        #saving the currently passed data to retrain later
        global i
        global shape
        global Amp1
        Amp = np.concatenate((t1.flatten(),t2.flatten()),axis = 0)
        Amp = np.reshape(Amp,(1,Amp.size))
        if i==0:
            order, ls2 , ls4 =  reg.get_order_parameters_and_indices(t1,t2,thres)
            order = np.reshape(order,(1,order.size))
            np.save('ls2',ls2)
            np.save('ls4',ls4)
        if i>0:
            ls2=np.load('ls2.npy')
            ls4=np.load('ls4.npy')
            #print(ls2.size)
            if ls2.size==1:
                op1=[]
            else:
                #print('what are you doing here?')
                op1=t1[ls2[:,0],ls2[:,1]]
            if ls4.size==1:
                op2=[]
            else:
                op2=t2[ls4[:,0],ls4[:,1],ls4[:,2],ls4[:,3]]
            order=np.concatenate((np.array(op1),np.array(op2)))
            order=order.reshape(1,order.size)
        #----------------------------------------------------------------------------
        if(i>0):
                try:
                    concatX = np.load("temp_train_data_Order.npy")
                    concatY = np.load("temp_train_data_Amp.npy")
                except Exception:
                                pass
                
        
        #concatenating the previous iter data with current data for weights update in model
        #--------------------------------------------------------------------------------
        if(i>0):
                try:
                    Amp = np.concatenate((concatY, Amp),axis = 0)
                    order = np.concatenate((concatX, order),axis = 0)
                except Exception:
                                pass
        
        #---------------------------------------------------------------------------------
        
        #fitting  the dataset into the model----------------------------------------------
        model.fit(order, Amp)
        np.save("temp_train_data_Order",order)
        np.save("temp_train_data_Amp",Amp)
        i = i+1
        
        #returns the two matrices with position of order amplitudes
        #return ls1 , ls2
      
      
    def reg_predict(X):
        
        global shape1 ,shape2 , size1 , size2
        X =np.reshape(X.flatten(),(1,X.size))
        pred = model.predict(X)

        t1_ten = pred[0,:size1].reshape(shape1)
        t2_ten = pred[0,size1:].reshape(shape2)
        #pred = np.reshape(pred,shape)
        return t1_ten , t2_ten
    
    
    def accu(X,Y):
        #calculating accuracy of original and predicted tensor, run only after reg_predict
        from sklearn.metrics import r2_score
        size = X.size
        x1 = np.reshape(X, (size))
        y1 = np.reshape(Y, (size))
        #print(r2_score(x1, y1))
        return (r2_score(x1, y1))
     
