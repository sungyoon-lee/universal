import numpy as np

def proj_lp(v, xi, p):

    # Project on the lp ball centered at 0 and of radius xi

    # SUPPORTS only p = 2 and p = Inf for now
    
    # easy
    if p == 2:
        v = v * min(1, xi/np.linalg.norm(v.flatten(1)))
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
         raise ValueError('Values of p different from 2 and Inf are currently not supported...')

    return v

def defensive_perturbation(dataset, f, grads, delta=0.2, max_iter_uni = np.inf, xi=10, xi1=0.1, p=np.inf, num_classes=10, overshoot=0.02, max_iter_dp=1):
    """
    :param dataset: Images of size MxHxWxC (M: number of images)
    :param f: feedforward function (input: images, output: values of activation BEFORE softmax).
    :param grads: gradient functions with respect to input (as many gradients as classes).
    :param delta: controls the desired fooling rate (default = 80% fooling rate) P < 1 - delta
    :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = np.inf)
    :param xi: controls the l_p magnitude of the perturbation (default = 10)
    :param p: norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)
    :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
    :param max_iter_dp: maximum number of iterations for defensivepull (default = 1)
    :return: the universal perturbation.
    """


    fooling_rate = 0.0
    num_images =  np.shape(dataset)[0] # The images should be stacked ALONG FIRST DIMENSION
    u = 0 
    itr = 0
    while fooling_rate < delta and itr < max_iter_uni:
        # Shuffle the dataset
        np.random.shuffle(dataset) # according to the first dimension

        print ('Starting pass number ', itr)

        # Go through the data set and compute the pulling vector sequentially

        for k in range(0, num_images): # use a subset of data 
            cur_img = dataset[k:(k+1), :, :, :] # kth image
            
            print('>> k = ', k, ', pass #', itr)

            # Compute adversarial perturbation
            dr,iter,_,_ = defensivepull(cur_img, f, grads, num_classes=num_classes, overshoot=overshoot, max_iter=max_iter_df)
            dr = proj_lp(dr, xi1, p)
            # Make sure it converged...
            u = u + dr

        itr = itr + 1

        # Project on l_p ball
        u = u / num_images
        u = proj_lp(u, xi, p)

        # Perturb the dataset with computed perturbation
        dataset_perturbed = dataset + u

        est_labels_orig = np.zeros((num_images))
        est_labels_pert = np.zeros((num_images))
        
        # batch proccessing
        batch_size = 100
        num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size)))

        # Compute the estimated labels in batches
        for ii in range(0, num_batches):
            m = (ii * batch_size)
            M = min((ii+1)*batch_size, num_images)
            est_labels_orig[m:M] = np.argmax(f(dataset[m:M, :, :, :]), axis=1).flatten()
            est_labels_pert[m:M] = np.argmax(f(dataset_perturbed[m:M, :, :, :]), axis=1).flatten()

        # Compute the fooling rate
        fooling_rate = float(np.sum(est_labels_pert != est_labels_orig) / float(num_images))
        print('FOOLING RATE = ', fooling_rate)

    return u
    
#########################    

def defensivepull(image, f, grads, num_classes=10, overshoot=0.02, max_iter=1):

    """
       :param image: Image of size HxWx3
       :param f: feedforward function (input: images, output: values of activation BEFORE softmax).
       :param grads: gradient functions with respect to input (as many gradients as classes).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 10)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """

    f_image = np.array(f(image)).flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.shape
    pert_image = image

    f_i = np.array(f(pert_image)).flatten()
    k_i = int(np.argmax(f_i))
    #f_sort = np.sort(f_i)

    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    while loop_i < max_iter:
        d = np.inf
        pert = np.inf
        gradients = np.asarray(grads(pert_image,I))

        for k in range(1, num_classes): #TODO

            # set new w_k and new f_k
            w_k = gradients[k, :, :, :, :] - gradients[0, :, :, :, :]
            w_k = w_k/np.linalg.norm(w_k)
            f_k = f_i[I[k]] - f_i[I[0]]
            pert_k = abs(f_k)
            
            # determine the tree smallest w_k to use
            if pert_k < np.min(pert1, pert2, pert3):
                pert3 = pert2
                pert2 = pert1
                pert1 = pert_k
                w3 = w2
                w2 = w1
                w1 = w_k
                k3 = k2
                k2 = k1
                k1 = k
            elif pert_k < np.min(pert2,pert3):
                pert3 = pert2
                pert2 = pert_k
                w3 = w2
                w2 = w_k
                k3 = k2
                k2 = k
            elif pert_k < pert3:
                pert3 = pert_k
                w3 = w_k
                k3 = k
            
            choose_d_k=(pert_k-pert)/(np.dot(w1,w_k)+1)
            if choose_d_k < d:
                d = choose_d_k
                
        
        # compute r_i and r_tot
        
        r_tot = choose_d*w 
        d3=np.inf
        for k in range(1, num_classes):
            if k in {k0,k1,k2}:
                pass
            else:
                choose_d3_k = pert_k-d*np.dot(w_1,w_k)
                if choose_d3_k < d3:
                    d3 = choose_d3_k
           
        r_tot = (w1+w2)/np.linalg.norm(w1+w2)*(d3-(pert1+d))

        # compute new perturbed image
        pert_image = image + (1+overshoot)*r_tot
        loop_i += 1

        # compute new label
        f_i = np.array(f(pert_image)).flatten()
        k_i = int(np.argmax(f_i))

    r_tot = (1+overshoot)*r_tot

    return r_tot, loop_i, k_i, pert_image
