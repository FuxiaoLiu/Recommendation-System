import autograd.numpy as np
from autograd import multigrad
import time
import random

def multiply_case(U,I,index):
	return np.einsum('f,f->', U[index[0]],I[index[1]])

def cost_abs_sparse_least_square(U,I,overall_rating_train,overall_rating_train_list, overall_rating_matrix, N,num_ratings):
	least_square_error = 0
	lmd_reg = 0.05
	error_reg = 0
	UI = np.einsum('Mf,Nf -> MN',U,I)

	for key in overall_rating_train:
		user_idx = int(key[0])
		item_idx = int(key[1])
		least_square_error += (overall_rating_train[key] - UI[user_idx][item_idx])**2

	least_square_error = least_square_error/num_ratings

	error = U.flatten()
	error_reg += np.sqrt((error ** 2).mean())
	error = I.flatten()
	error_reg += np.sqrt((error ** 2).mean())

	print('squre error:' + str(least_square_error))
	return least_square_error + lmd_reg * error_reg

def cost_abs_sparse_BPR(U,I,overall_rating_train,overall_rating_train_list, overall_rating_matrix, N,num_ratings):
	least_BPR_error = 0
	mini_batch_num = 100
	pair_num = 0
	error_reg = 0
	lmd_reg = 0.05

	for kkk in range (mini_batch_num):
		[key] = random.sample(overall_rating_train_list,1)
		user_idx = int(key[0])
		item_i = int(key[1])
		item_j = int(N*random.random())

		user_item_vector = overall_rating_matrix[user_idx,:]

		if user_item_vector[item_i] < user_item_vector[item_j]:
			tmp = item_i
			item_i = item_j
			item_j = tmp

		if user_item_vector[item_i] == user_item_vector[item_j]:
			current_error = 0
		else:
			pair_num += 1
			rating_diff = np.einsum('f,f -> ',U[user_idx],(I[item_i] - I[item_j]))
			current_error = np.log(1/(1+np.exp(0 - rating_diff)))
		least_BPR_error += current_error

	least_BPR_error = least_BPR_error/pair_num

	error = U.flatten()
	error_reg += np.sqrt((error ** 2).mean())
	error = I.flatten()
	error_reg += np.sqrt((error ** 2).mean())

	print('BPR error:' + str(least_BPR_error))
	return 0 - least_BPR_error + lmd_reg * error_reg

def learn_adagrad(X, Y, A, r, r_H, U_num, I_num, F_num, num_ratings, lmdx,lmdy,lmdu,lmdh,lmdv, num_iter=100, lr=0.1, dis=False, cost_function='abs', random_seed=0, eps=1e-10):

	print("users:" + str(U_num))
	print("items:" + str(I_num))
	print("features:" + str(F_num))



	U_dim_initial = (U_num, r)
	I_dim_initial = (I_num, r)
	F_dim_initial = (F_num, r)
	H_U_dim_initial = (U_num, r_H)
	H_I_dim_initial = (I_num, r_H)

	U = np.random.rand(*U_dim_initial)
	I = np.random.rand(*I_dim_initial)
	F = np.random.rand(*F_dim_initial)
	H1 = np.random.rand(*H_U_dim_initial)
	H2 = np.random.rand(*H_I_dim_initial)

	# SGD procedure
	for i in range(num_iter):
		starttime = time.time()
		print(i+1)
		
		Nume = lmdx * np.dot(X.T,U) + lmdy * np.dot(Y.T,I)
		Deno = lmdx * np.dot(F,np.dot(U.T,U)) + lmdy * np.dot(F,np.dot(I.T,I)) + lmdv * F
		#F = F * np.sqrt(Nume/Deno)
		for xxx in range(F_num):
			for yyy in range(r):
				F[xxx][yyy] = F[xxx][yyy] * np.sqrt(Nume[xxx][yyy]/Deno[xxx][yyy])
		
		Nume = np.dot(A,I) + lmdx * np.dot(X,F)
		Deno = np.dot(np.dot(U,I.T),I) + np.dot(np.dot(H1,H2.T),I) + lmdx * np.dot(U,np.dot(F.T,F)) + lmdu * U
		#U = U * np.sqrt(Nume/Deno)
		for xxx in range(U_num):
			for yyy in range(r):
				if Deno[xxx][yyy] == 0:
					U[xxx][yyy] = 0
				else:
					U[xxx][yyy] = U[xxx][yyy] * np.sqrt(Nume[xxx][yyy]/Deno[xxx][yyy])

		Nume = np.dot(A.T,U) + lmdy * np.dot(Y,F)
		Deno = np.dot(np.dot(I,U.T),U) + np.dot(np.dot(H2,H1.T),U) + lmdy * np.dot(I,np.dot(F.T,F)) + lmdu * I
		#I = I * np.sqrt(Nume/(Deno+0.00000001))
		for xxx in range(I_num):
			for yyy in range(r):
				if Deno[xxx][yyy] == 0:
					I[xxx][yyy] = 0
				else:
					I[xxx][yyy] = I[xxx][yyy] * np.sqrt(Nume[xxx][yyy]/Deno[xxx][yyy])

		Nume = np.dot(A,H2)
		Deno = np.dot(np.dot(U,I.T),H2) + np.dot(np.dot(H1,H2.T),H2) + lmdh * H1
		#H1 = H1 * np.sqrt(Nume/Deno)
		for xxx in range(U_num):
			for yyy in range(r_H):
				if Deno[xxx][yyy] == 0:
					H1[xxx][yyy] = 0
				else:
					H1[xxx][yyy] = H1[xxx][yyy] * np.sqrt(Nume[xxx][yyy]/Deno[xxx][yyy])

		Nume = np.dot(A.T,H1)
		Deno = np.dot(np.dot(I,U.T),H1) + np.dot(np.dot(H2,H1.T),H1) + lmdh * H2
		#H2 = H2 * np.sqrt(Nume/(Deno+0.00000001))
		for xxx in range(I_num):
			for yyy in range(r_H):
				if Deno[xxx][yyy] == 0:
					H2[xxx][yyy] = 0
				else:
					H2[xxx][yyy] = H2[xxx][yyy] * np.sqrt(Nume[xxx][yyy]/Deno[xxx][yyy])

		UH = np.concatenate((U,H1),axis=1)
		IH = np.concatenate((I,H2),axis=1)

		RMSE_X = np.sqrt(((X - np.einsum('Mf,Nf->MN',U,F))**2).mean())
		RMSE_Y = np.sqrt(((Y - np.einsum('Mf,Nf->MN',I,F))**2).mean())
		RMSE_Rating = np.sqrt(((A - np.einsum('Mf,Nf->MN',UH,IH))**2).mean())
		'''
		print(Deno5[6])
		for xxx in range(I_num):
			for yyy in range(r_H):
				if Deno5[xxx][yyy] == 0:
					print(str(xxx) + ' ' + str(yyy))
		'''
		print('RMSE_X: ' + str(RMSE_X))
		print('RMSE_Y: ' + str(RMSE_Y))
		print('RMSE_Rating: ' + str(RMSE_Rating))
		#print(cost(G, H, A, T, E_np_masked, user, item_i, item_j, O, lmd1, lmd2, case))		
		nowtime = time.time()
		timeleft = (nowtime - starttime)*(num_iter-i-1) 

		if timeleft/60 > 60:
			print('time left: ' + str(int(timeleft/3600)) + ' hr ' + str(int(timeleft/60%60)) + ' min ' + str(int(timeleft%60)) + ' s')
		else:
			print("time left: " + str(int(timeleft/60)) + ' min ' + str(int(timeleft%60)) + ' s')

	return U,I,F,H1,H2
