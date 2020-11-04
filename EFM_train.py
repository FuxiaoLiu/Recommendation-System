#Explicit Factor Models
import math
import random
import numpy as np
import adagrad_EFM as a_E
import pickle

def load_data(file):
    print('loading data...')
    sparse_tensor = {}
    indicator_sparse_tensor = {}
    with open(file, 'r') as fin:
        for line in fin.readlines():
            entry = line.strip().split(':')
            user_idx = int(entry[1].split('\t')[0])
            item_idx = int(entry[2].split('\t')[0])
            rating_ls = entry[3].strip().split('\t')
            ratings = np.array([float(rating) for rating in rating_ls])
            indicator = (ratings != -1).astype(int)
            ratings[ratings<0] = 0
            sparse_tensor[(user_idx,item_idx)] = ratings
            indicator_sparse_tensor[(user_idx,item_idx)] = indicator
    return sparse_tensor, indicator_sparse_tensor


if __name__ == '__main__':    
	trn_sparse_t, trn_indic_t = load_data('../processed_user5item5.train')
	tst_sparse_t, tst_indic_t = load_data('../processed_user5item5.test')
	fout_result = open('../result/processed_user5item5_baseline_efm.result', 'w')

	with open('../processed_user5item5.stas','r') as fin_data_stas:
		for line in fin_data_stas.readlines():
			if line.strip().split(':')[0] == 'num_users': num_users = int(line.strip().split(':')[1])
			elif line.strip().split(':')[0] == 'num_items': num_items = int(line.strip().split(':')[1])
	data_stas = (num_users, num_items, 9)
	
	U_num = num_users
	I_num = num_items
	F_num = 9

	user_feature_attention = np.zeros((U_num,F_num))
	item_feature_quality = np.zeros((I_num,F_num))
	overall_rating_train = np.zeros((U_num,I_num))

	user_feature_attention_test = np.zeros((U_num,F_num))
	item_feature_quality_test = np.zeros((I_num,F_num))
	overall_rating_test = np.zeros((U_num,I_num))

	item_feature_mentioned = np.zeros((I_num,F_num))
	item_feature_mentioned_test = np.zeros((I_num,F_num))

	cnt_train = 0
	cnt_test = 0

	for ele in trn_sparse_t.keys():
		cnt_train += 1
		user_feature_attention[ele[0]] += trn_indic_t[ele]
		item_feature_mentioned += trn_indic_t[ele]
		item_feature_quality += trn_sparse_t[ele]

	for ele in tst_sparse_t.keys():
		cnt_train += 1
		user_feature_attention_test[ele[0]] += tst_indic_t[ele]
		item_feature_mentioned_test += trn_indic_t[ele]
		item_feature_quality_test += trn_sparse_t[ele]

	for i in range(U_num):
		for j in range(F_num):
			if user_feature_attention[i][j] != 0:
				user_feature_attention[i][j] = 1 + 4*(2/1+np.exp(0-user_feature_attention[i][j]) - 1)
			if user_feature_attention_test[i][j] != 0:
				user_feature_attention_test[i][j] = 1 + 4*(2/(1+np.exp(0-user_feature_attention_test[i][j])) - 1)

	for i in range(I_num):
		for j in range(F_num):
			if item_feature_mentioned[i][j] > 0:
				item_feature_quality[i][j] /= item_feature_mentioned[i][j]
			if item_feature_mentioned_test[i][j] > 0:
				item_feature_quality_test[i][j] /= item_feature_mentioned_test[i][j]

	print("train size:" + str(cnt_train) + '\n')
	print("test size:" + str(cnt_test) + '\n')

	r = 18
	r_H = 12
	lmdx = 0.1
	lmdy = 0.1
	lmdu = 0.05
	lmdh = 0.05
	lmdv = 0.05

	(U,I,V,H_U,H_I) = a_E.learn_adagrad(user_feature_attention, item_feature_quality, overall_rating_train, 
										r, r_H, U_num, I_num, F_num, num_ratings, lmdx,lmdy,lmdu,lmdh,lmdv,
										num_iter=3, lr=0.1, dis=False, cost_function='abs', random_seed=0, eps=1e-10)

	params = {'U':U, 'I':I, 'V':V, 'H_U':H_U, 'H_I':H_I}
	with open('../learnt_models/processed_user5item5_baseline_efm.paras','wb') as output:
        pickle.dump(params, output)  

	U_rating = np.concatenate((U,H_U),axis=1)
	I_rating = np.concatenate((I,H_I),axis=1)
	predict_rating = np.einsum('Mf,Nf -> MN', U_rating, I_rating)
	predict_user_feature_attention = np.einsum('Mf,Nf -> MN', U, V)
	predict_item_feature_quality = np.einsum('Mf,Nf -> MN', I, V)

	mean_square_error = 0
	cnt_test = 0
	for i in range(U_num):
		for j in range(I_num):
			if overall_rating_test[i][j] != 0:
				mean_square_error += (overall_rating_test[i][j] - predict_rating[i][j])**2
				cnt_test += 1

	RMSE = np.sqrt(mean_square_error/cnt_test)

	fout_result.write('num_train:' + str(num_ratings) + '\n')
	fout_result.write('num_test:' + str(cnt_test) + '\n')
	fout_result.write('RMSE:' + str(RMSE))
	
