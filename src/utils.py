import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from sklearn import metrics as skmet
import torch
from tqdm import tqdm

def make_batches(x_mat,y_mat,batch_size,shuffle=True):
	if shuffle:
		shuffled_order=np.random.permutation(x_mat.shape[0])
		x_mat=x_mat[shuffled_order,:]
		y_mat=y_mat[shuffled_order,:]
	train_dataset=[]
	start=0
	while start + batch_size <= x_mat.shape[0]:
		end=min(start+batch_size,x_mat.shape[0])
		train_dataset.append((x_mat[start:end,:],
							y_mat[start:end,:]))
		start+=batch_size
	return train_dataset

def my_sim(y_batch_cp, anchor_y, factors):
	y_batch = y_batch_cp
	y_batch[np.where(y_batch == 0)] = -1
	sims = np.sum((y_batch * (anchor_y*factors)) == 1, axis=1) #/ np.sum(np.logical_or((y_batch * (anchor_y*factors)) == 1, (y_batch * (anchor_y*factors))==-1), axis=1)
	sims[np.where(np.isnan(sims))] = 0
	return sims



def select_triplets(embeddings,y_batch,max_negatives_per_pos,max_trips_per_anchor,factors,debug=False):
	triplets=[]
	for i in range(0,embeddings.shape[0]):
		anchor=embeddings[i,:]
		anchor_y=y_batch[i,:]
		# get similarity scores
		sim_scores=np.nansum(y_batch * (anchor_y*factors),axis=1)
		# sim_scores = my_sim(y_batch, anchor_y, factors)
		# get embedding distances
		dists=np.sqrt(np.sum((embeddings-anchor)**2,axis=1))
		distance_order=np.argsort(dists)
		sim_distance_order=sim_scores[distance_order]
		num_anchor_triplets=0
		positive_idcs=np.nonzero(sim_distance_order>0)[0]
		num_fine=0
		num_coarse=0
		# mine positives first, starting with the back
		for j,pos_idx in enumerate(np.flip(positive_idcs)):
			# its the anchor
			if distance_order[pos_idx]==i:
				continue
			pos_sim=sim_distance_order[pos_idx]
			# generate fine triplets
			positive_misorderings=np.logical_and(sim_distance_order[:pos_idx]<pos_sim,sim_distance_order[:pos_idx]>0)
			for neg_idx in np.nonzero(positive_misorderings)[0]:
				triplets.append((i,
								distance_order[pos_idx],
								distance_order[neg_idx]))
				num_anchor_triplets+=1
				num_fine+=1
			# generate coarse triplets
			zero_idcs=np.nonzero(sim_distance_order[:pos_idx]==0)[0]
			if len(zero_idcs)==0:
				continue
			num_negatives=np.minimum(max_negatives_per_pos,zero_idcs.shape[0])
			for _ in range(0,num_negatives):
				# choose a negative randomly, because there are a lot of negatives
				# and since as we go down the positive_idcs, the previous zero_idcs
				# is included, so we don't want to keep choosing the same negatives
				k=np.random.randint(0,len(zero_idcs))
				neg_idx=zero_idcs[k]
				triplets.append((i,
								distance_order[pos_idx],
								distance_order[neg_idx]))
				num_anchor_triplets+=1
				num_coarse+=1
				if debug:
					print((i,distance_order[pos_idx],distance_order[neg_idx]))
			
			if num_anchor_triplets>=max_trips_per_anchor:
				break
	return triplets


def get_triplets(embeddings,y_batch,max_negatives_per_pos,max_trips_per_anchor,factors,debug=False):
	trips_list=select_triplets(embeddings.detach().numpy(),y_batch,max_negatives_per_pos,max_trips_per_anchor,factors,debug)
	if len(trips_list)==0:
		return None
	anchors=[]
	positives=[]
	negatives=[]
	for (a,p,n) in trips_list:
		anchors.append(embeddings[a,:].reshape(1,embeddings.shape[1]))
		positives.append(embeddings[p,:].reshape(1,embeddings.shape[1]))
		negatives.append(embeddings[n,:].reshape(1,embeddings.shape[1]))
	anchors=torch.cat(anchors,0)
	positives=torch.cat(positives,0)
	negatives=torch.cat(negatives,0)
	return anchors,positives,negatives


def scheduled_value(epoch,list_of_vals):
	sched_val=None
	for (st,val) in list_of_vals:
		if st<=epoch:
			sched_val=val
	return sched_val


def precision_at_k(y_tst,probs_pred,k):
	assert(k>0 and int(k)==k)
	assert(y_tst.shape[0]==probs_pred.shape[0])
	
	##MMD
	probs_pred = 1 - probs_pred
	top_k=np.argsort(probs_pred,axis=1)[:,:k]
	##END
	#top_k=np.argsort(probs_pred,axis=1)[:,-k:]
	total=0
	for s_idx in range(0,y_tst.shape[0]):
		best_labels=top_k[s_idx,:]
		total+=np.nansum(y_tst[s_idx,best_labels])
	p_at_k=total/y_tst.shape[0]
	p_at_k=p_at_k/k
	return p_at_k


def compute_mlr_metrics(nbrs,num_neighbours,y_trn,emb_tst,y_tst,prefix):
	nbr_distances, nbr_indices = nbrs.kneighbors(emb_tst)
	y_nbr=y_trn[nbr_indices,:]
	assert(y_nbr.shape==(emb_tst.shape[0],num_neighbours,y_tst.shape[1]))
	y_pred=np.nanmean(y_nbr,axis=1)
	metrics_df=pd.DataFrame(index=[0])
	metrics_df.loc[0,prefix+"p@1"]=precision_at_k(y_tst,y_pred,1)
	metrics_df.loc[0,prefix+"p@3"]=precision_at_k(y_tst,y_pred,3)
	metrics_df.loc[0,prefix+"p@5"]=precision_at_k(y_tst,y_pred,5)
	# metrics_df.loc[0,prefix+"ranking_loss"]=skmet.label_ranking_loss(y_tst,y_pred)
	# metrics_df.loc[0,prefix+"coverage_error"]=skmet.coverage_error(y_tst,y_pred)
	# metrics_df.loc[0,prefix+"avg_prec_score"]=skmet.label_ranking_average_precision_score(y_tst,y_pred)
	return metrics_df


def log_epoch_metrics(logfilename,epoch,epoch_loss,model,x_trn,y_trn,x_val,y_val,num_neighbours,do_print=True):
	# get embeddings
	emb_trn=model(torch.from_numpy(x_trn.astype('float32'))).detach().numpy()
	emb_val=model(torch.from_numpy(x_val.astype('float32'))).detach().numpy()
	# create knn model
	nbrs = NearestNeighbors(n_neighbors=num_neighbours, algorithm='ball_tree').fit(emb_trn)
	# get training metrics
	trn_metrics=compute_mlr_metrics(nbrs,num_neighbours,y_trn,emb_trn,y_trn,"")
	trn_metrics.loc[0,"loss"]=epoch_loss
	trn_metrics.loc[0,"epoch"]=epoch
	trn_metrics.loc[0,"trn/val"]="trn"
	val_metrics=compute_mlr_metrics(nbrs,num_neighbours,y_trn,emb_val,y_val,"")
	val_metrics.loc[0,"loss"]=0
	val_metrics.loc[0,"epoch"]=epoch
	val_metrics.loc[0,"trn/val"]="val"
	# log metrics
	with open(logfilename,"a") as fi:
		trn_metrics.to_csv(fi,index=False)
		val_metrics.to_csv(fi,index=False)
	
	if do_print:
		print("Epoch : "+ str(epoch)+"\n")
		print(trn_metrics)
		print(val_metrics)
