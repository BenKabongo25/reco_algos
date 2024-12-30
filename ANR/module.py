# Ben Kabongo
# December 2024

# ANR: Aspect-based Neural Recommender
# Paper: https://raihanjoty.github.io/papers/chin-et-al-cikm-18.pdf
# Source: https://github.com/almightyGOSU/ANR


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class AspectBasedRepresentationLearner(nn.Module):
	""" Aspect-based Representation Learning """

	def __init__(self, config):
		super().__init__()
		self.config = config
		self.aspects_embeddings = nn.Embedding(self.config.n_aspects, 
										 	   self.config.context_windows_size * self.config.hidden_size1)
		self.aspects_embeddings.weight.requires_grad = True
		self.aspects_proj_matrices = nn.Parameter(
			torch.Tensor(self.config.n_aspects, self.config.d_words, self.config.hidden_size1)
		)

		self.aspects_embeddings.weight.data.uniform_(-0.01, 0.01)
		self.aspects_proj_matrices.data.uniform_(-0.01, 0.01)

	def forward(self, document_embeddings: torch.Tensor) -> Tuple[torch.Tensor]:
		# document_embeddings: (batch, doc_len, d_words)

		aspects_attention = []
		aspects_representation = []
		for a in range(self.config.n_aspects):
			a_document_embeddings = torch.matmul(document_embeddings, self.aspects_proj_matrices[a]) # (batch, doc_len, hidden_size1)

			batch_size = document_embeddings.size()[0]
			aspect_embeddings = self.aspects_embeddings(
				torch.LongTensor(batch_size, 1).fill_(a).to(self.config.device)
			) 
			aspect_embeddings = torch.transpose(aspect_embeddings, 1, 2) # (batch, hidden_size1, 1)

			if self.config.context_windows_size == 1:
				aspect_attention = torch.matmul(a_document_embeddings, aspect_embeddings)
				aspect_attention = F.softmax(aspect_attention, dim=1) # (batch, doc_len, 1)

			else:
				pad_size = int((self.config.context_windows_size - 1) / 2)
				a_document_embeddings_padded = F.pad(a_document_embeddings, (0, 0, pad_size, pad_size), "constant", 0)

				a_document_embeddings_padded = a_document_embeddings_padded.unfold(1, self.config.context_windows_size, 1)
				a_document_embeddings_padded = torch.transpose(a_document_embeddings_padded, 2, 3)
				a_document_embeddings_padded = (
					a_document_embeddings_padded
					.contiguous()
					.view(-1, self.config.doc_len, self.config.context_windows_size * self.config.hidden_size1)
				) # (batch, doc_len, context_windows_size * hidden_size1)

				aspect_attention = torch.matmul(a_document_embeddings_padded, aspect_embeddings)
				aspect_attention = F.softmax(aspect_attention, dim=1) # (batch, doc_len, 1)

			aspect_representation = a_document_embeddings * aspect_attention.expand_as(a_document_embeddings)
			aspect_representation = torch.sum(aspect_representation, dim=1) # (batch, hidden_size1)

			aspects_attention.append(torch.transpose(aspect_attention, 1, 2))
			aspects_representation.append(torch.unsqueeze(aspect_representation, 1))

		aspects_attention = torch.cat(aspects_attention, dim=1) # (batch, n_aspects, doc_len)
		aspects_representation = torch.cat(aspects_representation, dim=1) # (batch, n_aspects, hidden_size1)
		return aspects_attention, aspects_representation
	

class AspectsImportanceEstimator(nn.Module):
	""" Aspects Importance Estimation """

	def __init__(self, config):
		super().__init__()
		self.config = config
		self.aspects_interaction_matrix = nn.Parameter(torch.Tensor(self.config.hidden_size1, self.config.hidden_size1))

		self.user_proj = nn.Parameter(torch.Tensor(self.config.hidden_size2, self.config.hidden_size1))
		self.user_weights = nn.Parameter(torch.Tensor(self.config.hidden_size2, 1))

		self.item_proj = nn.Parameter(torch.Tensor(self.config.hidden_size2, self.config.hidden_size1))
		self.item_weights = nn.Parameter(torch.Tensor(self.config.hidden_size2, 1))

		self.aspects_interaction_matrix.data.uniform_(-0.01, 0.01)
		self.user_proj.data.uniform_(-0.01, 0.01)
		self.user_weights.data.uniform_(-0.01, 0.01)
		self.item_proj.data.uniform_(-0.01, 0.01)
		self.item_weights.data.uniform_(-0.01, 0.01)

	def forward(self, U_aspects_document, I_aspects_document):
		# U_aspects_document: (batch, n_aspects, hidden_size1)
		# I_aspects_document: (batch, n_aspects, hidden_size1)

		U_aspects_document_t = torch.transpose(U_aspects_document, 1, 2)
		I_aspects_document_t = torch.transpose(I_aspects_document, 1, 2)

		affinity_matrix = torch.matmul(U_aspects_document, self.aspects_interaction_matrix)
		affinity_matrix = torch.matmul(affinity_matrix, I_aspects_document_t)
		affinity_matrix = F.relu(affinity_matrix)

		H_u_1 = torch.matmul(self.user_proj, U_aspects_document_t)
		H_u_2 = torch.matmul(self.item_proj, I_aspects_document_t)
		H_u_2 = torch.matmul(H_u_2, torch.transpose(affinity_matrix, 1, 2))
		H_u = H_u_1 + H_u_2
		H_u = F.relu(H_u)
		U_aspects_importance = torch.matmul(torch.transpose(self.user_weights, 0, 1), H_u)
		U_aspects_importance = torch.transpose(U_aspects_importance, 1, 2).squeeze(2)
		U_aspects_importance = F.softmax(U_aspects_importance, dim=1)  # (batch, n_aspects)

		H_i_1 = torch.matmul(self.item_proj, I_aspects_document_t)
		H_i_2 = torch.matmul(self.user_proj, U_aspects_document_t)
		H_i_2 = torch.matmul(H_i_2, affinity_matrix)
		H_i = H_i_1 + H_i_2
		H_i = F.relu(H_i)
		I_aspects_importance = torch.matmul(torch.transpose(self.item_weights, 0, 1), H_i)
		I_aspects_importance = torch.transpose(I_aspects_importance, 1, 2).squeeze(2)
		I_aspects_importance = F.softmax(I_aspects_importance, dim=1) # (batch, n_aspects)

		return U_aspects_importance, I_aspects_importance
	

class RatingPredictor(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.config = config

		self.user_dropout = nn.Dropout(self.config.dropout)
		self.item_dropout = nn.Dropout(self.config.dropout)

		if self.config.bias:
			self.Bu = nn.Parameter(torch.zeros(self.config.n_users))
			self.Bi = nn.Parameter(torch.zeros(self.config.n_items))
			self.B = nn.Parameter(torch.zeros(1))

	def forward(self, U_ids: torch.Tensor, I_ids: torch.Tensor, 
			 	U_aspects_document: torch.Tensor, I_aspects_document: torch.Tensor, 
				U_aspects_importance: torch.Tensor, I_aspects_importance: torch.Tensor) -> Dict[str, torch.Tensor]:
		# U_ids, I_ids: (batch)
		# U_aspects_document, I_aspects_document: (batch, num_aspects, hidden_size1)
		# U_aspects_importance, I_aspects_importance: (batch, num_aspects)

		U_aspects_document = self.user_dropout(U_aspects_document)
		I_aspects_document = self.item_dropout(I_aspects_document)
		U_aspects_document = torch.transpose(U_aspects_document, 0, 1)
		I_aspects_document = torch.transpose(I_aspects_document, 0, 1)

		aspects_ratings = []
		for a in range(self.config.n_aspects):
			Ua = torch.unsqueeze(U_aspects_document[a], 1)
			Ia = torch.unsqueeze(I_aspects_document[a], 2)
			a_rating = torch.matmul(Ua, Ia).squeeze(2)
			aspects_ratings.append(a_rating)
		aspects_ratings = torch.cat(aspects_ratings, dim=1) # (batch, n_aspects)

		R = U_aspects_importance * I_aspects_importance * aspects_ratings
		R = torch.sum(R, dim=1)

		if self.config.bias:
			Bu = self.Bu[U_ids]
			Bi = self.Bi[I_ids]
			R += Bu + Bi + self.B

		_out = {
			"overall_rating": R,
			"aspects_ratings": aspects_ratings
		}
		return _out
	

class RatingsLoss(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.config = config
		self.overall_rating_loss = nn.MSELoss()
		self.aspect_rating_loss = nn.MSELoss()

	def forward(self, R: torch.Tensor, R_hat: torch.Tensor,
				A_ratings: torch.Tensor=None, A_ratings_hat: torch.Tensor=None) -> Dict[str, torch.Tensor]:
		overall_rating_loss = self.overall_rating_loss(R_hat, R)
		if getattr(self.config, "aspects", None) is not None:
			aspect_rating_loss = self.aspect_rating_loss(A_ratings_hat.flatten(), A_ratings.flatten())
			total_loss = self.config.alpha * overall_rating_loss + self.config.beta * aspect_rating_loss
		else:
			aspect_rating_loss = torch.tensor(0.0).to(self.config.device)
			total_loss = overall_rating_loss
		return {"total": total_loss, "overall_rating": overall_rating_loss, "aspects_ratings": aspect_rating_loss}


class ANR(nn.Module):
	""" ANR: Aspect-based Neural Recommender """

	def __init__(self, config, 
			  	 U_documents: torch.Tensor, I_documents: torch.Tensor, words_embeddings: torch.Tensor):
		super().__init__()
		self.config = config
		
		self.user_document = nn.Parameter(U_documents, requires_grad=False)
		self.item_document = nn.Parameter(I_documents, requires_grad=False)
		self.words_embeddings = nn.Parameter(words_embeddings, requires_grad=False)

		self.aspects_representation_learner = AspectBasedRepresentationLearner(self.config)
		self.aspects_importance_estimator = AspectsImportanceEstimator(self.config)
		self.rating_predictor = RatingPredictor(self.config)

	def forward(self, U_ids: torch.Tensor, I_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
		U_documents = self.user_document[U_ids] # (batch, doc_len)
		I_documents = self.item_document[I_ids] # (batch, doc_len)
		U_documents_embeddings = self.words_embeddings[U_documents.long()] # (batch, doc_len, d_words)
		I_documents_embeddings = self.words_embeddings[I_documents.long()] # (batch, doc_len, d_words)

		U_aspects_attention, U_aspects_document = self.aspects_representation_learner(U_documents_embeddings)
		I_aspects_attention, I_aspects_document = self.aspects_representation_learner(I_documents_embeddings)
		U_co_attention, I_co_attention = self.aspects_importance_estimator(U_aspects_document, I_aspects_document)
		_out = self.rating_predictor(U_ids, I_ids, U_aspects_document, I_aspects_document, U_co_attention, I_co_attention)
		
		_out.update({
			"U_aspects_attention": U_aspects_attention,
			"I_aspects_attention": I_aspects_attention,
			"U_aspects_document": U_aspects_document,
			"I_aspects_document": I_aspects_document,
			"U_co_attention": U_co_attention,
			"I_co_attention": I_co_attention,
		})
		return _out

	def save_model(self, path):
		torch.save(self.state_dict(), path)

	def load_model(self, path):
		self.load_state_dict(torch.load(path))

