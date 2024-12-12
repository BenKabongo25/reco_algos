# Ben Kabongo
# December 2024

# NCF: Neural Collaborative Filtering
# Doc: https://cornac.readthedocs.io/en/stable/api_ref/models.html#module-cornac.models.ncf.recom_neumf
# Sigmoid suppression
# Pytorch implementation


import torch
import torch.nn as nn

import numpy as np
from tqdm.auto import trange

from cornac.models import Recommender
from cornac.utils import get_rng
from cornac.exception import ScoreException
from cornac.metrics import RMSE
from cornac.eval_methods import rating_eval


optimizer_dict = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "rmsprop": torch.optim.RMSprop,
    "adagrad": torch.optim.Adagrad,
}

activation_functions = {
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "elu": nn.ELU(),
    "selu": nn.SELU(),
    "relu": nn.ReLU(),
    "relu6": nn.ReLU6(),
    "leakyrelu": nn.LeakyReLU(),
}


class GMF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_factors: int = 8,
    ):
        super(GMF, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.user_embedding = nn.Embedding(num_users, num_factors)
        self.item_embedding = nn.Embedding(num_items, num_factors)

        self.logit = nn.Linear(num_factors, 1)

        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.user_embedding.weight, std=1e-2)
        nn.init.normal_(self.item_embedding.weight, std=1e-2)
        nn.init.normal_(self.logit.weight, std=1e-2)

    def from_pretrained(self, pretrained_gmf):
        self.user_embedding.weight.data.copy_(pretrained_gmf.user_embedding.weight)
        self.item_embedding.weight.data.copy_(pretrained_gmf.item_embedding.weight)
        self.logit.weight.data.copy_(pretrained_gmf.logit.weight)
        self.logit.bias.data.copy_(pretrained_gmf.logit.bias)

    def h(self, users, items):
        return self.user_embedding(users) * self.item_embedding(items)

    def forward(self, users, items):
        h = self.h(users, items)
        output = self.logit(h).view(-1)
        return output


class MLP(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        layers=(64, 32, 16, 8),
        act_fn="relu",
    ):
        super(MLP, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.user_embedding = nn.Embedding(num_users, layers[0] // 2)
        self.item_embedding = nn.Embedding(num_items, layers[0] // 2)

        mlp_layers = []
        for idx, factor in enumerate(layers[:-1]):
            mlp_layers.append(nn.Linear(factor, layers[idx + 1]))
            mlp_layers.append(activation_functions[act_fn.lower()])

        # unpacking layers in to torch.nn.Sequential
        self.mlp_model = nn.Sequential(*mlp_layers)

        self.logit = nn.Linear(layers[-1], 1)

        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.user_embedding.weight, std=1e-2)
        nn.init.normal_(self.item_embedding.weight, std=1e-2)
        for layer in self.mlp_model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.normal_(self.logit.weight, std=1e-2)

    def from_pretrained(self, pretrained_mlp):
        self.user_embedding.weight.data.copy_(pretrained_mlp.user_embedding.weight)
        self.item_embedding.weight.data.copy_(pretrained_mlp.item_embedding.weight)
        for layer, pretrained_layer in zip(self.mlp_model, pretrained_mlp.mlp_model):
            if isinstance(layer, nn.Linear) and isinstance(pretrained_layer, nn.Linear):
                layer.weight.data.copy_(pretrained_layer.weight)
                layer.bias.data.copy_(pretrained_layer.bias)
        self.logit.weight.data.copy_(pretrained_mlp.logit.weight)
        self.logit.bias.data.copy_(pretrained_mlp.logit.bias)

    def h(self, users, items):
        embed_user = self.user_embedding(users)
        embed_item = self.item_embedding(items)
        embed_input = torch.cat((embed_user, embed_item), dim=-1)
        return self.mlp_model(embed_input)

    def forward(self, users, items):
        h = self.h(users, items)
        output = self.logit(h).view(-1)
        return output

    def __call__(self, *args):
        return self.forward(*args)


class NeuMF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_factors: int = 8,
        layers=(64, 32, 16, 8),
        act_fn="relu",
    ):
        super(NeuMF, self).__init__()

        # layer for MLP
        if layers is None:
            layers = [64, 32, 16, 8]
        if num_factors is None:
            num_factors = layers[-1]

        #assert layers[-1] == num_factors

        self.logit = nn.Linear(num_factors + layers[-1], 1)

        self.gmf = GMF(num_users, num_items, num_factors)
        self.mlp = MLP(
            num_users=num_users, num_items=num_items, layers=layers, act_fn=act_fn
        )

        nn.init.normal_(self.logit.weight, std=1e-2)

    def from_pretrained(self, pretrained_gmf, pretrained_mlp, alpha):
        self.gmf.from_pretrained(pretrained_gmf)
        self.mlp.from_pretrained(pretrained_mlp)
        logit_weight = torch.cat(
            [
                alpha * self.gmf.logit.weight,
                (1.0 - alpha) * self.mlp.logit.weight,
            ],
            dim=1,
        )
        logit_bias = alpha * self.gmf.logit.bias + (1.0 - alpha) * self.mlp.logit.bias
        self.logit.weight.data.copy_(logit_weight)
        self.logit.bias.data.copy_(logit_bias)

    def forward(self, users, items, gmf_users=None):
        # gmf_users is there to take advantage of broadcasting
        h_gmf = (
            self.gmf.h(users, items)
            if gmf_users is None
            else self.gmf.h(gmf_users, items)
        )
        h_mlp = self.mlp.h(users, items)
        h = torch.cat([h_gmf, h_mlp], dim=-1)
        output = self.logit(h).view(-1)
        return output


class NCFBase(Recommender):

    def __init__(
        self,
        name="NCF",
        num_epochs=20,
        batch_size=256,
        num_neg=4,
        lr=0.001,
        learner="adam",
        backend="pytorch",
        early_stopping=None,
        trainable=True,
        verbose=True,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.lr = lr
        self.learner = learner
        self.backend = backend
        self.early_stopping = early_stopping
        self.seed = seed
        self.rng = get_rng(seed)
        self.ignored_attrs.extend(
            [
                "graph",
                "user_id",
                "item_id",
                "labels",
                "interaction",
                "prediction",
                "loss",
                "train_op",
                "initializer",
                "saver",
                "sess",
            ]
        )

    def fit(self, train_set, val_set=None):
        Recommender.fit(self, train_set, val_set)

        if self.trainable:
            self.num_users = self.num_users
            self.num_items = self.num_items

            self._fit_pt(train_set, val_set)
            
        return self

    def _build_model_pt(self):
        raise NotImplementedError()

    def _fit_pt(self, train_set, val_set):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)

        self.model = self._build_model_pt().to(self.device)

        optimizer = optimizer_dict[self.learner](
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.reg,
        )
        criteria = nn.MSELoss()

        loop = trange(self.num_epochs, disable=not self.verbose)
        for _ in loop:
            count = 0
            sum_loss = 0
            for batch_id, (batch_users, batch_items, batch_ratings) in enumerate(
                train_set.uir_iter(
                    self.batch_size, shuffle=True, binary=True, num_zeros=self.num_neg
                )
            ):
                batch_users = torch.from_numpy(batch_users).to(self.device)
                batch_items = torch.from_numpy(batch_items).to(self.device)
                batch_ratings = torch.tensor(batch_ratings, dtype=torch.float).to(
                    self.device
                )

                optimizer.zero_grad()
                outputs = self.model(batch_users, batch_items)
                loss = criteria(outputs, batch_ratings)
                loss.backward()
                optimizer.step()

                count += len(batch_users)
                sum_loss += len(batch_users) * loss.data.item()

                if batch_id % 10 == 0:
                    loop.set_postfix(loss=(sum_loss / count))

            if self.early_stopping is not None and self.early_stop(
                train_set, val_set, **self.early_stopping
            ):
                break
        loop.close()

    def _score_pt(self, user_idx, item_idx):
        raise NotImplementedError()

    def save(self, save_dir=None):
        if save_dir is None:
            return

        model_file = Recommender.save(self, save_dir)

        if self.backend == "tensorflow":
            self.saver.save(self.sess, model_file.replace(".pkl", ".cpt"))
        elif self.backend == "pytorch":
            # TODO: implement model saving for PyTorch
            raise NotImplementedError()

        return model_file

    @staticmethod
    def load(model_path, trainable=False):
        model = Recommender.load(model_path, trainable)
        if hasattr(model, "pretrained"):  # NeuMF
            model.pretrained = False

        if model.backend == "tensorflow":
            model._build_graph()
            model.saver.restore(model.sess, model.load_from.replace(".pkl", ".cpt"))
        elif model.backend == "pytorch":
            # TODO: implement model loading for PyTorch
            raise NotImplementedError()

        return model

    def monitor_value(self, train_set, val_set):
        if val_set is None:
            return None

        rmse = rating_eval(
            model=self,
            metrics=[RMSE()],
            train_set=train_set,
            test_set=val_set,
        )[0][0]

        return rmse

    def score(self, user_idx, item_idx=None):
        if self.is_unknown_user(user_idx):
            raise ScoreException("Can't make score prediction for user %d" % user_idx)

        if item_idx is not None and self.is_unknown_item(item_idx):
            raise ScoreException("Can't make score prediction for item %d" % item_idx)

        if self.backend == "tensorflow":
            pred_scores = self._score_tf(user_idx, item_idx)
        elif self.backend == "pytorch":
            pred_scores = self._score_pt(user_idx, item_idx)

        return pred_scores.ravel()
    

class GMFRecommender(NCFBase):

    def __init__(
        self,
        name="GMF",
        num_factors=8,
        reg=0.0,
        num_epochs=20,
        batch_size=256,
        num_neg=4,
        lr=0.001,
        learner="adam",
        backend="tensorflow",
        early_stopping=None,
        trainable=True,
        verbose=True,
        seed=None,
    ):
        super().__init__(
            name=name,
            trainable=trainable,
            verbose=verbose,
            num_epochs=num_epochs,
            batch_size=batch_size,
            num_neg=num_neg,
            lr=lr,
            learner=learner,
            backend=backend,
            early_stopping=early_stopping,
            seed=seed,
        )
        self.num_factors = num_factors
        self.reg = reg

    def _build_model_pt(self):
        return GMF(self.num_users, self.num_items, self.num_factors)

    def _score_pt(self, user_idx, item_idx):
        import torch

        with torch.no_grad():
            users = torch.tensor(user_idx).unsqueeze(0).to(self.device)
            items = (
                torch.from_numpy(np.arange(self.num_items))
                if item_idx is None
                else torch.tensor(item_idx).unsqueeze(0)
            ).to(self.device)
            output = self.model(users, items)
        return output.squeeze().cpu().numpy()
    

class MLPRecommender(NCFBase):

    def __init__(
        self,
        name="MLP",
        layers=(64, 32, 16, 8),
        act_fn="relu",
        reg=0.0,
        num_epochs=20,
        batch_size=256,
        num_neg=4,
        lr=0.001,
        learner="adam",
        backend="tensorflow",
        early_stopping=None,
        trainable=True,
        verbose=True,
        seed=None,
    ):
        super().__init__(
            name=name,
            trainable=trainable,
            verbose=verbose,
            num_epochs=num_epochs,
            batch_size=batch_size,
            num_neg=num_neg,
            lr=lr,
            learner=learner,
            backend=backend,
            early_stopping=early_stopping,
            seed=seed,
        )
        self.layers = layers
        self.act_fn = act_fn
        self.reg = reg

    def _build_model_pt(self):
        return MLP(
            num_users=self.num_users,
            num_items=self.num_items,
            layers=self.layers,
            act_fn=self.act_fn,
        )

    def _score_pt(self, user_idx, item_idx):
        with torch.no_grad():
            if item_idx is None:
                users = torch.from_numpy(np.ones(self.num_items, dtype=int) * user_idx)
                items = (torch.from_numpy(np.arange(self.num_items))).to(self.device)
            else:
                users = torch.tensor(user_idx).unsqueeze(0)
                items = torch.tensor(item_idx).unsqueeze(0)
            output = self.model(users.to(self.device), items.to(self.device))
        return output.squeeze().cpu().numpy()
    
    
class NeuMFRecommender(NCFBase):

    def __init__(
        self,
        name="NeuMF",
        num_factors=8,
        layers=(64, 32, 16, 8),
        act_fn="relu",
        reg=0.0,
        num_epochs=20,
        batch_size=256,
        num_neg=4,
        lr=0.001,
        learner="adam",
        backend="tensorflow",
        early_stopping=None,
        trainable=True,
        verbose=True,
        seed=None,
    ):
        super().__init__(
            name=name,
            trainable=trainable,
            verbose=verbose,
            num_epochs=num_epochs,
            batch_size=batch_size,
            num_neg=num_neg,
            lr=lr,
            learner=learner,
            backend=backend,
            early_stopping=early_stopping,
            seed=seed,
        )
        self.num_factors = num_factors
        self.layers = layers
        self.act_fn = act_fn
        self.reg = reg
        self.pretrained = False
        self.ignored_attrs.extend(
            [
                "gmf_user_id",
                "mlp_user_id",
                "pretrained_gmf",
                "pretrained_mlp",
                "alpha",
            ]
        )

    def from_pretrained(self, pretrained_gmf, pretrained_mlp, alpha=0.5):
        self.pretrained = True
        self.pretrained_gmf = pretrained_gmf
        self.pretrained_mlp = pretrained_mlp
        self.alpha = alpha
        return self

    def _build_model_pt(self):
        model = NeuMF(
            num_users=self.num_users,
            num_items=self.num_items,
            layers=self.layers,
            act_fn=self.act_fn,
        )
        if self.pretrained:
            model.from_pretrained(
                self.pretrained_gmf.model, self.pretrained_mlp.model, self.alpha
            )
        return model

    def _score_pt(self, user_idx, item_idx):
        with torch.no_grad():
            if item_idx is None:
                users = torch.from_numpy(np.ones(self.num_items, dtype=int) * user_idx)
                items = (torch.from_numpy(np.arange(self.num_items))).to(self.device)
            else:
                users = torch.tensor(user_idx).unsqueeze(0)
                items = torch.tensor(item_idx).unsqueeze(0)
            gmf_users = torch.tensor(user_idx).unsqueeze(0).to(self.device)
            output = self.model(
                users.to(self.device), items.to(self.device), gmf_users.to(self.device)
            )
        return output.squeeze().cpu().numpy()
    