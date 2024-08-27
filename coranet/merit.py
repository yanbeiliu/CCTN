import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from layers import AvgReadout, Discriminator
alf = 0.8
class MLP(nn.Module):

    def __init__(self, inp_size, outp_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, outp_size)
        )

    def forward(self, x):
        return self.net(x)


class GraphEncoder(nn.Module):

    def __init__(self, 
                  gnn,
                  projection_hidden_size,
                  projection_size):
        
        super().__init__()
        
        self.gnn = gnn
        self.projector = MLP(512, projection_size, projection_hidden_size)           
        
    def forward(self, adj, in_feats, sparse):
        representations = self.gnn(in_feats, adj, sparse)
        representations1 = representations.view(-1, representations.size(-1))
        projections = self.projector(representations1)  # (batch, proj_dim)
        return representations, projections

    
class EMA():
    
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def sim(h1, h2):
    z1 = F.normalize(h1, dim=-1, p=2)
    z2 = F.normalize(h2, dim=-1, p=2)
    return torch.mm(z1, z2.t())


def contrastive_loss_wo_cross_network(h1, h2, z):
    f = lambda x: torch.exp(x)
    intra_sim = f(sim(h1, h1))
    inter_sim = f(sim(h1, h2))
    return -torch.log(inter_sim.diag() /
                     (intra_sim.sum(dim=-1) + inter_sim.sum(dim=-1) - intra_sim.diag()))


def contrastive_loss_wo_cross_view(h1, h2, z):
    f = lambda x: torch.exp(x)
    cross_sim = f(sim(h1, z))
    return -torch.log(cross_sim.diag() / cross_sim.sum(dim=-1))


class MERIT(nn.Module):

    def __init__(self, 
                 gnn,
                 feat_size,
                 projection_size, 
                 projection_hidden_size,
                 prediction_size,
                 prediction_hidden_size,
                 moving_average_decay,
                 beta):
        
        super().__init__()
        ######################
        self.dgigcn = gnn
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(projection_size)
        ######################
        self.online_encoder = GraphEncoder(gnn, projection_hidden_size, projection_size)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(self.target_encoder, False)
        self.target_ema_updater = EMA(moving_average_decay)
        self.online_predictor = MLP(projection_size, prediction_size, prediction_hidden_size)
        self.beta = beta
                   
    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_ma(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    # def forward(self, aug_adj_1, aug_adj_2, aug_feat_1, aug_feat_2, sparse):
    def forward(self, aug_adj_1, aug_adj_2, aug_feat_1, aug_feat_2, negative_adj, negative_feat, nb_nodes, sparse):
        ######
        h_n ,_= self.online_encoder(negative_adj, negative_feat, sparse)

        h_p1, _ = self.online_encoder(aug_adj_1, aug_feat_1, sparse)
        c1 = self.read(h_p1, None)
        c1 = self.sigm(c1)

        h_p2, _ = self.online_encoder(aug_adj_2, aug_feat_2, sparse)
        c2 = self.read(h_p2, None)
        c2 = self.sigm(c2)

        # print(c1.size(), h_p1.size(), h_p2.size(),h_n.size())
        ret1 = self.disc(c1, h_p1, h_n, None, None)
        ret2 = self.disc(c2, h_p2, h_n, None, None)

        lbl_1 = torch.ones(1, nb_nodes)
        lbl_2 = torch.zeros(1, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)
        lbl = lbl.cuda()
        b_xent = nn.BCEWithLogitsLoss()

        loss1 = 0.5 * (b_xent(ret1, lbl) + b_xent(ret2, lbl))

        ######
        _, online_proj_one = self.online_encoder(aug_adj_1, aug_feat_1, sparse)
        _, online_proj_two = self.online_encoder(aug_adj_2, aug_feat_2, sparse)
        # print('online_proj_one', type(online_proj_one), online_proj_one.size())
        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)
                      
        with torch.no_grad():
            _, target_proj_one = self.target_encoder(aug_adj_1, aug_feat_1, sparse)
            _, target_proj_two = self.target_encoder(aug_adj_2, aug_feat_2, sparse)
        criterion = nn.CosineSimilarity(dim=1)
        # l1 = self.beta * contrastive_loss_wo_cross_network(online_pred_one, online_pred_two, target_proj_two.detach()) + \
        #     (1.0 - self.beta) * contrastive_loss_wo_cross_view(online_pred_one, online_pred_two, target_proj_two.detach())
        #
        # l2 = self.beta * contrastive_loss_wo_cross_network(online_pred_two, online_pred_one, target_proj_one.detach()) + \
        #     (1.0 - self.beta) * contrastive_loss_wo_cross_view(online_pred_two, online_pred_one, target_proj_one.detach())
        # l1 = contrastive_loss_wo_cross_network(online_pred_one, online_pred_two, target_proj_two.detach()).mean() #+ (1.0 - self.beta) * contrastive_loss_wo_cross_view(online_pred_one, online_pred_two, target_proj_two.detach())
        l1 = -criterion(online_pred_one, target_proj_two.detach()).mean()
        # l2 = contrastive_loss_wo_cross_network(online_pred_two, online_pred_one, target_proj_one.detach()).mean() #+ (1.0 - self.beta) * contrastive_loss_wo_cross_view(online_pred_two, online_pred_one, target_proj_one.detach())
        l2 = -criterion(online_pred_two, target_proj_one.detach()).mean()
        loss = (1-alf)*0.5*(l1 + l2) + alf*loss1

        return loss#.mean()