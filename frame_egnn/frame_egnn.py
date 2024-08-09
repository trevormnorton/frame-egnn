from torch import nn
import torch
import roma

class FEGCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """

    def __init__(self, 
                 input_nf,
                 output_nf, 
                 hidden_nf, 
                 edges_in_d=0, 
                 update_frames=True,
                 act_fn=nn.SiLU(), 
                 residual=True, 
                 attention=False, 
                 normalize=False, 
                 coords_agg='mean', 
                 quats_agg='mean', 
                 tanh=False):
        super(FEGCL, self).__init__()
        input_edge = input_nf * 2
        self.update_frames = update_frames
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.quats_agg = quats_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1 + 4 + 3

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        if self.update_frames:
            coord_mlp = []
            coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
            coord_mlp.append(act_fn)
            coord_mlp.append(layer)
            if self.tanh:
                coord_mlp.append(nn.Tanh())
            self.coord_mlp = nn.Sequential(*coord_mlp)

        #Outputs sie of i, j, k coordinates
        if self.update_frames:
            self.quat_mlp = nn.Sequential(
                nn.Linear(hidden_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, 3)
            )

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, quat_prod, unit_vecs, edge_attr):
        #TODO Add radial basis to embed distances smoothly
        #TODO determine how to guarantee smoothness (i.e. if distance is greater than cutoff it should be zeroed out)
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial, quat_prod, unit_vecs], dim=1)
        else:
            out = torch.cat([source, target, radial, quat_prod, unit_vecs, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, h, edge_index, edge_feat, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_feat, row, num_segments=h.size(0))
        if node_attr is not None:
            agg = torch.cat([h, agg, node_attr], dim=1)
        else:
            agg = torch.cat([h, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = h + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def quat_model(self, quat, edge_index, edge_feat):
        row, col = edge_index
        tan_vecs = self.quat_mlp(edge_feat)
        if self.quats_agg == 'sum':
            agg = unsorted_segment_sum(tan_vecs, row, num_segments=quat.size(0))
        elif self.quats_agg == 'mean':
            agg = unsorted_segment_mean(tan_vecs, row, num_segments=quat.size(0))
        else:
            raise Exception('Wrong quats_agg parameter' % self.quats_agg)
        rots = quat_exponential(agg)
        quat = roma.quat_product(quat, rots)
        return quat



    def compute_invariants(self, edges, coord, quat):
        row, col = edges
        coord_diff = coord[row] - coord[col]
        quat_inv = roma.quat_inverse(quat)
        quat_prod = roma.quat_product(quat_inv[row], quat[col])
        rotation = roma.RotationUnitQuat(quat_inv[row])
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)
        norm = torch.sqrt(radial)+ self.epsilon
        if self.normalize:
            coord_diff = coord_diff / norm
            unit_vecs = -rotation.apply(coord_diff)
        else:
            unit_vecs = -rotation.apply(coord_diff / norm)

        return radial, coord_diff, quat_prod, unit_vecs

    def forward(self, h, edge_index, coord, quat, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff, quat_prod, unit_vecs = self.compute_invariants(edge_index, coord, quat)

        edge_feat = self.edge_model(h[row], h[col], radial, quat_prod, unit_vecs, edge_attr)
        if self.update_frames:
            coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
            quat = self.quat_model(quat, edge_index, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, quat, edge_attr


class FrameEGNN(nn.Module):
    def __init__(self, 
                 in_node_nf, 
                 hidden_nf, 
                 out_node_nf, 
                 in_edge_nf=0, 
                 update_frames=True,
                 act_fn=nn.SiLU(), 
                 n_layers=4, 
                 residual=True, 
                 attention=False, 
                 normalize=False, 
                 tanh=False,
                 final_update_frames=True):
        '''

        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param update_frames: Sets whether frame information is updated each layer
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        :param final_update_frames: Sets whether the final layer updates the frames
        '''

        super(FrameEGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            update_frames_on_layer = update_frames and (final_update_frames or (i != n_layers - 1))
            self.add_module("gcl_%d" % i, FEGCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                update_frames=update_frames_on_layer, act_fn=act_fn, residual=residual, 
                                                attention=attention, normalize=normalize, tanh=tanh))

    def forward(self, h, x, q, edges, edge_attr):
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h, x, q, _ = self._modules["gcl_%d" % i](h, edges, x, q, edge_attr=edge_attr)
        h = self.embedding_out(h)
        return h, x, q


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr

def quat_exponential(imag_quat):
    #exp(r q) = cos(r) + q sin(r) when q is a unit, imaginary quaternion
    r = torch.norm(imag_quat, dim = 1, keepdim= True)
    unit_quat = imag_quat / r
    exp = torch.cat([unit_quat * torch.sin(r), torch.cos(r)], dim=1)
    return exp

if __name__ == "__main__":
    # Dummy parameters
    batch_size = 8
    n_nodes = 4
    n_feat = 1
    x_dim = 3

    # Dummy variables h, x and fully connected edges
    h = torch.ones(batch_size *  n_nodes, n_feat)
    x = torch.ones(batch_size * n_nodes, x_dim)
    q = roma.random_unitquat(batch_size * n_nodes)
    edges, edge_attr = get_edges_batch(n_nodes, batch_size)
    
    print('Initialize EGNN')
    # Initialize EGNN
    egnn = FrameEGNN(in_node_nf=n_feat, hidden_nf=32, out_node_nf=1, in_edge_nf=1)
    print('Run EGNN')
    # Run EGNN
    h, x, q = egnn(h, x, q, edges, edge_attr)