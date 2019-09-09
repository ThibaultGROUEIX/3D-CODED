import torch
import my_utils
import meter
import extension.dist_chamfer_idx as ext

distChamfer = ext.chamferDist()


def batch_cycle_2(P1_P2, idx1_P2_P1, batch_size):
    """
    index P1_P2 to correspond to the cycles of points doing P1 -> P2 -> P1
    """
    return P1_P2.view(-1, 3)[idx1_P2_P1].view(batch_size, -1, 3)


def batch_cycle_3(P1_P3, idx1_P2_P1, idx1_P3_P2, batch_size):
    """
    index P1_P3 to correspond to the cycles of points doing P1 -> P2 -> P3 -> P1
    """
    return P1_P3.view(-1, 3)[idx1_P3_P2].contiguous()[idx1_P2_P1].contiguous().view(batch_size, -1, 3)


def L2(tens1, tens2, dim=None):
    """
    Returns the L2 loss between two tensors
    :param tens1:
    :param tens2:
    :param dim:
    :return: loss
    """
    if tens1.nelement() == 0 or tens2.nelement() == 0:
        return 0
    if dim is None:
        return torch.sqrt(torch.mean((tens1 - tens2) ** 2) + 0.00000001)
    else:
        return torch.sqrt(torch.mean((tens1 - tens2) ** 2, dim=dim) + 0.00000001)


def cosine(tens1, tens2):
    if tens1.nelement() == 0 or tens2.nelement() == 0:
        return 0
    # input tens1 : NFeatures
    # input tens2 : NFeatures
    # output : cosine distance
    tens1 = tens1.squeeze() / torch.norm(tens1.squeeze())
    tens2 = tens2.squeeze() / torch.norm(tens2.squeeze())
    similarity = torch.dot(tens1, tens2)
    return 1 - similarity


def chamferL2(dist1, dist2):
    """
    :param dist1: squared distances from pointcloud 1 to pointcloud 2
    :param dist2: squared distances from pointcloud 2 to pointcloud 1
    :return: average square root of inputs
    """
    if dist1.nelement() == 0 or dist2.nelement() == 0:
        return 0
    return (torch.mean(torch.sqrt(dist1))) + (torch.mean(torch.sqrt(dist2)))


def chamferL2_assym1(dist1, dist2):
    """
    :param dist1: squared distances from pointcloud 1 to pointcloud 2
    :param dist2: squared distances from pointcloud 2 to pointcloud 1
    :return: average square root of dist1
    """
    # Check if there are elements in dist1 and dist 2 else return 0 (otherwize torch.mean returns NaN)
    if dist1.nelement() == 0 or dist2.nelement() == 0:
        return 0
    return (torch.mean(torch.sqrt(dist1)))


def chamferL2_assym2(dist1, dist2):
    """
    :param dist1: squared distances from pointcloud 1 to pointcloud 2
    :param dist2: squared distances from pointcloud 2 to pointcloud 1
    :return: average square root of dist2
    """
    # Check if there are elements in dist1 and dist 2 else return 0 (otherwize torch.mean returns NaN)
    if dist1.nelement() == 0 or dist2.nelement() == 0:
        return 0
    return (torch.mean(torch.sqrt(dist2)))


class ChamferLoss(object):
    """
    Choose from chamferL2, chamferL2_assym1 or chamferL2_assym2
    """

    def __init__(self, loss_type="SYM", verbose=True):
        self.loss_type = loss_type
        # Define the loss type
        if self.loss_type == "SYM":
            if verbose:
                my_utils.cyan_print("Using SYM chamfer")
            self.forward = chamferL2
        elif self.loss_type == "ASSYM_1":
            if verbose:
                my_utils.cyan_print("Using ASSYM_1 chamfer")
            self.forward = chamferL2_assym1
        elif self.loss_type == "ASSYM_2":
            if verbose:
                my_utils.cyan_print("Using ASSYM_2 chamfer")
            self.forward = chamferL2_assym2
        else:
            my_utils.cyan_print("please provide a loss type")


def batch_idx(idx, fix):
    """
    This is perhaps the most critical operation of the full repo.
    It adds constant stored in fix to the idx vectors.
    :param idx:
    :param fix:
    :return:
    """
    idx = idx.view(-1).data.long()
    idx = idx + fix[:len(idx)]
    return idx.long()


def forward_chamfer(network, P1, P2, local_fix=None, latent=False, latent_P1=None, latent_P2=None, distChamfer=None):
    """
    :param network:
    :param P1: input points 1 ->  batch, num_points, 3
    :param P2: input points 2  ->  batch, num_points, 3
    :param local_fix: Vector storing constants for batch_idx function
    :param latent: Boolean to decide if latent vector should be stored for slight optimisation of runtime
    :param latent_P1: Latent vector 1 precomputed
    :param latent_P2: Latent vector 2 precomputed
    :param distChamfer: Chamfer distance function
    :return: P2 reconstructed from P1, latent vectors, chamfer outputs
    """
    P1_tmp = P1.transpose(2, 1).contiguous()
    P2_tmp = P2.transpose(2, 1).contiguous()
    if latent:
        P2_P1, latent_vector_P1, latent_vector_P2 = network.forward_classic_with_latent(P1_tmp, P2_tmp,
                                                                                        x_latent=latent_P1,
                                                                                        y_latent=latent_P2)  # forward pass
    else:
        P2_P1 = network(P1_tmp, P2_tmp)  # forward pass
        # P2_P1 = network(P2_P1, P2.transpose(2, 1).contiguous())  # forward pass (twice forward pass)

    P2_P1 = P2_P1.transpose(2, 1).contiguous()
    # reconstruction losses
    dist1, dist2, idx1, idx2 = distChamfer(P2_P1, P2)

    # This is the critical step that allows the batched indexing of the computation of the cycle losses to runs smoothly.
    if not local_fix is None:
        idx1 = batch_idx(idx1, local_fix)
        idx2 = batch_idx(idx2, local_fix)

    # dist1 = dist1.view(-1)[idx2.view(-1).long()]
    if latent:
        # return P1, dist1, dist2, idx1.view(-1).long(), idx2.view(-1).long(), latent_vector
        return P2_P1, dist1, dist2, idx1.view(-1).long(), idx2.view(-1).long(), latent_vector_P1, latent_vector_P2
    else:
        # return P1, dist1, dist2, idx1.view(-1).long(), idx2.view(-1).long()
        return P2_P1, dist1, dist2, idx1.view(-1).long(), idx2.view(-1).long()


class ForwardChamferByPart(object):

    def __init__(self, chamfer_loss_type="SYM"):
        self.chamfer_loss_type = chamfer_loss_type
        self.chamfer_loss = ChamferLoss(chamfer_loss_type)

    def __call__(self, network, P1, LABEL1, P2, LABEL2, local_fix=None, latent=False, latent_P1=None, latent_P2=None,
                 distChamfer=None):
        """
        :param network:
        :param P1: input points 1 ->  batch, num_points, 3
        :param P2: input points 2  ->  batch, num_points, 3
        :param local_fix: Vector storing constants for batch_idx function
        :param latent: Boolean to decide if latent vector should be stored for slight optimisation of runtime
        :param latent_P1: Latent vector 1 precomputed
        :param latent_P2: Latent vector 2 precomputed
        :param distChamfer: Chamfer distance function
        :return: P2 reconstructed from P1, latent vectors, chamfer outputs
        """
        if latent:
            P2_P1, latent_vector_P1, latent_vector_P2 = network.forward_classic_with_latent(
                P1.transpose(2, 1).contiguous(), P2.transpose(2, 1).contiguous(), x_latent=latent_P1,
                y_latent=latent_P2)  # forward pass
        else:
            P2_P1 = network(P1.transpose(2, 1).contiguous(), P2.transpose(2, 1).contiguous())  # forward pass

        P2_P1 = P2_P1.transpose(2, 1).contiguous()

        # reconstruction losses
        dist1, dist2, idx1, idx2 = distChamfer(P2_P1, P2)
        loss = meter.AverageValueMeter()
        for shape in range(P1.size(0)):
            for label in set(LABEL1[shape].cpu().numpy()):
                # A for loop on the batch_size in neccessary as shape parts don't have the same number of points for each shape.
                try:
                    dist1_label, dist2_label, idx1_label, idx2_label = distChamfer(
                        P2_P1[shape][LABEL1[shape] == label].unsqueeze(0),
                        P2[shape][LABEL2[shape] == label].unsqueeze(0))
                    loss.update(self.chamfer_loss.forward(dist1_label, dist2_label))
                except:
                    continue

        # TODO :find a good way to make sure points with no matching labels are not accounted for in cycle consistency
        # This is the critical step that allows the batched indexing of the computation of the cycle losses to runs smoothly.
        if not local_fix is None:
            idx1 = batch_idx(idx1, local_fix)
            idx2 = batch_idx(idx2, local_fix)

        if latent:
            return P2_P1, dist1, dist2, idx1.long(), idx2.long(), latent_vector_P1, latent_vector_P2, loss.avg
        else:
            return P2_P1, dist1, dist2, idx1.long(), idx2.long(), loss.avg


def forward_chamfer_atlasnet(network, P, local_fix=None, distChamfer=None):
    P2 = network(P.transpose(2, 1).contiguous())  # forward pass
    P2 = P2.transpose(2, 1).contiguous()
    # reconstruction losses
    dist1, dist2, idx1, idx2 = distChamfer(P2, P)
    if not local_fix is None:
        idx1 = batch_idx(idx1, local_fix)
        idx2 = batch_idx(idx2, local_fix)
    return P2, dist1, dist2, idx1.long(), idx2.long()


def NN_metric(P1, P2):
    P1 = torch.from_numpy(P1).view(-1, 3).unsqueeze(0).cuda().float()
    P2 = torch.from_numpy(P2).view(-1, 3).unsqueeze(0).cuda().float()
    size_1 = P1.size(1)
    orig_size_tmp_1 = (P1.view(-1)[P1.view(-1) == 100]).nelement()
    orig_size_1 = orig_size_tmp_1 / 3
    assert orig_size_1 * 3 == orig_size_tmp_1
    orig_size_1 = size_1 - orig_size_1

    size_2 = P2.size(1)
    orig_size_tmp_2 = (P2.view(-1)[P2.view(-1) == 100]).nelement()
    orig_size_2 = orig_size_tmp_2 / 3
    assert orig_size_2 * 3 == orig_size_tmp_2
    orig_size_2 = size_2 - orig_size_2

    dist1, dist2, idx1, idx2 = distChamfer(P1, P2)
    return torch.sum(dist1).item() / float(orig_size_1) + torch.sum(dist2).item() / float(orig_size_2)
