import torch

class LaplacianLoss(object):
    """
    Encourages minimal mean curvature shapes.
    """
    def __init__(self, faces, vert, toref=True):
        # Input:
        #  faces: B x F x 3
        self.toref = toref
        from laplacian import Laplacian
        # V x V
        self.laplacian = Laplacian(faces)
        self.Lx = None
        tmp = self.laplacian(vert)
        self.curve_gt = torch.norm(tmp.view(-1, tmp.size(2)), p=2, dim=1).float()
        if not self.toref:
            self.curve_gt = self.curve_gt*0
    
    def __call__(self, verts):
        self.Lx = self.laplacian(verts)
        # Reshape to BV x 3
        Lx = self.Lx.view(-1, self.Lx.size(2))
        loss = (torch.norm(Lx, p=2, dim=1).float()-self.curve_gt).mean()
        return loss
