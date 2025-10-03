import clip.model
import torch.nn as nn
import clip
from torch.nn import functional as F
import torch
from einops import rearrange
from . import model_hyperclip
from torch import Tensor

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def init_weights_eye(m):
	if type(m) == nn.Linear:
		nn.init.eye_(m.weight)
		if m.bias is not None:
			nn.init.constant_(m.bias, 0)


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        res = (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


def artanh(x):
    return Artanh.apply(x)


class BlockDiagonalLinear(nn.Module):
    def __init__(self, block_size, in_features, out_features, curvature: float = 2.5):
        super(BlockDiagonalLinear, self).__init__()
        self.block_size = block_size
        self.r = int(out_features / block_size)
        self.in_features = in_features
        self.out_features = out_features
        self.curvature = curvature 
    def cayley_batch(self, data):
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.transpose(1, 2))
        # I = torch.eye(r, device=data.device).unsqueeze(0).repeat(b, 1, 1)
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

        # Perform the Cayley parametrization
        Q = torch.bmm(I - skew, torch.inverse(I + skew))

        return Q
    def block_diagonal(self, R):
        if len(R.shape) == 2:
            # Create a list of R repeated block_count times
            blocks = [R] * self.r
        else:
            # Create a list of R slices along the third dimension
            blocks = [R[i, ...] for i in range(self.r)]

        # Use torch.block_diag to create the block diagonal matrix
        A = torch.block_diag(*blocks)

        return A

    def exp_map0(self, x: Tensor, eps: float = 1e-8) -> Tensor:
        """
        Exponential map: map points from the tangent space at the vertex
        to the hyperboloid using the exponential map of Lorentz model.
        """
        #if torch.norm(x) < eps:
        #    return torch.zeros_like(x)
        rc_xnorm = self.curvature ** 0.5 * torch.norm(x, dim=-1, keepdim=True)
        sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2 ** 15))
        _output = torch.sinh(sinh_input) * x / rc_xnorm
        return _output

    def expmap0(self, u):
        """
        Exponential map: map points from the tangent space at the vertex
        to the hyperboloid using the exponential map of poincare ball model.
        """
        #sqrt_c = self.curvature ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), 1e-5)
        gamma_1 = self.tanh(self.curvature ** 0.5 * u_norm) * u / (self.curvature ** 0.5 * u_norm)
        return gamma_1

    def log_map0(self, x: Tensor, eps: float = 1e-8) -> Tensor:
        """
        Logarithmic map: map points from the hyperboloid to the tangent space
        at the vertex using the logarithmic map of Lorentz model.
        """
        #if torch.norm(x) < eps:
        #    return torch.zeros_like(x)
        rc_x_time = torch.sqrt(1 + self.curvature * torch.sum(x ** 2, dim=-1, keepdim=True))
        _distance0 = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))
        rc_xnorm = self.curvature ** 0.5 * torch.norm(x, dim=-1, keepdim=True)
        _output = _distance0 * x / torch.clamp(rc_xnorm, min=eps)
        return _output

    def logmap0(self, y):
        """
        Logarithmic map: map points from the hyperboloid to the tangent space
        at the vertex using the logarithmic map of poincare ball model.
        Logarithmic map for :math:`y` from :math:`0` on the manifold.
        """
        sqrt_c = self.curvature ** 0.5
        y_norm = torch.clamp_min(y.norm(dim=-1, p=2, keepdim=True), 1e-5)
        return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)

    def pairwise_inner(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute pairwise Lorentzian inner product between input vectors for 4D inputs.

        Args:
            x: Tensor of shape `(B, H, W)` giving space components of a batch of vectors 
            on the hyperboloid (where B is batch size, H and W are dimensions).
            y: Tensor of shape `(B, H, W)` giving space components of another 
            batch of points on the hyperboloid.
            curv: Positive scalar denoting negative hyperboloid curvature.

        Returns:
            Tensor of shape `(B, H, H)` giving pairwise Lorentzian inner product
            between input vectors.
        """
        # Compute the time component for both x and y (last dimension is time-like)
        x_time = torch.sqrt(1 / self.curvature + torch.sum(x**2, dim=-1, keepdim=True))
        y_time = torch.sqrt(1 / self.curvature + torch.sum(y**2, dim=-1, keepdim=True))
        
        # x @ y.T equivalent for batch processing using Einstein summation
        xyl = torch.einsum('bij,bkj->bik', x, y) - torch.einsum('bij,bkj->bik', x_time, y_time)
        
        return xyl

    def tanh(self, x, clamp=15):
        return x.clamp(-clamp, clamp).tanh()
       
    def rotation_transform(self, x):

        cosh_vals = torch.cosh(self.rotation)
        sinh_vals = torch.sinh(self.rotation)
        # 将输入 x 也 reshape 为 (N, d//2, 2)
        x = x.view(x.shape[0], -1, 2)
        # 应用 Givens 旋转
        x_rot = cosh_vals.unsqueeze(-1) * x + sinh_vals.unsqueeze(-1) * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
        # 恢复形状 (N, d)
        return x_rot.view(x.shape[0], -1)

    def reflection_transform(self, x):

        cosh_vals = torch.cosh(self.reflection)
        sinh_vals = torch.sinh(self.reflection)
        # 将输入 x 也 reshape 为 (N, d//2, 2)
        x = x.view(x.shape[0], -1, 2)
        # 应用 Givens 反射
        x_ref = cosh_vals.unsqueeze(-1) * x - sinh_vals.unsqueeze(-1) * torch.cat((x[:, :, 1:], -x[:, :, 0:1]), dim=-1)
        # 恢复形状 (N, d)
        return x_ref.view(x.shape[0], -1)

    def reflection_transform_householder(self, x):
        v = self.reflection_householder / (torch.norm(self.reflection_householder) + 1e-8)  # 归一化反射向量
        I = torch.eye(self.out_features, device=x.device)  # 单位矩阵
        H = I - 2 * torch.outer(v, v)  # Householder反射矩阵

        # 应用Householder反射
        return H @ x  # 结果转置以匹配输入的形状
    def mobius_matvec(self, m, x):
        r"""
        Generalization for matrix-vector multiplication to hyperbolic space defined as
        .. math::
            M \otimes_c x = (1/\sqrt{c}) \tanh\left(
                \frac{\|Mx\|_2}{\|x\|_2}\tanh^{-1}(\sqrt{c}\|x\|_2)
            \right)\frac{Mx}{\|Mx\|_2}
        Parameters
        ----------
        m : tensor
            matrix for multiplication
        x : tensor
            point on poincare ball
        c : float|tensor
            negative ball curvature
        Returns
        -------
        tensor
            Mobius matvec result
        """
        #c = torch.as_tensor(c).type_as(x)
        return self._mobius_matvec(m, x)


    def _mobius_matvec(self, m, x):
        x_norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), 1e-5)
        sqrt_c = self.curvature ** 0.5
        mx = x @ m.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2)
        res_c = self.tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        res = torch.where(cond.bool(), res_0, res_c)
        return self._project(res)

    def mobius_add(self, x, y):
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        y2 = y.pow(2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 + 2 * self.curvature * xy + self.curvature * y2) * x + (1 - self.curvature * x2) * y
        denom = 1 + 2 * self.curvature * xy + self.curvature ** 2 * x2 * y2
        return num / (denom + 1e-5)
    
    def _project(self, x):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), 1e-5)
        maxnorm = (1 - 1e-3) / (self.curvature ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def tanh(self, x, clamp=15):
        return x.clamp(-clamp, clamp).tanh()
    

    def forward(self, x, scaling_matrix, visual=False):
        output_hyperbolic = self.expmap0(x)
        fix_filt = output_hyperbolic.data

        block_diagonal_weight = self.block_diagonal(scaling_matrix)

        output_hyperbolic_filt_stretch = self.mobius_matvec(block_diagonal_weight, fix_filt)

        output_euclidean = self.logmap0(output_hyperbolic_filt_stretch) 
        return output_euclidean



def oft_forward(self, x):
    B, N, C = x.shape
    res_x = x
    orig_dtype = x.dtype

    _, N1, _, _ = self.attn_q_proj_oft_layer_R.shape
    attn_tensor = rearrange(self.attn_q_proj_oft_layer_R.cuda().to(orig_dtype), 'B1 N1 L1 M1 -> B1 (N1 L1) M1')
    attn_tensor = self.attn_q_proj_oft_relation_m_R[int((self.count-4)/4)](attn_tensor)[:,:, :4]
    attn_tensor = rearrange(attn_tensor, 'B1 (N1 L1) M1 -> M1 B1 N1 L1', N1=N1)
    attn_re = self.attn_q_proj_oft_relation_l_R(attn_tensor)[..., self.count-4:self.count]
    q_R, k_R, v_R, proj_R = attn_re[..., 0], attn_re[..., 1], attn_re[..., 2], attn_re[..., 3]


    in_proj_weight_new = self.hyperbolic_attn_in(self.attn.in_proj_weight, q_R, k_R, v_R, visual=False)
    qkv = nn.functional.linear(input=self.ln_1(x), weight=in_proj_weight_new, bias=self.attn.in_proj_bias)#.reshape(B, N, 3, self.n_head, C // self.n_head).permute(2, 1, 3, 0, 4)
    qkv = qkv.reshape(B,N,3,
                    self.attn.num_heads,
                    C // self.attn.num_heads).permute(
                    2, 1, 3, 0, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = (q @ k.transpose(-2,-1)) * (float(self.attn.head_dim) ** -0.5)
    attn = attn + self.attn_mask.unsqueeze(0).unsqueeze(1).cuda().to(orig_dtype)
    attn = attn.softmax(dim=-1)
    oft_out = ((attn @ v).transpose(1,2)).permute(1,0,2,3).reshape(B, N, C)
    out_proj_weight_new = self.hyperbolic_attn(self.attn.out_proj.weight, proj_R, visual=False)
    oft_out = nn.functional.linear(input=oft_out, weight=out_proj_weight_new, bias=self.attn.out_proj.bias)

    oft_out = self.dp(oft_out)
    final = res_x + oft_out #+ ori_attn_x
    #final = res_x + oft_out
    final = final + self.mlp(self.ln_2(final))
    return final



def set_oft(model, dim=8, hidden_size=512, length=12, s=0.1, r=4, count=0):
    for _ in model.children():
        #print('length',length)
        if isinstance(_, model_hyperclip.ResidualAttentionBlock):
            count+=4
            _.hyperbolic_attn_in = Adapter_init_in(hidden_size, dim, curvature_ratio=0.01)
            _.hyperbolic_attn = Adapter_init(hidden_size, dim, curvature_ratio=0.01)
            _.dp = nn.Dropout(_.attn.dropout)
            _.s = s
            _.dim = dim
            _.hidden_size = hidden_size
            _.count = count
            bound_method = oft_forward.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_oft(_, dim, hidden_size, length, s, r, count)



def oft_forward_vision(self, x):
    orig_dtype = x.dtype

    _, N1, _, _ = self.attn_q_proj_oft_layer_R.shape
    attn_tensor = rearrange(self.attn_q_proj_oft_layer_R.cuda().to(orig_dtype), 'B1 N1 L1 M1 -> B1 (N1 L1) M1')
    attn_tensor = self.attn_q_proj_oft_relation_m_R[int((self.count-4)/4)](attn_tensor)[:,:,-6:]
    attn_tensor = rearrange(attn_tensor, 'B1 (N1 L1) M1 -> M1 B1 N1 L1', N1=N1)
    attn_re = self.attn_q_proj_oft_relation_l_R(attn_tensor)[..., self.count-4:self.count]
    q_R, k_R, v_R, proj_R = attn_re[..., 0], attn_re[..., 1], attn_re[..., 2], attn_re[..., 3]

    if self.count <= 44:
        B, N, C = x.shape
        res_x = x
        in_proj_weight_new = self.hyperbolic_attn_in(self.attn.in_proj_weight, q_R, k_R, v_R, visual=True)
        qkv = nn.functional.linear(input=self.ln_1(x), weight=in_proj_weight_new, bias=self.attn.in_proj_bias)#.reshape(B, N, 3, self.n_head, C // self.n_head).permute(2, 1, 3, 0, 4)
        qkv = qkv.reshape(B,N,3,
                        self.n_head,
                        C // self.n_head).permute(
                        2, 1, 3, 0, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2,-1)) * (float(self.attn.head_dim) ** -0.5)
        attn = attn.softmax(dim=-1)
        oft_out = ((attn @ v).transpose(1,2)).permute(1,0,2,3).reshape(B, N, C)
        out_proj_weight_new = self.hyperbolic_attn(self.attn.out_proj.weight, proj_R, visual=True)
        oft_out = nn.functional.linear(input=oft_out, weight=out_proj_weight_new, bias=self.attn.out_proj.bias)
        oft_out = self.dp(oft_out)
        final = res_x + oft_out 
        final = final + self.mlp(self.ln_2(final))
        return final
    else:
        in_proj_weight_new = self.hyperbolic_attn_in(self.attn.in_proj_weight, q_R, k_R, v_R, visual=True)
        y = nn.functional.linear(input=self.ln_1(x), weight=in_proj_weight_new, bias=self.attn.in_proj_bias)
        L, N, D = y.shape # L N 3D        
        y = y.reshape(L, N, 3, D // 3).permute(2, 1, 0, 3).reshape(3 * N, L, D // 3)
        out_proj_weight_new = self.hyperbolic_attn(self.attn.out_proj.weight, proj_R, visual=True)
        y = nn.functional.linear(input=y, weight=out_proj_weight_new, bias=self.attn.out_proj.bias)      
        q, k, v = y.tensor_split(3, dim=0)      
        v = v.transpose(1, 0) + x[:1] # L N D
        v = v + self.mlp(self.ln_2(v))
        return v

def set_oft_vision(model, dim=8, hidden_size=512, length=12, s=0.1, r=6, count=0):

    for _ in model.children():
        #print('length',length)
        if isinstance(_, model_hyperclip.ResidualAttentionBlock):
            count+=4
            print('_.count',count)
            _.dp = nn.Dropout(_.attn.dropout)
            _.s = s
            _.dim = dim

            _.hyperbolic_attn_in = Adapter_init_in(hidden_size, dim, curvature_ratio=0.01)
            _.hyperbolic_attn = Adapter_init(hidden_size, dim, curvature_ratio=0.01)

            _.hidden_size = hidden_size
            _.count = count
            bound_method = oft_forward_vision.__get__(_, _.__class__)
            if count <= 44:
                setattr(_, 'forward', bound_method)
            else:
                setattr(_, 'forward_dense', bound_method)
        elif len(list(_.children())) != 0:
            set_oft_vision(_, dim, hidden_size, length, s, r, count)
    print('count',count)


class Adapter_init(nn.Module):
    def __init__(self, hidden_size, dim, curvature_ratio=1.0):
        super().__init__()

        self.adapter_attn_o = BlockDiagonalLinear(block_size=dim, in_features=hidden_size, out_features=hidden_size, curvature=curvature_ratio)
        self.dim = dim


    def forward(self, attn, proj = None, visual=False):
        #B, N, C = attn.shape
        orig_dtype = attn.dtype

        fix_filt = attn.data


        filt = self.adapter_attn_o(fix_filt, scaling_matrix=proj, visual=visual)

        return filt.to(orig_dtype)


class Adapter_init_in(nn.Module):
    def __init__(self, hidden_size, dim, curvature_ratio=1.0):
        super().__init__()

        self.adapter_attn_q = BlockDiagonalLinear(block_size=dim, in_features=hidden_size, out_features=hidden_size, curvature=curvature_ratio)

        self.adapter_attn_k = BlockDiagonalLinear(block_size=dim, in_features=hidden_size, out_features=hidden_size, curvature=curvature_ratio)

        self.adapter_attn_v = BlockDiagonalLinear(block_size=dim, in_features=hidden_size, out_features=hidden_size, curvature=curvature_ratio)

        self.dim = dim


    def forward(self, attn, q_R = None, k_R = None, v_R = None, visual=False):
        #B, N, C = attn.shape
        orig_dtype = attn.dtype

        fix_filt = attn.data
        q_proj_weight, k_proj_weight, v_proj_weight = fix_filt.chunk(3, dim=0)

        filt_q = self.adapter_attn_q(q_proj_weight, scaling_matrix=q_R, visual=visual)

        filt_k = self.adapter_attn_k(k_proj_weight, scaling_matrix=k_R, visual=visual) 

        filt_v = self.adapter_attn_v(v_proj_weight, scaling_matrix=v_R, visual=visual)

        filt  = torch.cat([filt_q, filt_k, filt_v], dim=0)
        return filt.to(orig_dtype)

