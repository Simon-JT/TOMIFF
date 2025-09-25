import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.autograd import Variable
import numbers
import math

class SimpleDWT(nn.Module):
    def __init__(self, J=1, mode='periodization', wave='haar'):
        super().__init__()
        self.J = J
        self.mode = mode
        self.wave = wave
        self.low_pass = nn.Parameter(torch.tensor([0.7071, 0.7071]).view(1, 1, 1, 2))
        self.high_pass = nn.Parameter(torch.tensor([0.7071, -0.7071]).view(1, 1, 1, 2))
    
    def forward(self, x):
        B, C, H, W = x.shape
        L = F.conv2d(x, self.low_pass, padding=(0, 1))
        L = F.conv2d(L, self.low_pass.transpose(-1, -2), padding=(1, 0))
        H_coeff = F.conv2d(x, self.high_pass, padding=(0, 1))
        H_coeff = F.conv2d(H_coeff, self.high_pass.transpose(-1, -2), padding=(1, 0))
        L = L[:, :, ::2, ::2]
        H_coeff = H_coeff[:, :, ::2, ::2]
        return L, [H_coeff]


DWTForward = SimpleDWT
DWTInverse = None 

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

# ===================== Global Memory Module =====================
class GlobalMemoryModule(nn.Module):

    def __init__(self, wavelet_name="haar", memory_scales=[1, 2, 3, 4, 5]):
        super(GlobalMemoryModule, self).__init__()
        
        self.wavelet_name = wavelet_name
        self.memory_scales = memory_scales
        self.dwt_modules = nn.ModuleDict()
        for scale in memory_scales:
            self.dwt_modules[f'dwt_J{scale}'] = DWTForward(
                J=scale, 
                mode='periodization', 
                wave=wavelet_name
            )
        
        self.global_processors = nn.ModuleDict()
        for scale in memory_scales:
            self.global_processors[f'processor_J{scale}'] = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
    
    def extract_global_info(self, x):
        global_memory = {}
        
        for scale in self.memory_scales:
            dwt_key = f'dwt_J{scale}'
            processor_key = f'processor_J{scale}'
            
            L, Yh = self.dwt_modules[dwt_key](x)
            
            processed_L = self.global_processors[processor_key](L)
            
            global_memory[f'scale_{scale}'] = {
                'L': L,  
                'Yh': Yh,  
                'processed_L': processed_L  
            }
        
        return global_memory
    
    def get_global_memory(self, x):
        return self.extract_global_info(x)
    
    def forward(self, x):
        return self.extract_global_info(x)


class GlobalMemoryKeeper(nn.Module):

    
    def __init__(self, wavelet_name="haar", memory_scales=[1, 2, 3, 4, 5]):
        super(GlobalMemoryKeeper, self).__init__()
        
        self.global_memory_module = GlobalMemoryModule(wavelet_name, memory_scales)
        self.memory_scales = memory_scales
        
        self.register_buffer('global_memory_state', torch.zeros(1))
        self.global_memory_cache = {}
    
    def update_global_memory(self, x, update_cache=True):
        current_memory = self.global_memory_module(x)
        
        if update_cache:
            self.global_memory_cache = current_memory
        
        return current_memory
    
    def get_cached_memory(self):
        return self.global_memory_cache
    
    def forward(self, x):
        return self.update_global_memory(x)


class MultiScaleGlobalMemory(nn.Module):
    def __init__(self, wavelet_name="haar"):
        super(MultiScaleGlobalMemory, self).__init__()
        
       
        self.scale_128 = GlobalMemoryModule(wavelet_name, [1])  
        self.scale_64 = GlobalMemoryModule(wavelet_name, [2])  
        self.scale_32 = GlobalMemoryModule(wavelet_name, [3])   
        self.scale_16 = GlobalMemoryModule(wavelet_name, [4])   
        self.scale_8 = GlobalMemoryModule(wavelet_name, [5])   
        
    def extract_multi_scale_global_info(self, x):
        scale_128_memory = self.scale_128(x)
        scale_64_memory = self.scale_64(x)
        scale_32_memory = self.scale_32(x)
        scale_16_memory = self.scale_16(x)
        scale_8_memory = self.scale_8(x)
        

        multi_scale_memory = {
            'scale_128': scale_128_memory['scale_1'], 
            'scale_64': scale_64_memory['scale_2'],    
            'scale_32': scale_32_memory['scale_3'],    
            'scale_16': scale_16_memory['scale_4'],    
            'scale_8': scale_8_memory['scale_5']       
        }
        
        return multi_scale_memory
    
    def forward(self, x):
        return self.extract_multi_scale_global_info(x)

# ===================== Enhanced Base Extraction Module =====================
class EnhancedBaseExtraction(nn.Module):

    def __init__(self, wavelet_name="haar"):
        super().__init__()
       
        self.global_memory = MultiScaleGlobalMemory(wavelet_name)
        
        self.global_fusion = nn.Sequential(
            nn.Conv2d(64*5, 128, kernel_size=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.global_to_base = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )
    
    def extract_global_base(self, x):
        multi_scale_memory = self.global_memory(x)
        global_features = []
        for scale_name, memory in multi_scale_memory.items():
            
            processed_L = memory['processed_L']
            global_features.append(processed_L)
        
        combined_global = torch.cat(global_features, dim=1)
        fused_global = self.global_fusion(combined_global)
        base = self.global_to_base(fused_global)
        return base
    
    def forward(self, x):
        
        return self.extract_global_base(x)

def unit_test():
    import numpy as np
    x = torch.tensor(np.random.rand(16,2,128,128).astype(np.float32)).cuda()
    model = ReFusion()
    model.cuda()
    y = model(x)
    print('output shape:', y.shape)

def count_parameter(net):
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.3fM" % (total/1e6))
    return total

#===============================================================================
# 新的基于Gabor分解的融合网络
#===============================================================================
def generate_gabor_kernel(ksize, sigma, theta, lambd, psi, gamma):
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, ksize),
        torch.linspace(-1, 1, ksize)
    )
    x_theta = x * torch.cos(theta) + y * torch.sin(theta)
    y_theta = -x * torch.sin(theta) + y * torch.cos(theta)
    gb_real = torch.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2)) * torch.cos(2 * math.pi * x_theta / lambd + psi)
    gb_imag = torch.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2)) * torch.sin(2 * math.pi * x_theta / lambd + psi)
    return gb_real, gb_imag

# ===================== Gabor Guided Transformer =====================
class LightweightGaborGuidedTransformer(nn.Module):
   
    def __init__(self, in_channels=1, embed_dim=32, num_heads=4, window_size=8):
        super().__init__()
        self.gabor = LearnableGaborDecompositionModule(ksize=5, num_orientations=6)
        self.q_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.kv_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)  

        self.mlp = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 4, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim * 4, embed_dim, 1)
        )

        self.out_proj = nn.Conv2d(embed_dim, in_channels, kernel_size=1)

    def forward(self, x):
        B, _, H, W = x.shape
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        need_pad = pad_h != 0 or pad_w != 0
        if need_pad:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        base, detail = self.gabor(x)
        q = self.q_proj(detail)  
        kv = self.kv_proj(base)  
        B, embed_dim, H_pad, W_pad = q.shape
        assert H_pad % ws == 0 and W_pad % ws == 0, "H, W必须能被window_size整除"
        q_windows = q.unfold(2, ws, ws).unfold(3, ws, ws)  
        kv_windows = kv.unfold(2, ws, ws).unfold(3, ws, ws)
        num_h, num_w = q_windows.shape[2], q_windows.shape[3]
        q_seq = q_windows.permute(0,2,3,1,4,5).reshape(-1, embed_dim, ws*ws).transpose(1,2)  
        kv_seq = kv_windows.permute(0,2,3,1,4,5).reshape(-1, embed_dim, ws*ws).transpose(1,2)
        q_seq = q_seq.permute(1,0,2)  
        kv_seq = kv_seq.permute(1,0,2)
        attn_out, _ = self.attn(q_seq, kv_seq, kv_seq)
        attn_out = attn_out.permute(1,2,0)  
        attn_out = attn_out.reshape(-1, embed_dim, ws, ws)  
        attn_out = attn_out.reshape(B, num_h, num_w, embed_dim, ws, ws).permute(0,3,1,4,2,5).reshape(B, embed_dim, H_pad, W_pad)
        out = attn_out + self.mlp(attn_out)
        out = self.out_proj(out)
        if need_pad:
            out = out[..., :H, :W]
        out = out + detail  # 添加残差连接，保持细节信息
        return out

class LearnableGaborDecompositionModule(nn.Module):
    def __init__(self, ksize=5, num_orientations=6):
        super().__init__()
        self.ksize = ksize
        self.num_orientations = num_orientations
        self.sigma = nn.Parameter(torch.ones(num_orientations) * 2.0)
        self.lambd = nn.Parameter(torch.ones(num_orientations) * 5.0)
        self.psi = nn.Parameter(torch.zeros(num_orientations))
        self.gamma = nn.Parameter(torch.ones(num_orientations) * 1.2)
        
        initial_angles = torch.linspace(0, 2*math.pi, num_orientations+1)[:-1]
        self.theta = nn.Parameter(initial_angles)
        self.direction_weights = nn.Parameter(torch.ones(num_orientations) / num_orientations)
        self.register_buffer('sigma_min', torch.tensor(0.1))
        self.register_buffer('lambd_min', torch.tensor(0.1))
        self.register_buffer('gamma_min', torch.tensor(0.1))

    def get_constrained_params(self):
        sigma = F.softplus(self.sigma) + self.sigma_min
        lambd = F.softplus(self.lambd) + self.lambd_min
        gamma = F.softplus(self.gamma) + self.gamma_min
        theta = torch.tanh(self.theta) * math.pi
        psi = torch.tanh(self.psi) * math.pi
        
        return sigma, lambd, gamma, theta, psi

    def generate_gabor_kernels(self):
        sigma, lambd, gamma, theta, psi = self.get_constrained_params()
        
    
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, self.ksize, device=sigma.device),
            torch.linspace(-1, 1, self.ksize, device=sigma.device)
        )
        
    
        filters_real = []
        filters_imag = []
        for i in range(self.num_orientations):
            x_theta = x * torch.cos(theta[i]) + y * torch.sin(theta[i])
            y_theta = -x * torch.sin(theta[i]) + y * torch.cos(theta[i])
            
        
            gb_real = torch.exp(-(x_theta**2 + gamma[i]**2 * y_theta**2) / (2 * sigma[i]**2)) * \
                     torch.cos(2 * math.pi * x_theta / lambd[i] + psi[i])
            gb_imag = torch.exp(-(x_theta**2 + gamma[i]**2 * y_theta**2) / (2 * sigma[i]**2)) * \
                     torch.sin(2 * math.pi * x_theta / lambd[i] + psi[i])
            
            filters_real.append(gb_real)
            filters_imag.append(gb_imag)
            
        return torch.stack(filters_real), torch.stack(filters_imag)

    def forward(self, x):
    
        filters_real, filters_imag = self.generate_gabor_kernels()
        conv_real = F.conv2d(x, filters_real.unsqueeze(1), padding=self.ksize//2)
        conv_imag = F.conv2d(x, filters_imag.unsqueeze(1), padding=self.ksize//2)
        weights = F.softmax(self.direction_weights, dim=0)
        weights = weights.view(1, -1, 1, 1)      
        base_layer = (conv_real * weights).sum(dim=1, keepdim=True)
        detail_layer = (conv_imag * weights).sum(dim=1, keepdim=True)
        
        return base_layer, detail_layer

# ===================== Fusion Modules =====================
class SpatialFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_conv = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        self.detail_conv = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        self.final_fusion = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1)
        )
    def forward(self, base_fused, detail_fused):
        if base_fused.shape[-2:] != detail_fused.shape[-2:]:
            base_fused = F.interpolate(base_fused, size=detail_fused.shape[-2:], mode='bilinear', align_corners=False)
        
        base_feat = self.base_conv(base_fused)
        detail_feat = self.detail_conv(detail_fused)
        fused = torch.cat([base_feat, detail_feat], dim=1)
        output = self.final_fusion(fused)
        return output

class LocalAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)       
        attention = self.conv(avg_map)
        return x * attention

class DDFM(nn.Module):
    def __init__(self, in_channels=16):
        super().__init__()
       
        self.ir_conv = nn.Conv2d(1, in_channels, kernel_size=3, padding=1)
        self.vi_conv = nn.Conv2d(1, in_channels, kernel_size=3, padding=1)
        self.gabor_conv = nn.Conv2d(1, in_channels, kernel_size=3, padding=1)
        self.sa_ir = LocalAttention(in_channels)
        self.sa_vi = LocalAttention(in_channels)
        self.sa_gabor = LocalAttention(in_channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, gabor_fused, ir_img, vis_img):
       
        ir_feat = self.ir_conv(ir_img)
        vi_feat = self.vi_conv(vis_img)
        gabor_feat = self.gabor_conv(gabor_fused)       
        ir_feat = self.sa_ir(ir_feat)
        vi_feat = self.sa_vi(vi_feat)
        gabor_feat = self.sa_gabor(gabor_feat)
        combined = torch.cat([ir_feat, vi_feat, gabor_feat], dim=1)
        output = self.fusion(combined)     
        output = (output - output.min()) / (output.max() - output.min() + 1e-6)
        return output

class LearnableDetailFusion(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, detail_ir, detail_vis):
        abs_ir = torch.abs(detail_ir)
        abs_vis = torch.abs(detail_vis)
        max_abs = torch.max(abs_ir, abs_vis)
        alpha = abs_ir / (max_abs + 1e-6)
        detail_fused = alpha * detail_ir + (1 - alpha) * detail_vis
        return detail_fused

# ===================== Cross-Modality Correlation Fusion =====================
class CrossModalityCorrelationFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1)
        )
    
    def forward(self, base_ir, base_vis):
        mu_ir = base_ir.mean(dim=[2,3], keepdim=True)
        mu_vis = base_vis.mean(dim=[2,3], keepdim=True)
        sigma_ir = base_ir.std(dim=[2,3], keepdim=True)
        sigma_vis = base_vis.std(dim=[2,3], keepdim=True)
        corr = ((base_ir - mu_ir) * (base_vis - mu_vis)).mean(dim=[2,3], keepdim=True) / (sigma_ir * sigma_vis + 1e-6)
        gate = torch.sigmoid(self.mlp(corr))
        fused = gate * base_ir + (1 - gate) * base_vis
        return fused

class GaborFusion(nn.Module):
    
    def __init__(self, use_lightweight=True):
        super().__init__()
        self.enhanced_base_extraction = EnhancedBaseExtraction(wavelet_name="haar")
        self.gabor = LightweightGaborGuidedTransformer(in_channels=1, embed_dim=32, num_heads=4, window_size=8)
        self.detail_fusion = LearnableDetailFusion()
        self.fusion = SpatialFusion()
        self.ddfm = DDFM()
        self.base_gate = CrossModalityCorrelationFusion(channels=1)
        
    def forward(self, x, meta=False):
        if x.shape[1] == 2:
            ir_img = x[:, 0:1, :, :]
            vis_img = x[:, 1:2, :, :]
        else:
            ir_img = x[:, :1, :, :]
            vis_img = x[:, 1:, :, :]
            
        base_ir = self.enhanced_base_extraction(ir_img)
        base_vis = self.enhanced_base_extraction(vis_img)
        detail_ir = self.gabor(ir_img)
        detail_vis = self.gabor(vis_img)
        base_fused = self.base_gate(base_ir, base_vis)
        detail_fused = self.detail_fusion(detail_ir, detail_vis)
        
        gabor_fused = self.fusion(base_fused, detail_fused)
        output = self.ddfm(gabor_fused, ir_img, vis_img)
        output = torch.clamp(output, 0, 1)
        
        return output

if __name__ == '__main__':
    unit_test()
    count_parameter(ReFusion())

 

    
