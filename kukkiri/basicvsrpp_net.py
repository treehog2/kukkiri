from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


class ModulatedDeformConv2d(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deform_groups=1,
        bias=True,
    ):
        super().__init__()
        k = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.in_channels, self.out_channels, self.kernel_size = in_channels, out_channels, k
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
        self.groups, self.deform_groups = groups, deform_groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *k))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        stdv = 1.0 / math.sqrt(in_channels // groups * k[0] * k[1])
        nn.init.uniform_(self.weight, -stdv, stdv)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def deform_conv(self, x, offset, mask):
        return deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding, self.dilation, mask=mask)


def flow_warp(x, flow, interpolation="bilinear", padding_mode="zeros", align_corners=True):
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f"spatial sizes of input {x.size()[-2:]} and flow {flow.size()[1:3]} differ")
    _, _, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, device=flow.device, dtype=x.dtype),
        torch.arange(0, w, device=flow.device, dtype=x.dtype),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), 2)
    grid.requires_grad = False
    grid_flow = grid + flow
    gx = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    gy = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((gx, gy), dim=3).type(x.type())
    return F.grid_sample(x, grid_flow, mode=interpolation, padding_mode=padding_mode, align_corners=align_corners)


def _default_init_weights(module: nn.Module, scale: float = 1.0) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="leaky_relu")
            m.weight.data *= scale
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class ResidualBlockNoBN(nn.Module):
    def __init__(self, mid_channels: int = 64, res_scale: float = 1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        if res_scale == 1.0:
            for m in (self.conv1, self.conv2):
                _default_init_weights(m, 0.1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x))) * self.res_scale


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 64, num_blocks: int = 30):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Sequential(*[ResidualBlockNoBN(mid_channels=out_channels) for _ in range(num_blocks)]),
        )

    def forward(self, feat):
        return self.main(feat)


class PixelShufflePack(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int, upsample_kernel: int):
        super().__init__()
        self.scale_factor = scale_factor
        self.upsample_conv = nn.Conv2d(
            in_channels, out_channels * scale_factor * scale_factor, upsample_kernel, padding=(upsample_kernel - 1) // 2
        )
        _default_init_weights(self, 1)

    def forward(self, x):
        return F.pixel_shuffle(self.upsample_conv(x), self.scale_factor)


class _ConvAct(nn.Module):

    def __init__(self, cin: int, cout: int, k: int, act: bool):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, 1, k // 2)
        self.activate = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.activate(self.conv(x))


class SPyNetBasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.Sequential(
            _ConvAct(8, 32, 7, True),
            _ConvAct(32, 64, 7, True),
            _ConvAct(64, 32, 7, True),
            _ConvAct(32, 16, 7, True),
            _ConvAct(16, 2, 7, False),
        )

    def forward(self, x):
        return self.basic_module(x)


class SPyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        n, _, h, w = ref.size()
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]
        for _ in range(5):
            ref.append(F.avg_pool2d(ref[-1], kernel_size=2, stride=2, count_include_pad=False))
            supp.append(F.avg_pool2d(supp[-1], kernel_size=2, stride=2, count_include_pad=False))
        ref, supp = ref[::-1], supp[::-1]
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            flow_up = (
                flow if level == 0 else F.interpolate(flow, scale_factor=2, mode="bilinear", align_corners=True) * 2.0
            )
            warped = flow_warp(supp[level], flow_up.permute(0, 2, 3, 1), padding_mode="border")
            flow = flow_up + self.basic_module[level](torch.cat([ref[level], warped, flow_up], 1))
        return flow

    def forward(self, ref, supp):
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(ref, size=(h_up, w_up), mode="bilinear", align_corners=False)
        supp = F.interpolate(supp, size=(h_up, w_up), mode="bilinear", align_corners=False)
        flow = F.interpolate(self.compute_flow(ref, supp), size=(h, w), mode="bilinear", align_corners=False)
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)
        return flow


class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop("max_residue_magnitude", 10)
        super().__init__(*args, **kwargs)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )
        nn.init.constant_(self.conv_offset[-1].weight, 0)
        nn.init.constant_(self.conv_offset[-1].bias, 0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        out = self.conv_offset(torch.cat([extra_feat, flow_1, flow_2], dim=1))
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)
        mask = torch.sigmoid(mask)
        return self.deform_conv(x, offset, mask)


class BasicVSRPlusPlusNet(nn.Module):

    def __init__(self, mid_channels=64, num_blocks=7, max_residue_magnitude=10, is_low_res_input=True):
        super().__init__()
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input
        self.spynet = SPyNet()
        if is_low_res_input:
            self.feat_extract = ResidualBlocksWithInputConv(3, mid_channels, 5)
        else:
            self.feat_extract = nn.Sequential(
                nn.Conv2d(3, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ResidualBlocksWithInputConv(mid_channels, mid_channels, 5),
            )
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        for i, module in enumerate(["backward_1", "forward_1", "backward_2", "forward_2"]):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                2 * mid_channels,
                mid_channels,
                3,
                padding=1,
                deform_groups=16,
                max_residue_magnitude=max_residue_magnitude,
            )
            self.backbone[module] = ResidualBlocksWithInputConv((2 + i) * mid_channels, mid_channels, num_blocks)
        self.reconstruction = ResidualBlocksWithInputConv(5 * mid_channels, mid_channels, 5)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def compute_flow(self, lqs):
        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:].reshape(-1, c, h, w)
        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)
        flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)
        return flows_forward, flows_backward

    def propagate(self, feats, flows, module_name):
        n, t, _, h, w = flows.size()
        frame_idx = list(range(0, t + 1))
        flow_idx = list(range(-1, t))
        mapping_idx = list(range(0, len(feats["spatial"])))
        mapping_idx += mapping_idx[::-1]
        if "backward" in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx
        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats["spatial"][mapping_idx[idx]]
            if i > 0:
                flow_n1 = flows[:, flow_idx[i]]
                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)
                if i > 1:
                    feat_n2 = feats[module_name][-2]
                    flow_n2 = flows[:, flow_idx[i - 1]]
                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond, flow_n1, flow_n2)
            feat = [feat_current] + [feats[k][idx] for k in feats if k not in ["spatial", module_name]] + [feat_prop]
            feat_prop = feat_prop + self.backbone[module_name](torch.cat(feat, dim=1))
            feats[module_name].append(feat_prop)
        if "backward" in module_name:
            feats[module_name] = feats[module_name][::-1]
        return feats

    def upsample(self, lqs, feats):
        outputs = []
        mapping_idx = list(range(0, len(feats["spatial"])))
        mapping_idx += mapping_idx[::-1]
        for i in range(0, lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != "spatial"]
            hr.insert(0, feats["spatial"][mapping_idx[i]])
            hr = self.reconstruction(torch.cat(hr, dim=1))
            hr = self.lrelu(self.upsample1(hr))
            hr = self.lrelu(self.upsample2(hr))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            hr = hr + (self.img_upsample(lqs[:, i]) if self.is_low_res_input else lqs[:, i])
            outputs.append(hr)
        return torch.stack(outputs, dim=1)

    def forward(self, lqs):
        n, t, c, h, w = lqs.size()
        if self.is_low_res_input:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(lqs.view(-1, c, h, w), scale_factor=0.25, mode="bicubic").view(
                n, t, c, h // 4, w // 4
            )
        feats_ = self.feat_extract(lqs.view(-1, c, h, w))
        fh, fw = feats_.shape[2:]
        feats_ = feats_.view(n, t, -1, fh, fw)
        feats = {"spatial": [feats_[:, i] for i in range(0, t)]}
        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, "flow input must be >= 64px"
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)
        for iter_ in [1, 2]:
            for direction in ["backward", "forward"]:
                module = f"{direction}_{iter_}"
                feats[module] = []
                feats = self.propagate(feats, flows_backward if direction == "backward" else flows_forward, module)
        return self.upsample(lqs, feats)
