/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    colorwarp OpenCL kernels — mirror of the CPU process() core path
    (mask_mode 0, guided-filter smoothing; eigf / prefilter / correction-
    smoothing paths fall back to CPU).

    Per-node resolved parameters are passed as a flat float array P[] to avoid
    dozens of kernel arguments. Index map (must match process_cl in colorwarp.c):
      0 strength   1 hc        2 hw_h     3 sigma_h   4 sel_sat  5 hw_s
      6 sat_sigma  7 light_c   8 hw_l     9 light_sig 10 guard2  11 invert
     12 sel_hue   13 shift_hue 14 shift_chroma 15 shift_light
     16 conv      17 affinity  18 neutral 19 nfall    20 priority
     21 chroma_w  22 luma_w    23 preserve_texture    24 per_component
     25 conv_h 26 aff_h 27 nz_h 28 nzf_h 29 prio_h 30 cw_h
     31 conv_c 32 aff_c 33 nz_c 34 nzf_c 35 prio_c 36 lw_c
     37 conv_l 38 aff_l 39 nz_l 40 nzf_l 41 prio_l
*/

#include "common.h"
#include "colorspace.h"

#define CW_RMAX 1.5f
#define CW_CEXP 1.33654221029386f

static inline float4 _cw_rgb_to_jch(const float4 rgb, constant const float *const M, const float L_white)
{
  const float4 XYZ = matrix_product_float4(rgb, M);
  const float4 xyY = dt_D65_XYZ_to_xyY(XYZ);
  const float Ls = Y_to_dt_UCS_L_star(xyY.z);
  const float2 UV = xyY_to_dt_UCS_UV(xyY);
  return dt_UCS_LUV_to_JCH(Ls, L_white, UV);   // .x = J, .y = C, .z = hue(rad)
}

static inline float _cw_S_to_C(const float S, const float J)
{
  if(S <= 0.0f || J <= 0.0f) return 0.0f;
  float C = S * J;
  for(int i = 0; i < 6; i++) C = S * J * (dtcl_pow(fmax(C, 0.0f), CW_CEXP) + 1.0f);
  return C;
}

static inline float _cw_band_w(const float ad, const float hw, const float inv_2sig2)
{
  const float e = fmax(ad - hw, 0.0f);
  return exp(-(e * e) * inv_2sig2);
}

static inline float _cw_neutral_radius(const float neutral)
{
  const float n = clamp(neutral, 0.0f, 1.0f);
  return CW_RMAX * (exp(3.0f * n) - 1.0f) / (exp(3.0f) - 1.0f);
}

static inline float _cw_affinity_weight(const float affinity, const float neutral,
                                        const float falloff, const float dist)
{
  if(affinity >= 0.0f) return 1.0f + affinity;
  const float r = _cw_neutral_radius(neutral);
  float hole;
  if(dist <= r) hole = 1.0f;
  else
  {
    const float edge = fmax(r * (0.15f + 2.5f * clamp(falloff, 0.0f, 1.0f)), 1e-4f);
    const float u = clamp((dist - r) / edge, 0.0f, 1.0f);
    hole = 1.0f - u * u * (3.0f - 2.0f * u);
  }
  return 1.0f - fabs(affinity) * hole;
}

static inline float _cw_chroma_fac(const float cw, const float cn)
{
  return (cw >= 0.0f) ? (1.0f - cw * (1.0f - cn)) : (1.0f + cw * cn);
}
static inline float _cw_luma_fac(const float lw, const float ln)
{
  return (lw >= 0.0f) ? (1.0f - lw * (1.0f - ln)) : (1.0f + lw * ln);
}
static inline float _cw_val_weight(const float cw, const float lw, const float cn, const float ln)
{
  return _cw_chroma_fac(cw, cn) * _cw_luma_fac(lw, ln);
}
static inline float _cw_pt_factor(const float pt, const float dist)
{
  return (pt > 0.0f) ? dist / (dist + pt * 0.15f) : 1.0f;
}

static inline float _cw_axis_move(const float dtc, const float dt, const float w_base,
                                  const float conv, const float aff, const float nz,
                                  const float falloff, const float pt, const float prio)
{
  const float w = w_base * _cw_affinity_weight(aff, nz, falloff, fabs(dt));
  const float wt = w * (1.0f - conv);
  float wc = (conv > 0.0f) ? fmin(w * conv, 1.0f) : w * conv;
  wc *= _cw_pt_factor(pt, fabs(dt));
  const float p = clamp(1.0f - prio * tanh(dt * 4.0f), 0.0f, 2.0f);
  return p * (wt * dtc + wc * dt);
}

// zero the accumulators before the per-node loop
kernel void colorwarp_clear(global float *acc_h, global float *acc_s, global float *acc_l,
                            const int width, const int height)
{
  const int x = get_global_id(0), y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int k = mad24(y, width, x);
  acc_h[k] = 0.0f; acc_s[k] = 0.0f; acc_l[k] = 0.0f;
}

// build one node's selection mask (single-channel image, so guided_filter_cl can filter it)
kernel void colorwarp_build_mask(read_only image2d_t in, write_only image2d_t mask,
                                 constant float *P, constant float *M, const float L_white,
                                 const int width, const int height)
{
  const int x = get_global_id(0), y = get_global_id(1);
  if(x >= width || y >= height) return;
  const float4 rgb = read_imagef(in, samplerA, (int2)(x, y));
  const float4 JCH = _cw_rgb_to_jch(rgb, M, L_white);
  const float J = JCH.x, C = JCH.y, hh = JCH.z;
  const float S = (J > 1e-6f) ? C / (J * (dtcl_pow(C, CW_CEXP) + 1.0f)) : 0.0f;
  const float sat_p = sqrt(fmax(S, 0.0f) / 0.1f);
  const float dh = fabs(atan2(sin(hh - P[1]), cos(hh - P[1])));
  const float hue_w = _cw_band_w(dh, P[2], 1.0f / (2.0f * P[3] * P[3]));
  const float sat_w = _cw_band_w(fabs(sat_p - P[4]), P[5], 1.0f / (2.0f * P[6] * P[6]));
  const float light_w = _cw_band_w(fabs(J - P[7]), P[8], 1.0f / (2.0f * P[9] * P[9]));
  float sel = hue_w * sat_w * light_w;
  if(P[11] != 0.0f) sel = 1.0f - sel;
  const float g2 = P[10];
  const float m = (g2 > 0.0f) ? sel * (S * S) / (S * S + g2) : sel;
  write_imagef(mask, (int2)(x, y), (float4)(m, 0.0f, 0.0f, 0.0f));
}

// accumulate one node's move into acc_h / acc_s / acc_l
kernel void colorwarp_accumulate(read_only image2d_t in, read_only image2d_t selbuf,
                                 global float *acc_h, global float *acc_s, global float *acc_l,
                                 constant float *P, constant float *M, const float L_white,
                                 const int polar_move, const int width, const int height)
{
  const int x = get_global_id(0), y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int k = mad24(y, width, x);
  const float strength = P[0];
  if(strength <= 0.0f) return;

  const float4 rgb = read_imagef(in, samplerA, (int2)(x, y));
  const float4 JCH = _cw_rgb_to_jch(rgb, M, L_white);
  const float J = JCH.x, C = JCH.y, hh = JCH.z;
  const float S_in = (J > 1e-6f) ? C / (J * (dtcl_pow(C, CW_CEXP) + 1.0f)) : 0.0f;
  const float sat_p = sqrt(fmax(S_in, 0.0f) / 0.1f);
  const float cn = clamp(sat_p, 0.0f, 1.0f), ln = clamp(J, 0.0f, 1.0f);
  const float w_raw = strength * read_imagef(selbuf, samplerA, (int2)(x, y)).x;

  const float conv = P[16], affinity = P[17], neutral = P[18], nfall = P[19];
  const float chroma_w = P[21], luma_w = P[22], pt = P[23];
  const int per_comp = (P[24] != 0.0f);

  if(polar_move)
  {
    const float c_hue = P[12], t_hue = P[12] + P[13];
    const float c_sat = P[4],  t_sat = P[4] + P[14];
    const float c_lgt = P[7],  t_lgt = P[7] + P[15];
    const float hue_p = (hh + M_PI_F) / (2.0f * M_PI_F);
    const float lgt_p = J;
    float dth = t_hue - hue_p; dth -= round(dth);
    const float dts = t_sat - sat_p, dtl = t_lgt - lgt_p;
    float dct_h = t_hue - c_hue; dct_h -= round(dct_h);
    const float dct_s = t_sat - c_sat, dct_l = t_lgt - c_lgt;
    if(per_comp)
    {
      acc_h[k] += _cw_axis_move(dct_h, dth, w_raw * _cw_chroma_fac(P[30], cn), P[25], P[26], P[27], P[28], pt, P[29]);
      acc_s[k] += _cw_axis_move(dct_s, dts, w_raw * _cw_luma_fac(P[36], ln), P[31], P[32], P[33], P[34], pt, P[35]);
      acc_l[k] += _cw_axis_move(dct_l, dtl, w_raw, P[37], P[38], P[39], P[40], pt, P[41]);
    }
    else
    {
      const float w_base = w_raw * _cw_val_weight(chroma_w, luma_w, cn, ln);
      const float dist = sqrt(dth * dth + dts * dts + dtl * dtl);
      const float w = w_base * _cw_affinity_weight(affinity, neutral, nfall, dist);
      const float wt = w * (1.0f - conv);
      const float wc = ((conv > 0.0f) ? fmin(w * conv, 1.0f) : w * conv) * _cw_pt_factor(pt, dist);
      const float ph = clamp(1.0f - P[20] * tanh(dth * 4.0f), 0.0f, 2.0f);
      const float ps = clamp(1.0f - P[20] * tanh(dts * 4.0f), 0.0f, 2.0f);
      const float pl = clamp(1.0f - P[20] * tanh(dtl * 4.0f), 0.0f, 2.0f);
      acc_h[k] += ph * (wt * dct_h + wc * dth);
      acc_s[k] += ps * (wt * dct_s + wc * dts);
      acc_l[k] += pl * (wt * dct_l + wc * dtl);
    }
  }
  else
  {
    const float th_ang = (P[12] + P[13]) * (2.0f * M_PI_F) - M_PI_F;
    const float ch_ang = P[12] * (2.0f * M_PI_F) - M_PI_F;
    const float t_lgt = P[7] + P[15], c_lgt = P[7];
    const float ts = clamp(P[4] + P[14], 0.0f, 2.0f), cs = clamp(P[4], 0.0f, 2.0f);
    const float Ct = _cw_S_to_C(ts * ts * 0.1f, fmax(t_lgt, 1e-4f));
    const float Cc = _cw_S_to_C(cs * cs * 0.1f, fmax(c_lgt, 1e-4f));
    const float a_t = Ct * cos(th_ang), b_t = Ct * sin(th_ang);
    const float a_c = Cc * cos(ch_ang), b_c = Cc * sin(ch_ang);
    const float a_p = C * cos(hh), b_p = C * sin(hh);
    if(per_comp)
    {
      const float wpc = w_raw * _cw_chroma_fac(P[30], cn) * _cw_luma_fac(P[36], ln);
      const float da = a_t - a_p, db = b_t - b_p;
      const float distc = sqrt(da * da + db * db);
      const float w = wpc * _cw_affinity_weight(P[32], P[33], P[34], distc);
      const float wt = w * (1.0f - P[31]);
      const float wc = ((P[31] > 0.0f) ? fmin(w * P[31], 1.0f) : w * P[31]) * _cw_pt_factor(pt, distc);
      acc_h[k] += wt * (a_t - a_c) + wc * da;
      acc_s[k] += wt * (b_t - b_c) + wc * db;
      acc_l[k] += _cw_axis_move(t_lgt - c_lgt, t_lgt - J, w_raw, P[37], P[38], P[39], P[40], pt, P[41]);
    }
    else
    {
      const float w_base = w_raw * _cw_val_weight(chroma_w, luma_w, cn, ln);
      const float da = a_t - a_p, db = b_t - b_p;
      const float dist = sqrt(da * da + db * db);
      const float w = w_base * _cw_affinity_weight(affinity, neutral, nfall, dist);
      const float wt = w * (1.0f - conv);
      const float wc = ((conv > 0.0f) ? fmin(w * conv, 1.0f) : w * conv) * _cw_pt_factor(pt, dist);
      acc_h[k] += wt * (a_t - a_c) + wc * da;
      acc_s[k] += wt * (b_t - b_c) + wc * db;
      acc_l[k] += w * (t_lgt - c_lgt);
    }
  }
}

// apply the accumulated move and write the output
kernel void colorwarp_apply(read_only image2d_t in, write_only image2d_t out,
                            global const float *acc_h, global const float *acc_s, global const float *acc_l,
                            constant float *M, constant float *Mout, const float L_white,
                            const int polar_move, const int width, const int height)
{
  const int x = get_global_id(0), y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int k = mad24(y, width, x);
  const float4 rgb = read_imagef(in, samplerA, (int2)(x, y));
  const float4 JCH = _cw_rgb_to_jch(rgb, M, L_white);
  const float J = JCH.x, C = JCH.y, hh = JCH.z;

  float J2, C2, h2;
  if(polar_move)
  {
    const float hue_p = (hh + M_PI_F) / (2.0f * M_PI_F);
    const float S_in = (J > 1e-6f) ? C / (J * (dtcl_pow(C, CW_CEXP) + 1.0f)) : 0.0f;
    const float sat_p = sqrt(fmax(S_in, 0.0f) / 0.1f);
    const float hue_o = hue_p + acc_h[k];
    const float sat_o = fmax(sat_p + acc_s[k], 0.0f);
    J2 = fmax(J + acc_l[k], 0.0f);
    const float S_out = sat_o * sat_o * 0.1f;
    C2 = fmax(C * (S_out / fmax(S_in, 1e-6f)), 0.0f);
    h2 = hue_o * (2.0f * M_PI_F) - M_PI_F;
  }
  else
  {
    const float a_o = C * cos(hh) + acc_h[k];
    const float b_o = C * sin(hh) + acc_s[k];
    J2 = fmax(J + acc_l[k], 0.0f);
    C2 = hypot(a_o, b_o);
    h2 = atan2(b_o, a_o);
  }
  const float4 JCH2 = (float4)(J2, C2, h2, 0.0f);
  const float4 xyY = dt_UCS_JCH_to_xyY(JCH2, L_white);
  const float4 XYZ = dt_xyY_to_XYZ(xyY);
  float4 outv = matrix_product_float4(XYZ, Mout);
  outv.w = rgb.w;
  write_imagef(out, (int2)(x, y), outv);
}
