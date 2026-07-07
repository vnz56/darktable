/*
  This file is part of darktable,
  Copyright (C) 2010-2026 darktable developers.

  Color Uniformity -- OpenCL kernel V8
  darktable UCS JCH

  This kernel is the exact mirror of the CPU code in coloruniformity.c (V8).
  Any change to the CPU engine must be reflected here.

  V8: P0 (asymmetric priority), P1 (per-component t_norm),
       P2 (texture preservation)
*/

#include "colorspace.h"

/* ---------------------------------------------------------------------
   Inline utilities
   --------------------------------------------------------------------- */

static inline float wrap_hue(float h)
{
  h = fmod(h, 1.0f);
  if(h < 0.0f) h += 1.0f;
  return h;
}

// Shortest angular distance [-0.5, 0.5]
static inline float hue_dist(const float h1, const float h2)
{
  float d = fmod(h1 - h2, 1.0f);
  if(d > 0.5f) d -= 1.0f;
  else if(d < -0.5f) d += 1.0f;
  return d;
}

/* ---------------------------------------------------------------------
   Selection engine -- hard pizza slice + asymmetric luma window
   --------------------------------------------------------------------- */

static inline float compute_affinity_weight(
    const float Y_pix,  const float h_pix,
    const float Y_src,  const float h_src,
    const float hue_width,
    const float shadow_limit, const float highlight_limit,
    const float mask_hardness, const float mask_feather)
{
  // 1. HUE
  const float half_angle = max(hue_width * 0.5f, 1e-6f);
  const float dH = hue_dist(h_pix, h_src);
  const float t_h = fabs(dH) / half_angle;

  if(t_h >= 1.0f) return 0.0f;

  float mask_h;
  const float hard = clamp(mask_hardness, 0.0f, 0.999f);
  if(t_h <= hard)
  {
    mask_h = 1.0f;
  }
  else
  {
    const float u = (t_h - hard) / max(1.0f - hard, 1e-6f);
    const float ss3 = u * u * (3.0f - 2.0f * u);
    const float ss5 = u * u * u * (u * (u * 6.0f - 15.0f) + 10.0f);
    mask_h = 1.0f - (ss3 + mask_feather * (ss5 - ss3));
  }

  // 2. LUMINANCE
  if(Y_pix < 1e-5f || Y_src < 1e-5f) return 0.0f;

  const float dY = log2(max(Y_pix, 1e-6f)) - log2(max(Y_src, 1e-6f));

  float t_l;
  if(dY >= 0.0f)
  {
    const float safe_hl = max(highlight_limit, 1e-3f);
    t_l = dY / safe_hl;
  }
  else
  {
    const float safe_sl = max(shadow_limit, 1e-3f);
    t_l = -dY / safe_sl;
  }

  if(t_l >= 1.0f) return 0.0f;

  float mask_l;
  if(t_l <= hard)
  {
    mask_l = 1.0f;
  }
  else
  {
    const float u = (t_l - hard) / max(1.0f - hard, 1e-6f);
    const float ss3 = u * u * (3.0f - 2.0f * u);
    const float ss5 = u * u * u * (u * (u * 6.0f - 15.0f) + 10.0f);
    mask_l = 1.0f - (ss3 + mask_feather * (ss5 - ss3));
  }

  return mask_h * mask_l;
}

// Per-component affinity
static inline float apply_affinity(const float base_weight, const float aff,
                                   const float preserve, const float t_norm)
{
  if(base_weight < 1e-6f) return 0.0f;
  if(aff >= 0.0f)
  {
    return base_weight * (1.0f + aff * base_weight);
  }
  else
  {
    const float safe_preserve = max(preserve, 1e-3f);
    const float hole = exp(-t_norm / (safe_preserve * safe_preserve));
    return base_weight * (1.0f - fabs(aff) * hole);
  }
}


/* ---------------------------------------------------------------------
   Main kernel -- V8
   --------------------------------------------------------------------- */

kernel void coloruniformity_apply(
    read_only image2d_t dev_in,
    write_only image2d_t dev_out,
    const int width,
    const int height,
    /* matrices */
    global float *inm,           // work RGB -> XYZ D65  (composite D50->D65)
    global float *outm,          // XYZ D65 -> work RGB  (composite D65->D50)
    global float *xyz_to_lms,    // XYZ D65 -> LMS 2006
    global float *lms_to_xyz,    // LMS 2006 -> XYZ D65
    /* source */
    const float source_hue,
    const float source_chroma,
    const float source_lightness,
    /* selection */
    const float hue_width,
    const float shadow_limit,
    const float highlight_limit,
    const float mask_hardness,
    const float mask_feather,
    const float chroma_gate,
    /* target */
    const float target_hue,
    const float target_chroma,
    const float target_lightness,
    /* hue correction */
    const float strength_h,
    const float affinity_h,
    const float preserve_h,
    /* chroma correction */
    const float strength_c,
    const float affinity_c,
    const float preserve_c,
    /* lightness correction */
    const float strength_l,
    const float affinity_l,
    const float preserve_l,
    /* offsets */
    const float offset_h,
    const float offset_c,
    const float offset_l,
    const int offsets_weighted,
    /* ui */
    const int preview_mode,
    const float Lwhite,
    /* V12: asymmetric priority (P0) */
    const float priority_h,
    const float priority_c,
    const float priority_l,
    /* V12: texture preservation (P2) */
    const float preserve_texture_h,
    const float preserve_texture_c,
    const float preserve_texture_l)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  float4 pix_in = read_imagef(dev_in, sampleri, (int2)(x, y));

  // Input safety check
  if(!isfinite(pix_in.x) || !isfinite(pix_in.y) || !isfinite(pix_in.z)
     || pix_in.x < -1e-3f || pix_in.y < -1e-3f || pix_in.z < -1e-3f)
  {
    write_imagef(dev_out, (int2)(x, y), pix_in);
    return;
  }

  // --- A. COLOR SPACE CONVERSIONS ---

  // Work RGB -> XYZ D65
  float4 px_xyz_d65;
  px_xyz_d65.x = dot((float4)(pix_in.xyz, 0.0f), (float4)(inm[0], inm[1], inm[2],  0.0f));
  px_xyz_d65.y = dot((float4)(pix_in.xyz, 0.0f), (float4)(inm[4], inm[5], inm[6],  0.0f));
  px_xyz_d65.z = dot((float4)(pix_in.xyz, 0.0f), (float4)(inm[8], inm[9], inm[10], 0.0f));
  px_xyz_d65.w = 0.0f;

  // UCS JCH (hue)
  float4 px_xyY = dt_D65_XYZ_to_xyY(px_xyz_d65);
  float4 px_JCH = xyY_to_dt_UCS_JCH(px_xyY, Lwhite);
  const float h_pix = wrap_hue(px_JCH.z / (2.0f * M_PI_F));

  // Filmlight Yrg (luminance + chroma)
  float4 px_lms;
  px_lms.x = dot(px_xyz_d65, (float4)(xyz_to_lms[0], xyz_to_lms[1], xyz_to_lms[2],  0.0f));
  px_lms.y = dot(px_xyz_d65, (float4)(xyz_to_lms[4], xyz_to_lms[5], xyz_to_lms[6],  0.0f));
  px_lms.z = dot(px_xyz_d65, (float4)(xyz_to_lms[8], xyz_to_lms[9], xyz_to_lms[10], 0.0f));
  px_lms.w = 0.0f;

  float4 px_yrg = LMS_to_Yrg(px_lms);
  const float Y_pix = px_yrg.x;

  // D65 neutral point
  const float4 d65_xyz = (float4)(0.95047f, 1.0f, 1.08883f, 0.0f);
  float4 d65_lms;
  d65_lms.x = dot(d65_xyz, (float4)(xyz_to_lms[0], xyz_to_lms[1], xyz_to_lms[2],  0.0f));
  d65_lms.y = dot(d65_xyz, (float4)(xyz_to_lms[4], xyz_to_lms[5], xyz_to_lms[6],  0.0f));
  d65_lms.z = dot(d65_xyz, (float4)(xyz_to_lms[8], xyz_to_lms[9], xyz_to_lms[10], 0.0f));
  d65_lms.w = 0.0f;
  const float4 d65_yrg = LMS_to_Yrg(d65_lms);
  const float rn = d65_yrg.y;
  const float gn = d65_yrg.z;

  // Pixel chroma
  const float dr_pix = px_yrg.y - rn;
  const float dg_pix = px_yrg.z - gn;
  const float c_pix = sqrt(dr_pix * dr_pix + dg_pix * dg_pix);

  // --- B. SELECTION WEIGHT ---
  float weight = compute_affinity_weight(
      Y_pix, h_pix,
      source_lightness, source_hue,
      hue_width, shadow_limit, highlight_limit,
      mask_hardness, mask_feather);

  // Hue reliability gate
  if(chroma_gate > 0.0f)
  {
    const float edge = max(source_chroma * 0.5f * chroma_gate, 1e-4f);
    weight *= smoothstep(0.0f, edge, c_pix);
  }

  // --- P1: PER-COMPONENT normalized distance for affinity ---
  const float t_norm_h = fabs(hue_dist(h_pix, source_hue))
                       / max(hue_width * 0.5f, 1e-6f);
  const float t_norm_c = (source_chroma > 1e-4f)
                       ? fabs(c_pix - source_chroma) / source_chroma
                       : 0.0f;
  const float dY_aff = log2(max(Y_pix, 1e-6f)) - log2(max(source_lightness, 1e-6f));
  const float t_norm_l = fabs(dY_aff)
                       / max(dY_aff >= 0.0f ? highlight_limit : shadow_limit, 1e-3f);

  // Checker pattern
  const int checker_size = max((int)(8), 2);
  const int checker_period = 2 * checker_size;

  // --- EARLY EXIT: pixel outside selection ---
  if(weight < 1e-4f)
  {
    float4 out;
    if(preview_mode == 1)  // MASK
    {
      out = (float4)(0.0f, 0.0f, 0.0f, pix_in.w);
    }
    else if(preview_mode == 2 || preview_mode == 3)  // OVERLAY
    {
      const int row_odd = (y % checker_period) < checker_size;
      const int col_odd = (x % checker_period) < checker_size;
      const float checker = (row_odd ^ col_odd) ? 0.45f : 0.18f;
      out = (float4)(checker, checker, checker, pix_in.w);
    }
    else if(preview_mode == 4)  // DELTA
    {
      out = (float4)(0.5f, 0.5f, 0.5f, pix_in.w);
    }
    else  // NONE
    {
      out = pix_in;
    }
    write_imagef(dev_out, (int2)(x, y), out);
    return;
  }

  // Passthrough if no corrections and no preview
  if(preview_mode == 0
     && strength_h == 0.0f && strength_c == 0.0f && strength_l == 0.0f
     && offset_h == 0.0f   && offset_c == 0.0f   && offset_l == 0.0f)
  {
    write_imagef(dev_out, (int2)(x, y), pix_in);
    return;
  }

  // --- C. CORRECTIONS ---

  // Per-component weights with independent affinity and per-component t_norm (P1)
  const float w_h = apply_affinity(weight, affinity_h, preserve_h, t_norm_h);
  const float w_c = apply_affinity(weight, affinity_c, preserve_c, t_norm_c);
  const float w_l = apply_affinity(weight, affinity_l, preserve_l, t_norm_l);

  // Offset application factor
  const float off_w = offsets_weighted ? weight : 1.0f;

  // ===============================================================
  // 1. Hue -- delta computed in UCS JCH, rotation applied in Yrg
  // ===============================================================
  float delta_h_val = 0.0f;
  {
    float h_side;
    if(strength_h >= 0.0f)
    {
      const float gap_h = hue_dist(target_hue, h_pix);
      delta_h_val = gap_h * w_h * strength_h;
      // Anti-overshoot
      if(fabs(delta_h_val) > fabs(gap_h))
        delta_h_val = gap_h;
      h_side = hue_dist(h_pix, target_hue);
    }
    else
    {
      delta_h_val = hue_dist(h_pix, source_hue) * w_h * fabs(strength_h);
      h_side = hue_dist(h_pix, source_hue);
    }

    // P2: Texture preservation (convergence only)
    if(preserve_texture_h > 0.0f && strength_h >= 0.0f)
    {
      const float d_norm = fabs(hue_dist(h_pix, target_hue))
                         / max(hue_width * 0.5f, 1e-6f);
      delta_h_val *= 1.0f - preserve_texture_h * exp(-d_norm * d_norm * 8.0f);
    }

    // P0: Asymmetric priority
    if(priority_h != 0.0f && fabs(h_side) > 1e-6f)
    {
      if(h_side * priority_h < 0.0f)
        delta_h_val *= max(0.0f, 1.0f - fabs(priority_h));
    }
  }
  // Add offset
  delta_h_val += offset_h * off_w;

  // Apply hue rotation as 2D rotation in the Yrg chromaticity plane
  // around the D65 neutral point. Preserves Y and chroma exactly.
  if(fabs(delta_h_val) > 1e-7f)
  {
    const float angle = delta_h_val * 2.0f * M_PI_F;
    const float cos_a = cos(angle);
    const float sin_a = sin(angle);
    const float dr = px_yrg.y - rn;
    const float dg = px_yrg.z - gn;
    px_yrg.y = rn + dr * cos_a - dg * sin_a;
    px_yrg.z = gn + dr * sin_a + dg * cos_a;
  }

  const float Y_base = px_yrg.x;
  const float r_base = px_yrg.y;
  const float g_base = px_yrg.z;
  const float dr_base = r_base - rn;
  const float dg_base = g_base - gn;
  const float c_base = sqrt(dr_base * dr_base + dg_base * dg_base);

  // ===============================================================
  // 2. Chroma (Yrg)
  // ===============================================================
  float delta_c_val;
  float c_side;
  if(strength_c >= 0.0f)
  {
    const float gap_c = target_chroma - c_base;
    delta_c_val = gap_c * w_c * strength_c;
    if(fabs(delta_c_val) > fabs(gap_c))
      delta_c_val = gap_c;
    c_side = (c_base > target_chroma) ? 1.0f : -1.0f;
  }
  else
  {
    delta_c_val = (c_base - source_chroma) * w_c * fabs(strength_c);
    c_side = (c_base > source_chroma) ? 1.0f : -1.0f;
  }

  // P2: Texture preservation
  if(preserve_texture_c > 0.0f && strength_c >= 0.0f)
  {
    const float ref_c = max(source_chroma, 1e-4f);
    const float d_norm = fabs(c_base - target_chroma) / ref_c;
    delta_c_val *= 1.0f - preserve_texture_c * exp(-d_norm * d_norm * 8.0f);
  }

  // P0: Asymmetric priority
  if(priority_c != 0.0f)
  {
    if(c_side * priority_c < 0.0f)
      delta_c_val *= max(0.0f, 1.0f - fabs(priority_c));
  }

  float c_corr = c_base + delta_c_val;
  c_corr *= max(1.0f + offset_c * off_w, 0.0f);
  c_corr = max(c_corr, 0.0f);

  // ===============================================================
  // 3. Lightness (Log2)
  // ===============================================================
  const float log_Y = log2(max(Y_base, 1e-6f));
  const float log_target_l = log2(max(target_lightness, 1e-6f));
  const float log_source_l = log2(max(source_lightness, 1e-6f));

  float delta_l_val;
  float l_side;
  if(strength_l >= 0.0f)
  {
    const float gap_l = log_target_l - log_Y;
    delta_l_val = gap_l * w_l * strength_l;
    if(fabs(delta_l_val) > fabs(gap_l))
      delta_l_val = gap_l;
    l_side = (log_Y > log_target_l) ? 1.0f : -1.0f;
  }
  else
  {
    delta_l_val = (log_Y - log_source_l) * w_l * fabs(strength_l);
    l_side = (log_Y > log_source_l) ? 1.0f : -1.0f;
  }

  // P2: Texture preservation
  if(preserve_texture_l > 0.0f && strength_l >= 0.0f)
  {
    const float range = max(shadow_limit + highlight_limit, 1e-3f);
    const float d_norm = fabs(log_Y - log_target_l) / range;
    delta_l_val *= 1.0f - preserve_texture_l * exp(-d_norm * d_norm * 8.0f);
  }

  // P0: Asymmetric priority
  if(priority_l != 0.0f)
  {
    if(l_side * priority_l < 0.0f)
      delta_l_val *= max(0.0f, 1.0f - fabs(priority_l));
  }

  const float log_Y_corr = log_Y + delta_l_val + offset_l * off_w;
  const float Y_corr = exp2(log_Y_corr);

  // Yrg reconstruction
  const float ratio = (c_base > 1e-5f) ? (c_corr / c_base) : 0.0f;
  px_yrg.x = max(Y_corr, 0.0f);
  px_yrg.y = rn + dr_base * ratio;
  px_yrg.z = gn + dg_base * ratio;

  // --- D. BACK TO RGB ---
  px_lms = Yrg_to_LMS(px_yrg);

  // LMS -> XYZ D65
  px_xyz_d65.x = dot(px_lms, (float4)(lms_to_xyz[0], lms_to_xyz[1], lms_to_xyz[2],  0.0f));
  px_xyz_d65.y = dot(px_lms, (float4)(lms_to_xyz[4], lms_to_xyz[5], lms_to_xyz[6],  0.0f));
  px_xyz_d65.z = dot(px_lms, (float4)(lms_to_xyz[8], lms_to_xyz[9], lms_to_xyz[10], 0.0f));
  px_xyz_d65.w = 0.0f;

  // XYZ D65 -> work RGB
  float4 px_rgb_out;
  px_rgb_out.x = dot(px_xyz_d65, (float4)(outm[0], outm[1], outm[2],  0.0f));
  px_rgb_out.y = dot(px_xyz_d65, (float4)(outm[4], outm[5], outm[6],  0.0f));
  px_rgb_out.z = dot(px_xyz_d65, (float4)(outm[8], outm[9], outm[10], 0.0f));
  px_rgb_out.w = pix_in.w;

  // Clamp output: prevent explosive values and negative out-of-gamut results
  px_rgb_out.xyz = clamp(px_rgb_out.xyz, 0.0f, 1e6f);

  // --- PREVIEWS ---
  float4 out;

  if(preview_mode == 1)  // MASK
  {
    out = (float4)(weight, weight, weight, pix_in.w);
  }
  else if(preview_mode == 2 || preview_mode == 3)  // OVERLAY
  {
    const int row_odd = (y % checker_period) < checker_size;
    const int col_odd = (x % checker_period) < checker_size;
    const float checker = (row_odd ^ col_odd) ? 0.45f : 0.18f;
    const float w_comp = 1.0f - weight;
    const float4 src = (preview_mode == 3) ? px_rgb_out : pix_in;
    out.x = w_comp * checker + weight * src.x;
    out.y = w_comp * checker + weight * src.y;
    out.z = w_comp * checker + weight * src.z;
    out.w = pix_in.w;
  }
  else if(preview_mode == 4)  // DELTA
  {
    out.x = 0.5f + (px_rgb_out.x - pix_in.x) * 5.0f;
    out.y = 0.5f + (px_rgb_out.y - pix_in.y) * 5.0f;
    out.z = 0.5f + (px_rgb_out.z - pix_in.z) * 5.0f;
    out.w = pix_in.w;
  }
  else  // NONE
  {
    out = px_rgb_out;
  }

  write_imagef(dev_out, (int2)(x, y), out);
}
