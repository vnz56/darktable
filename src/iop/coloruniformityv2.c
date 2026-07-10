/*
This file is part of darktable,
Copyright (C) 2010-2026 darktable developers.

Color Uniformity - CPU Engine and GUI
darktable UCS JCH

V8 - Additions over V7:
  - P0: Asymmetric priority (above/below) per component (H, C, L)
  - P1: Per-component normalized distance for affinity (instead of a
         shared global t_norm). Chroma affinity now depends on
         chroma distance, not on position in the hue/luma mask.
  - P2: Texture preservation (anti-posterization) per component.
         Slows convergence when the pixel is already close to the target.
  - UI rename: "preserve texture" -> "neutral zone" (affinity donut radius)
  - Advanced sections in correction tab (GtkExpander per component)
*/

#include "bauhaus/bauhaus.h"
#include "common/chromatic_adaptation.h"
#include "common/colorspaces.h"
#include "common/colorspaces_inline_conversions.h"
#include "common/iop_profile.h"
#include "common/math.h"
#include "develop/imageop.h"
#include "develop/imageop_gui.h"
#include "dtgtk/gradientslider.h"
#include "gui/accelerators.h"
#include "gui/color_picker_proxy.h"
#include "gui/gtk.h"
#include "iop/iop_api.h"
#include "libs/lib.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Preview modes (for vectorscope and mask display)
typedef enum {
    PREVIEW_NONE             = 0,
    PREVIEW_MASK             = 1,
    PREVIEW_DELTA_MASK       = 2,  // Coloured mask: selection weight + correction amplitude
    PREVIEW_OVERLAY_BEFORE   = 3,
    PREVIEW_OVERLAY          = 4,
    PREVIEW_DELTA            = 5
} dt_iop_coloruniformityv2_preview_mode_t;

// --- 1. PARAMETERS AND INTROSPECTION (VERSION 12) ---

// clang-format off
typedef struct dt_iop_coloruniformityv2_params_t {
  // --- CORRECTION SOURCE ANCHOR ---
  // Not just selection: the correction references these (negative strength
  // pushes away from source, offset & affinity are relative to it).
  float source_hue;        // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.12 $DESCRIPTION: "Source Hue"
  float source_chroma;     // $MIN: 0.0 $MAX: 1.5 $DEFAULT: 0.20 $DESCRIPTION: "Source Chroma"
  float source_lightness;  // $MIN: 0.0 $MAX: 1.5 $DEFAULT: 0.50 $DESCRIPTION: "Source Lightness"

  // --- SELECTION: parametric ranges, 4 handles [edge_lo, in_lo, in_hi, edge_hi] ---
  // Each on the channel's normalized [0,1] axis (see process() for axis mappings).
  // Trapezoid factor a la darktable parametric mask (_blendif_compute_factor).
  // HUE: circular, symmetric arc = center + plateau half-width + falloff half-width.
  // (Handles wrapping natively; default = whole circle.)
  float hue_center;        // $MIN: 0.0 $MAX: 1.0  $DEFAULT: 0.08 $DESCRIPTION: "hue center"
  float hue_plateau;       // $MIN: 0.0 $MAX: 0.5  $DEFAULT: 0.5  $DESCRIPTION: "hue plateau half-width"
  float hue_falloff;       // $MIN: 0.0 $MAX: 0.5  $DEFAULT: 0.0  $DESCRIPTION: "hue falloff half-width"
  // CHROMA axis (c_pix / C_MAX) -- default: exclude near-neutral, no upper limit
  float chroma_h0;         // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.02 $DESCRIPTION: "chroma edge low"
  float chroma_h1;         // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.05 $DESCRIPTION: "chroma low"
  float chroma_h2;         // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 1.0  $DESCRIPTION: "chroma high"
  float chroma_h3;         // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 1.0  $DESCRIPTION: "chroma edge high"
  // LIGHTNESS axis (perceptual, [0,1]) -- default: full range
  float light_h0;          // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0  $DESCRIPTION: "lightness edge low"
  float light_h1;          // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0  $DESCRIPTION: "lightness low"
  float light_h2;          // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 1.0  $DESCRIPTION: "lightness high"
  float light_h3;          // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 1.0  $DESCRIPTION: "lightness edge high"

  // --- CORRECTION (Target T) ---
  float target_hue;        // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.12 $DESCRIPTION: "Target Hue"
  float target_chroma;     // $MIN: 0.0 $MAX: 1.5 $DEFAULT: 0.20 $DESCRIPTION: "Target Chroma"
  float target_lightness;  // $MIN: 0.0 $MAX: 1.5 $DEFAULT: 0.50 $DESCRIPTION: "Target Lightness"

  // --- HUE CORRECTION: strength + independent affinity ---
  float strength_h;        // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "Hue Strength"
  float affinity_h;        // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "Hue Affinity"
  float preserve_h;        // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.50 $DESCRIPTION: "Hue Neutral Zone"

  // --- CHROMA CORRECTION: strength + independent affinity ---
  float strength_c;        // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "Chroma Strength"
  float affinity_c;        // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "Chroma Affinity"
  float preserve_c;        // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.50 $DESCRIPTION: "Chroma Neutral Zone"

  // --- LIGHTNESS CORRECTION: strength + independent affinity ---
  float strength_l;        // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "Lightness Strength"
  float affinity_l;        // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "Lightness Affinity"
  float preserve_l;        // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.50 $DESCRIPTION: "Lightness Neutral Zone"

  // --- GLOBAL OFFSETS ---
  float offset_h;          // $MIN: -0.5 $MAX: 0.5 $DEFAULT: 0.0 $DESCRIPTION: "Hue Offset"
  float offset_c;          // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "Chroma Offset"
  float offset_l;          // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "Lightness Offset"
  int   offsets_weighted;   // $MIN: 0 $MAX: 1 $DEFAULT: 0 $DESCRIPTION: "Offsets Weighted"

  // --- BYPASS ---
  int bypass_hue;          // $MIN: 0 $MAX: 1 $DEFAULT: 0 $DESCRIPTION: "Bypass Hue"
  int bypass_chroma;       // $MIN: 0 $MAX: 1 $DEFAULT: 0 $DESCRIPTION: "Bypass Chroma"
  int bypass_luma;         // $MIN: 0 $MAX: 1 $DEFAULT: 0 $DESCRIPTION: "Bypass Luma"

  // --- UI STATE ---
  int preview_mode;        // $MIN: 0 $MAX: 5 $DEFAULT: 0 $DESCRIPTION: "Preview Mode"

  // --- V12: ASYMMETRIC PRIORITY (P0) ---
  float priority_h;          // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "Hue Priority"
  float priority_c;          // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "Chroma Priority"
  float priority_l;          // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "Lightness Priority"

  // --- V12: TEXTURE PRESERVATION (P2) ---
  float preserve_texture_h;  // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "Hue Preserve Texture"
  float preserve_texture_c;  // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "Chroma Preserve Texture"
  float preserve_texture_l;  // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "Lightness Preserve Texture"
} dt_iop_coloruniformityv2_params_t;
// clang-format on

DT_MODULE_INTROSPECTION(1, dt_iop_coloruniformityv2_params_t)


// Legacy params migration V9 -> V12 (via V10 -> V11 -> V12)


typedef struct dt_iop_coloruniformityv2_gui_data_t {
  // Selection (parametric ranges, darktable-style 4-handle sliders)
  GtkWidget *source_hue_picker;   // standalone pipette: sets anchor + recenters ranges
  // hue: symmetric circular arc (temp bauhaus sliders; circular ring widget to come)
  GtkWidget *hue_center;
  GtkWidget *hue_plateau;
  GtkWidget *hue_falloff;
  GtkWidget *chroma_slider;       // 4-handle gradient slider (chroma)
  GtkWidget *light_slider;        // 4-handle gradient slider (lightness)

  // Correction
  GtkWidget *target_hue_picker;
  GtkWidget *target_hue_slider;
  GtkWidget *target_chroma;
  GtkWidget *target_lightness;

  GtkWidget *strength_h;
  GtkWidget *affinity_h;
  GtkWidget *preserve_h;
  GtkWidget *priority_h;           // V12
  GtkWidget *preserve_texture_h;   // V12

  GtkWidget *strength_c;
  GtkWidget *affinity_c;
  GtkWidget *preserve_c;
  GtkWidget *priority_c;           // V12
  GtkWidget *preserve_texture_c;   // V12

  GtkWidget *strength_l;
  GtkWidget *affinity_l;
  GtkWidget *preserve_l;
  GtkWidget *priority_l;           // V12
  GtkWidget *preserve_texture_l;   // V12

  // Adjustments
  GtkWidget *offset_h;
  GtkWidget *offset_c;
  GtkWidget *offset_l;
  GtkWidget *offsets_weighted_combo;

  GtkWidget *btn_bypass_hue, *btn_bypass_chroma, *btn_bypass_luma;
  GtkWidget *preview_mode_combo;
  
  GtkNotebook *notebook;
  
} dt_iop_coloruniformityv2_gui_data_t;


const char *name() { return _("color uniformity v2"); }
const char **description(dt_iop_module_t *self) {
  return dt_iop_set_description(
      self, _("harmonize skin tone colors, creative color grading"),
      _("creative, color grading"), _("linear, RGB, scene-referred"),
      _("darktable UCS JCH perceptual"), _("linear, RGB, scene-referred"));
}
int flags() {
  return IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_SUPPORTS_BLENDING;
}
int groups() { return IOP_GROUP_COLOR; }
int priority() { return 417; }
dt_iop_colorspace_type_t default_colorspace(dt_iop_module_t *self,
                                            dt_dev_pixelpipe_t *pipe,
                                            dt_dev_pixelpipe_iop_t *piece) {
  return IOP_CS_RGB;
}



void commit_params(dt_iop_module_t *self, dt_iop_params_t *p1,
                   dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece) {
  memcpy(piece->data, p1, self->params_size);
}

// --- 3. CPU PROCESSING LOGIC (ENGINE V8) ---

static inline float wrap_hue(float h) {
  h = fmodf(h, 1.0f);
  if (h < 0.0f) h += 1.0f;
  return h;
}

// Shortest angular distance [-0.5, 0.5]
static inline float hue_dist(const float h1, const float h2) {
  float d = fmodf(h1 - h2, 1.0f);
  if (d > 0.5f) d -= 1.0f;
  else if (d < -0.5f) d += 1.0f;
  return d;
}

// Input safety -- tolerates slight negative values from chromatic adaptation
static inline gboolean _rgb_invalid(const dt_aligned_pixel_t rgb) {
  return !isfinite(rgb[0]) || !isfinite(rgb[1]) || !isfinite(rgb[2]) ||
         rgb[0] < -1e-3f || rgb[1] < -1e-3f || rgb[2] < -1e-3f;
}

// --- Selection engine: parametric ranges (darktable-style 4-handle trapezoid) ---

// Chroma axis: normalize Yrg chroma to the channel's [0,1] display axis.
#define CU2_CHROMA_MAX 0.7f
static inline float _chroma_axis(const float c) { return CLAMPF(c / CU2_CHROMA_MAX, 0.0f, 1.0f); }
// Lightness axis: perceptual dt UCS L* (display lightness), normalized to white=1.
// (UCS is used here for LUMA only; the hue metric is Yrg -- see process().)
static inline float _light_axis(const float Y)
{
  return CLAMPF(Y_to_dt_UCS_L_star(fmaxf(Y, 0.0f)) / Y_to_dt_UCS_L_star(1.0f), 0.0f, 1.0f);
}

// Linear trapezoid factor from 4 handles [p0 edge_lo, p1 in_lo, p2 in_hi, p3 edge_hi].
// (Same shape as darktable's _blendif_compute_factor.)
static inline float _param_factor(float v, const float p0, const float p1,
                                  const float p2, const float p3)
{
  v = CLAMPF(v, 0.0f, 1.0f);
  float f = 1.0f;
  // low side: only cuts if the low handles are above 0 (handle at 0 => no low cut,
  // so clamped extremes and full-range selections pass through)
  if(v < p1) f = (v <= p0) ? 0.0f : (v - p0) / fmaxf(p1 - p0, 1e-6f);
  // high side: only cuts if the high handles are below 1 (handle at 1 => no high cut)
  if(v > p2)
  {
    const float hf = (v >= p3) ? 0.0f : 1.0f - (v - p2) / fmaxf(p3 - p2, 1e-6f);
    f = fminf(f, hf);
  }
  return f;
}

// Hue: symmetric circular arc. 1 within the plateau half-width of the center,
// linear ramp to 0 over the falloff half-width, 0 beyond. Wraps natively.
static inline float _hue_arc_factor(const float h, const float center,
                                    const float plateau, const float falloff)
{
  const float d = fabsf(hue_dist(h, center)); // circular distance in [0, 0.5]
  if(d <= plateau) return 1.0f;
  if(d >= plateau + falloff) return 0.0f;
  return 1.0f - (d - plateau) / fmaxf(falloff, 1e-6f);
}

// Full selection weight = hue arc factor x chroma trapezoid x lightness trapezoid.
static inline float compute_selection_weight(
    const float h_pix, const float chroma_axis, const float light_axis,
    const float hue_center, const float hue_plateau, const float hue_falloff,
    const float *const chroma_h, const float *const light_h)
{
  const float fh = _hue_arc_factor(h_pix, hue_center, hue_plateau, hue_falloff);
  if(fh <= 0.0f) return 0.0f;
  const float fc = _param_factor(chroma_axis, chroma_h[0], chroma_h[1], chroma_h[2], chroma_h[3]);
  if(fc <= 0.0f) return 0.0f;
  const float fl = _param_factor(light_axis, light_h[0], light_h[1], light_h[2], light_h[3]);
  return fh * fc * fl;
}

// Apply affinity (peak or donut) to a base weight, per component.
// base_weight: raw selection (pizza slice * luma)
// affinity > 0 -> peak (reinforces center)
// affinity < 0 -> donut (hollows out center)
// preserve: neutral zone radius for donut mode
// t_norm: per-component normalized distance (0 = center, ~1 = edge)
static inline float apply_affinity(const float base_weight, const float affinity,
                                   const float preserve, const float t_norm)
{
  if(base_weight < 1e-6f) return 0.0f;
  if(affinity >= 0.0f) {
    return base_weight * (1.0f + affinity * base_weight);
  } else {
    const float safe_preserve = fmaxf(preserve, 1e-3f);
    const float hole = expf(-t_norm / (safe_preserve * safe_preserve));
    return base_weight * (1.0f - fabsf(affinity) * hole);
  }
}

void process(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece,
             const void *const ivoid, void *const ovoid,
             const dt_iop_roi_t *const roi_in,
             const dt_iop_roi_t *const roi_out) 
{
  const dt_iop_coloruniformityv2_params_t *const p = piece->data;
  const size_t ch = piece->colors;
  if (!dt_iop_have_required_input_format(4, self, piece->colors, ivoid, ovoid, roi_in, roi_out)) return;
  const dt_iop_order_iccprofile_info_t *const wp = dt_ioppr_get_pipe_work_profile_info(piece->pipe);
  if (!wp) return;

  const int width = roi_out->width;
  const int height = roi_out->height;

  // Pre-computed matrices
  dt_colormatrix_t mat_xyz_to_lms, mat_lms_to_xyz;
  dt_colormatrix_transpose(mat_xyz_to_lms, XYZ_D65_to_LMS_2006_D65);
  dt_colormatrix_transpose(mat_lms_to_xyz, LMS_2006_D65_to_XYZ_D65);

  // Checker pattern for PREVIEW_OVERLAY
  const size_t checker_size = MAX((size_t)DT_PIXEL_APPLY_DPI(8), (size_t)2);
  const size_t checker_period = 2 * checker_size;
  const float checker_dark  = 0.18f;
  const float checker_light = 0.45f;

  // D65 neutral point in the (r, g) plane of Yrg space.
  const dt_aligned_pixel_t d65_xyz = {0.95047f, 1.0f, 1.08883f, 0.0f};
  dt_aligned_pixel_t d65_lms, d65_yrg;
  dt_apply_transposed_color_matrix(d65_xyz, mat_xyz_to_lms, d65_lms);
  LMS_to_Yrg(d65_lms, d65_yrg);
  const float rn = d65_yrg[1];
  const float gn = d65_yrg[2];

  DT_OMP_FOR() 
  for (int j = 0; j < height; j++) {
    const float *in_ptr = (const float *)ivoid + (size_t)ch * width * j;
    float *out_ptr = (float *)ovoid + (size_t)ch * width * j;

    for (int i = 0; i < width; i++, in_ptr += ch, out_ptr += ch) {
      dt_aligned_pixel_t px_rgb = {in_ptr[0], in_ptr[1], in_ptr[2], 0.0f};

      // Input safety check
      if (_rgb_invalid(px_rgb)) {
        for (int c = 0; c < 4; c++) out_ptr[c] = in_ptr[c];
        continue;
      }

      // --- A. CONVERT TO WORKING SPACE (Filmlight Yrg Ych, unified) ---
      dt_aligned_pixel_t px_xyz, px_xyz_d65;
      dt_apply_transposed_color_matrix(px_rgb, wp->matrix_in_transposed, px_xyz);
      XYZ_D50_to_D65(px_xyz, px_xyz_d65);

      // Filmlight Yrg: luminance + chromaticity (r,g)
      dt_aligned_pixel_t px_lms, px_yrg;
      dt_apply_transposed_color_matrix(px_xyz_d65, mat_xyz_to_lms, px_lms);
      LMS_to_Yrg(px_lms, px_yrg);

      const float Y_pix = px_yrg[0];

      // Pixel chroma (distance to D65 neutral in the r,g plane)
      const float dr_pix = px_yrg[1] - rn;
      const float dg_pix = px_yrg[2] - gn;
      const float c_pix = sqrtf(dr_pix*dr_pix + dg_pix*dg_pix);

      // Yrg hue: angle in the r,g plane around D65 neutral, in [0,1] turns
      const float h_pix = wrap_hue(atan2f(dg_pix, dr_pix) / (2.0f * M_PI_F));

      // --- B. SELECTION WEIGHT (parametric ranges: product of 3 trapezoid factors) ---
      // Neutral protection is now implicit in the chroma channel's low handles.
      const float chroma_axis = _chroma_axis(c_pix);
      const float light_axis  = _light_axis(Y_pix);
      const float chroma_h[4] = { p->chroma_h0, p->chroma_h1, p->chroma_h2, p->chroma_h3 };
      const float light_h[4]  = { p->light_h0, p->light_h1, p->light_h2, p->light_h3 };
      float weight = compute_selection_weight(h_pix, chroma_axis, light_axis,
                                              p->hue_center, p->hue_plateau, p->hue_falloff,
                                              chroma_h, light_h);

      // --- Per-component normalized distance for affinity (relative to the source anchor,
      // normalized by the selection range half-width on each channel's axis) ---
      const float hw_h = fmaxf(p->hue_plateau + p->hue_falloff, 1e-3f);
      const float t_norm_h = fabsf(hue_dist(h_pix, p->source_hue)) / hw_h;
      const float hw_c = fmaxf((chroma_h[3] - chroma_h[0]) * 0.5f, 1e-3f);
      const float t_norm_c = fabsf(chroma_axis - _chroma_axis(p->source_chroma)) / hw_c;
      const float hw_l = fmaxf((light_h[3] - light_h[0]) * 0.5f, 1e-3f);
      const float t_norm_l = fabsf(light_axis - _light_axis(p->source_lightness)) / hw_l;

      // --- EARLY EXIT ---
      if (weight < 1e-4f) {
        switch(p->preview_mode) {
          case PREVIEW_MASK:
            for(int c = 0; c < 3; c++) out_ptr[c] = 0.0f;
            break;
          case PREVIEW_OVERLAY:
          case PREVIEW_OVERLAY_BEFORE: {
            const gboolean row_odd = ((size_t)j % checker_period) < checker_size;
            const gboolean col_odd = ((size_t)i % checker_period) < checker_size;
            const float checker = (row_odd ^ col_odd) ? checker_light : checker_dark;
            for(int c = 0; c < 3; c++) out_ptr[c] = checker;
            break;
          }
          case PREVIEW_DELTA:
            // Neutral grey: no correction on this pixel
            out_ptr[0] = out_ptr[1] = out_ptr[2] = 0.18f;
            break;
          case PREVIEW_DELTA_MASK:
            // Black: pixel outside selection
            out_ptr[0] = out_ptr[1] = out_ptr[2] = 0.0f;
            break;
          default: // PREVIEW_NONE
            for(int c = 0; c < 3; c++) out_ptr[c] = in_ptr[c];
            break;
        }
        out_ptr[3] = in_ptr[3];
        continue;
      }
      // Passthrough if no corrections configured and no preview
      if (p->preview_mode == PREVIEW_NONE
          && p->strength_h == 0.f && p->strength_c == 0.f && p->strength_l == 0.f
          && p->offset_h == 0.f   && p->offset_c == 0.f   && p->offset_l == 0.f) {
        for(int c = 0; c < 4; c++) out_ptr[c] = in_ptr[c];
        continue;
      }

      // --- C. CORRECTIONS ---

      const float off_w = p->offsets_weighted ? weight : 1.0f;
      
      // Per-component weights with independent affinity and per-component t_norm (P1)
      const float w_h = apply_affinity(weight, p->affinity_h, p->preserve_h, t_norm_h);
      const float w_c = apply_affinity(weight, p->affinity_c, p->preserve_c, t_norm_c);
      const float w_l = apply_affinity(weight, p->affinity_l, p->preserve_l, t_norm_l);

      // ===============================================================
      // 1. Hue correction (UCS space)
      // ===============================================================
      // The hue delta is computed in UCS JCH (perceptually
      // uniform for hue distances), then applied as a
      // 2D rotation in the Yrg chromaticity plane (r, g) around
      // the D65 neutral point. This eliminates inter-space round-trips
      // and removes chroma/luminance drift.
      float delta_h = 0.0f;
      if (!p->bypass_hue) {
        float h_side;

        if (p->strength_h >= 0.f) {
          const float gap = hue_dist(p->target_hue, h_pix);  // signed distance to target
          delta_h = gap * w_h * p->strength_h;
          // Anti-overshoot: never correct beyond the target
          if(fabsf(delta_h) > fabsf(gap))
            delta_h = gap;
          h_side = hue_dist(h_pix, p->target_hue);
        } else {
          delta_h = hue_dist(h_pix, p->source_hue) * w_h * fabsf(p->strength_h);
          h_side = hue_dist(h_pix, p->source_hue);
        }

        // P2: Texture preservation (convergence only)
        if(p->preserve_texture_h > 0.0f && p->strength_h >= 0.0f)
        {
          const float d_norm = fabsf(hue_dist(h_pix, p->target_hue)) / hw_h;
          delta_h *= 1.0f - p->preserve_texture_h * expf(-d_norm * d_norm * 8.0f);
        }

        // P0: Asymmetric priority
        if(p->priority_h != 0.0f && fabsf(h_side) > 1e-6f)
        {
          if(h_side * p->priority_h < 0.0f)
            delta_h *= fmaxf(0.0f, 1.0f - fabsf(p->priority_h));
        }
      }
      // Add offset (even if bypass_hue is on, offset_h is controlled separately)
      delta_h += (p->bypass_hue ? 0.0f : p->offset_h * off_w);

      // Apply hue rotation as a 2D rotation in the Yrg chromaticity plane.
      // delta_h is in normalized hue turns [-0.5, +0.5], convert to radians.
      // The rotation is around the D65 neutral point (rn, gn).
      // This preserves Y and chroma magnitude exactly -- no round-trip needed.
      if(fabsf(delta_h) > 1e-7f)
      {
        const float angle = delta_h * 2.0f * M_PI_F;
        const float cos_a = cosf(angle);
        const float sin_a = sinf(angle);
        const float dr = px_yrg[1] - rn;
        const float dg = px_yrg[2] - gn;
        px_yrg[1] = rn + dr * cos_a - dg * sin_a;
        px_yrg[2] = gn + dr * sin_a + dg * cos_a;
      }

      const float Y_base = px_yrg[0];
      const float r_base = px_yrg[1];
      const float g_base = px_yrg[2];
      const float dr_base = r_base - rn;
      const float dg_base = g_base - gn;
      const float c_base = sqrtf(dr_base*dr_base + dg_base*dg_base);

      // ===============================================================
      // 2. Chroma correction (Yrg space)
      // ===============================================================
      float c_corr = c_base;
      if (!p->bypass_chroma) {
        float delta_c;
        float c_side;

        if (p->strength_c >= 0.f) {
          const float gap_c = p->target_chroma - c_base;
          delta_c = gap_c * w_c * p->strength_c;
          if(fabsf(delta_c) > fabsf(gap_c))
            delta_c = gap_c;
          c_side = (c_base > p->target_chroma) ? 1.0f : -1.0f;
        } else {
          delta_c = (c_base - p->source_chroma) * w_c * fabsf(p->strength_c);
          c_side = (c_base > p->source_chroma) ? 1.0f : -1.0f;
        }

        if(p->preserve_texture_c > 0.0f && p->strength_c >= 0.0f)
        {
          const float ref_c = fmaxf(p->source_chroma, 1e-4f);
          const float d_norm = fabsf(c_base - p->target_chroma) / ref_c;
          delta_c *= 1.0f - p->preserve_texture_c * expf(-d_norm * d_norm * 8.0f);
        }

        if(p->priority_c != 0.0f)
        {
          if(c_side * p->priority_c < 0.0f)
            delta_c *= fmaxf(0.0f, 1.0f - fabsf(p->priority_c));
        }

        c_corr += delta_c;
        c_corr *= fmaxf(1.0f + p->offset_c * off_w, 0.0f);
      }
      c_corr = fmaxf(c_corr, 0.0f);

      // ===============================================================
      // 3. Lightness correction (Log2 stops)
      // ===============================================================
      float log_Y = log2f(fmaxf(Y_base, 1e-6f));
      float log_Y_corr = log_Y;
      if (!p->bypass_luma) {
        float delta_l;
        float l_side;

        const float log_target = log2f(fmaxf(p->target_lightness, 1e-6f));
        const float log_source = log2f(fmaxf(p->source_lightness, 1e-6f));

        if (p->strength_l >= 0.f) {
          const float gap_l = log_target - log_Y;
          delta_l = gap_l * w_l * p->strength_l;
          // Anti-overshoot
          if(fabsf(delta_l) > fabsf(gap_l))
            delta_l = gap_l;
          l_side = (log_Y > log_target) ? 1.0f : -1.0f;
        } else {
          delta_l = (log_Y - log_source) * w_l * fabsf(p->strength_l);
          l_side = (log_Y > log_source) ? 1.0f : -1.0f;
        }

        // P2: Texture preservation
        if(p->preserve_texture_l > 0.0f && p->strength_l >= 0.0f)
        {
          // approximate EV span of the lightness selection (full [0,1] axis ~ 6 EV)
          const float range = fmaxf((light_h[3] - light_h[0]) * 6.0f, 1e-3f);
          const float d_norm = fabsf(log_Y - log_target) / range;
          delta_l *= 1.0f - p->preserve_texture_l * expf(-d_norm * d_norm * 8.0f);
        }

        // P0: Asymmetric priority
        if(p->priority_l != 0.0f)
        {
          if(l_side * p->priority_l < 0.0f)
            delta_l *= fmaxf(0.0f, 1.0f - fabsf(p->priority_l));
        }

        log_Y_corr += delta_l;
        log_Y_corr += p->offset_l * off_w;
      }
      float Y_corr = exp2f(log_Y_corr);

      // Yrg reconstruction
      float ratio = (c_base > 1e-5f) ? (c_corr / c_base) : 0.0f;
      px_yrg[0] = fmaxf(Y_corr, 0.0f);
      px_yrg[1] = rn + dr_base * ratio;
      px_yrg[2] = gn + dg_base * ratio;

      // --- D. BACK TO RGB AND PREVIEWS ---
      Yrg_to_LMS(px_yrg, px_lms);
      dt_apply_transposed_color_matrix(px_lms, mat_lms_to_xyz, px_xyz_d65);
      XYZ_D65_to_D50(px_xyz_d65, px_xyz);
      
      dt_aligned_pixel_t px_rgb_out;
      dt_apply_transposed_color_matrix(px_xyz, wp->matrix_out_transposed, px_rgb_out);
      for(int c = 0; c < 3; c++) px_rgb_out[c] = fminf(px_rgb_out[c], 1e6f);

      // UI preview modes
      if (p->preview_mode == PREVIEW_MASK) {
        for(int c = 0; c < 3; c++) out_ptr[c] = weight;
      }
      else if (p->preview_mode == PREVIEW_OVERLAY || p->preview_mode == PREVIEW_OVERLAY_BEFORE) {
        const gboolean row_odd = ((size_t)j % checker_period) < checker_size;
        const gboolean col_odd = ((size_t)i % checker_period) < checker_size;
        const float checker = (row_odd ^ col_odd) ? checker_light : checker_dark;
        const float w_comp = 1.0f - weight;
        const float *src = (p->preview_mode == PREVIEW_OVERLAY) ? px_rgb_out : in_ptr;
        for(int c = 0; c < 3; c++)
          out_ptr[c] = w_comp * checker + weight * src[c];
      }
      else if (p->preview_mode == PREVIEW_DELTA) {
        // Delta visualisation: shows the corrected colour of each selected pixel,
        // with brightness proportional to the actual correction amplitude.
        // Unselected pixels and pixels with no active correction → neutral grey.
        // This way: hue = real corrected hue (no angular convention issues),
        //           brightness = how much the pixel is actually being moved.
        const float delta_c = c_corr - c_base;
        const float amp_h = fabsf(delta_h);
        const float amp_c = fabsf(delta_c) / fmaxf(p->source_chroma, 0.01f);
        const float amplitude = fminf((amp_h * 3.0f + amp_c) * weight, 1.0f);

        if(amplitude < 1e-4f)
        {
          // No correction active — neutral grey
          out_ptr[0] = out_ptr[1] = out_ptr[2] = 0.18f;
        }
        else
        {
          // Blend corrected colour toward grey based on amplitude:
          // amplitude=1 → full corrected colour; amplitude→0 → grey
          const float grey = 0.18f;
          for(int c = 0; c < 3; c++)
            out_ptr[c] = CLAMPF(grey + amplitude * (px_rgb_out[c] - grey), 0.0f, 1.0f);
        }
      }
      else if (p->preview_mode == PREVIEW_DELTA_MASK) {
        // Coloured mask: black outside selection (early exit), inside:
        //   colour     = actual corrected RGB of the pixel (immediately readable)
        //   brightness = selection weight (bright = strongly selected)
        // This shows exactly what colour each selected pixel is being pushed toward.
        const float J_scale = weight * 1.4f; // boost to make it visible
        for(int c = 0; c < 3; c++)
          out_ptr[c] = CLAMPF(px_rgb_out[c] * J_scale, 0.0f, 1.0f);
      }
      else {
        for(int c = 0; c < 3; c++) out_ptr[c] = px_rgb_out[c]; // PREVIEW_NONE
      }
      out_ptr[3] = in_ptr[3];
    }
  }
}






#ifdef HAVE_OPENCL

#endif

// --- SLIDER PAINTING HELPERS ---

static inline void _hue_to_rgb(const float h, float *r, float *g, float *b)
{
  const float h6 = fmodf(h, 1.0f) * 6.0f;
  const float c = 1.0f;
  const float x = 1.0f - fabsf(fmodf(h6, 2.0f) - 1.0f);
  float R, G, B;
  if      (h6 < 1.0f) { R = c; G = x; B = 0; }
  else if (h6 < 2.0f) { R = x; G = c; B = 0; }
  else if (h6 < 3.0f) { R = 0; G = c; B = x; }
  else if (h6 < 4.0f) { R = 0; G = x; B = c; }
  else if (h6 < 5.0f) { R = x; G = 0; B = c; }
  else                 { R = c; G = 0; B = x; }
  const float sat = 0.75f;
  const float grey = 0.15f;
  *r = R * sat + grey * (1.0f - sat);
  *g = G * sat + grey * (1.0f - sat);
  *b = B * sat + grey * (1.0f - sat);
}

static void _paint_hue_slider(GtkWidget *slider)
{
  for(int i = 0; i < DT_BAUHAUS_SLIDER_MAX_STOPS; i++)
  {
    const float stop = (float)i / (float)(DT_BAUHAUS_SLIDER_MAX_STOPS - 1);
    float r, g, b;
    _hue_to_rgb(stop, &r, &g, &b);
    dt_bauhaus_slider_set_stop(slider, stop, r, g, b);
  }
}

static void _paint_luma_slider(GtkWidget *slider)
{
  for(int i = 0; i < DT_BAUHAUS_SLIDER_MAX_STOPS; i++)
  {
    const float stop = (float)i / (float)(DT_BAUHAUS_SLIDER_MAX_STOPS - 1);
    dt_bauhaus_slider_set_stop(slider, stop, stop, stop, stop);
  }
}

static void _paint_chroma_slider(GtkWidget *slider, const float hue)
{
  float r_hue, g_hue, b_hue;
  _hue_to_rgb(hue, &r_hue, &g_hue, &b_hue);
  for(int i = 0; i < DT_BAUHAUS_SLIDER_MAX_STOPS; i++)
  {
    const float stop = (float)i / (float)(DT_BAUHAUS_SLIDER_MAX_STOPS - 1);
    const float grey = 0.30f;
    dt_bauhaus_slider_set_stop(slider, stop,
        grey + (r_hue - grey) * stop,
        grey + (g_hue - grey) * stop,
        grey + (b_hue - grey) * stop);
  }
  gtk_widget_queue_draw(slider);
}

static void _paint_strength_slider(GtkWidget *slider, float r_pos, float g_pos, float b_pos)
{
  for(int i = 0; i < DT_BAUHAUS_SLIDER_MAX_STOPS; i++)
  {
    const float stop = (float)i / (float)(DT_BAUHAUS_SLIDER_MAX_STOPS - 1);
    const float grey = 0.25f;
    if(stop < 0.5f) {
      const float t = stop * 2.0f;
      dt_bauhaus_slider_set_stop(slider, stop,
          0.15f + grey * t, 0.15f + grey * t, 0.20f + (grey - 0.05f) * t);
    } else {
      const float t = (stop - 0.5f) * 2.0f;
      dt_bauhaus_slider_set_stop(slider, stop,
          grey + (r_pos - grey) * t,
          grey + (g_pos - grey) * t,
          grey + (b_pos - grey) * t);
    }
  }
}

// Paint the hue priority slider with a dynamic gradient:
// left (-1) = counterclockwise color from target
// center (0) = neutral gray
// droite (+1) = couleur horaire de la cible
// Repaints whenever target_hue changes.
static void _paint_hue_priority_slider(GtkWidget *slider, const float target_hue)
{
  // Counterclockwise color (target_hue - offset) and clockwise (target_hue + offset)
  // Use ±0.08 turns (~±30°) so colors are clearly distinguishable
  const float offset = 0.08f;
  float r_lo, g_lo, b_lo;  // antihoraire (gauche, priority = -1)
  float r_hi, g_hi, b_hi;  // horaire (droite, priority = +1)
  _hue_to_rgb(fmodf(target_hue - offset + 1.0f, 1.0f), &r_lo, &g_lo, &b_lo);
  _hue_to_rgb(fmodf(target_hue + offset, 1.0f), &r_hi, &g_hi, &b_hi);

  const float grey = 0.30f;
  for(int i = 0; i < DT_BAUHAUS_SLIDER_MAX_STOPS; i++)
  {
    const float stop = (float)i / (float)(DT_BAUHAUS_SLIDER_MAX_STOPS - 1);
    float r, g, b;
    if(stop < 0.5f)
    {
      // Gauche -> centre : antihoraire -> gris
      const float t = stop * 2.0f;  // 0 to 1
      r = r_lo + (grey - r_lo) * t;
      g = g_lo + (grey - g_lo) * t;
      b = b_lo + (grey - b_lo) * t;
    }
    else
    {
      // Centre -> droite : gris -> horaire
      const float t = (stop - 0.5f) * 2.0f;  // 0 to 1
      r = grey + (r_hi - grey) * t;
      g = grey + (g_hi - grey) * t;
      b = grey + (b_hi - grey) * t;
    }
    dt_bauhaus_slider_set_stop(slider, stop, r, g, b);
  }
  gtk_widget_queue_draw(slider);
}

// ====================== AUTO BUTTONS ======================
static void _shift_hue_cb(GtkWidget *widget, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  dt_iop_coloruniformityv2_params_t *p = (dt_iop_coloruniformityv2_params_t *)self->params;
  dt_iop_coloruniformityv2_gui_data_t *g = (dt_iop_coloruniformityv2_gui_data_t *)self->gui_data;

  p->offset_h = hue_dist(p->target_hue, p->source_hue);

  ++darktable.gui->reset;
  dt_bauhaus_slider_set(g->offset_h, p->offset_h);
  --darktable.gui->reset;
  dt_dev_add_history_item(self->dev, self, TRUE);
}


// --- GUI CALLBACKS ---

// Recenter a linear 4-handle range so its plateau center -> target, keeping widths (clamped to [0,1]).
static inline void _recenter_linear(float *h, const float target)
{
  const float center = 0.5f * (h[1] + h[2]);
  float d = target - center;
  // limit shift so handles stay in [0,1] without collapsing the shape
  d = CLAMPF(d, -h[0], 1.0f - h[3]);
  for(int i = 0; i < 4; i++) h[i] = CLAMPF(h[i] + d, 0.0f, 1.0f);
}
void color_picker_apply(dt_iop_module_t *self, GtkWidget *picker, dt_dev_pixelpipe_t *pipe)
{
  dt_iop_coloruniformityv2_params_t *p = (dt_iop_coloruniformityv2_params_t *)self->params;
  dt_iop_coloruniformityv2_gui_data_t *g = (dt_iop_coloruniformityv2_gui_data_t *)self->gui_data;
  
  const dt_iop_order_iccprofile_info_t *wp = dt_ioppr_get_pipe_work_profile_info(pipe);
  if(!wp) return;

  dt_aligned_pixel_t px_rgb = {self->picked_color[0], self->picked_color[1], self->picked_color[2], 0.0f};
  
  dt_aligned_pixel_t px_xyz, px_xyz_d65;
  dt_apply_transposed_color_matrix(px_rgb, wp->matrix_in_transposed, px_xyz);
  XYZ_D50_to_D65(px_xyz, px_xyz_d65);

  dt_colormatrix_t mat_xyz_to_lms;
  dt_colormatrix_transpose(mat_xyz_to_lms, XYZ_D65_to_LMS_2006_D65);
  dt_aligned_pixel_t px_lms, px_yrg;
  dt_apply_transposed_color_matrix(px_xyz_d65, mat_xyz_to_lms, px_lms);
  LMS_to_Yrg(px_lms, px_yrg);

  const dt_aligned_pixel_t d65_xyz = {0.95047f, 1.0f, 1.08883f, 0.0f};
  dt_aligned_pixel_t d65_lms, d65_yrg;
  dt_apply_transposed_color_matrix(d65_xyz, mat_xyz_to_lms, d65_lms);
  LMS_to_Yrg(d65_lms, d65_yrg);

  const float Y_picked = fmaxf(px_yrg[0], 0.0f);
  const float dr = px_yrg[1] - d65_yrg[1];
  const float dg = px_yrg[2] - d65_yrg[2];
  const float c_picked = sqrtf(dr*dr + dg*dg);
  // Yrg hue of the picked color (same metric as process())
  const float h_picked = wrap_hue(atan2f(dg, dr) / (2.0f * M_PI_F));

  ++darktable.gui->reset;
  if(picker == g->source_hue_picker) {
    p->source_hue = h_picked;
    p->source_chroma = c_picked;
    p->source_lightness = Y_picked;
    // hue: center the arc on the picked hue (widths kept)
    p->hue_center = h_picked;
    dt_bauhaus_slider_set(g->hue_center, p->hue_center);
    // chroma / lightness: recenter the linear ranges on the picked color (widths kept)
    float ch[4] = { p->chroma_h0, p->chroma_h1, p->chroma_h2, p->chroma_h3 };
    _recenter_linear(ch, _chroma_axis(c_picked));
    p->chroma_h0 = ch[0]; p->chroma_h1 = ch[1]; p->chroma_h2 = ch[2]; p->chroma_h3 = ch[3];
    float lh[4] = { p->light_h0, p->light_h1, p->light_h2, p->light_h3 };
    _recenter_linear(lh, _light_axis(Y_picked));
    p->light_h0 = lh[0]; p->light_h1 = lh[1]; p->light_h2 = lh[2]; p->light_h3 = lh[3];
    double dc[4] = { ch[0], ch[1], ch[2], ch[3] };
    double dl[4] = { lh[0], lh[1], lh[2], lh[3] };
    dtgtk_gradient_slider_multivalue_set_values(DTGTK_GRADIENT_SLIDER_MULTIVALUE(g->chroma_slider), dc);
    dtgtk_gradient_slider_multivalue_set_values(DTGTK_GRADIENT_SLIDER_MULTIVALUE(g->light_slider), dl);
  } else if(picker == g->target_hue_picker) {
    p->target_hue = h_picked;
    p->target_chroma = c_picked;
    p->target_lightness = Y_picked;
    dt_bauhaus_slider_set(g->target_hue_slider, p->target_hue);
    dt_bauhaus_slider_set(g->target_chroma, p->target_chroma);
    dt_bauhaus_slider_set(g->target_lightness, p->target_lightness);
    _paint_chroma_slider(g->target_chroma, p->target_hue);
    _paint_hue_priority_slider(g->priority_h, p->target_hue);
  }
  --darktable.gui->reset;
  dt_dev_add_history_item(self->dev, self, TRUE);
}

static void _slider_changed_cb(GtkWidget *widget, gpointer user_data) {
  if (darktable.gui->reset) return;
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  dt_iop_coloruniformityv2_params_t *p = (dt_iop_coloruniformityv2_params_t *)self->params;
  dt_iop_coloruniformityv2_gui_data_t *g = (dt_iop_coloruniformityv2_gui_data_t *)self->gui_data;
  
  const float val = dt_bauhaus_slider_get(widget);
  
  // Hue arc (chroma/lightness ranges are handled by _range_changed_cb)
  if (widget == g->hue_center) p->hue_center = val;
  else if (widget == g->hue_plateau) p->hue_plateau = val;
  else if (widget == g->hue_falloff) p->hue_falloff = val;
  // Correction
  else if (widget == g->target_hue_slider) {
    p->target_hue = val;
    _paint_chroma_slider(g->target_chroma, val);
    _paint_hue_priority_slider(g->priority_h, val);
  }
  else if (widget == g->target_chroma) p->target_chroma = val;
  else if (widget == g->target_lightness) p->target_lightness = val;
  else if (widget == g->strength_h) p->strength_h = val;
  else if (widget == g->affinity_h) p->affinity_h = val;
  else if (widget == g->preserve_h) p->preserve_h = val;
  else if (widget == g->priority_h) p->priority_h = val;
  else if (widget == g->preserve_texture_h) p->preserve_texture_h = val;
  else if (widget == g->strength_c) p->strength_c = val;
  else if (widget == g->affinity_c) p->affinity_c = val;
  else if (widget == g->preserve_c) p->preserve_c = val;
  else if (widget == g->priority_c) p->priority_c = val;
  else if (widget == g->preserve_texture_c) p->preserve_texture_c = val;
  else if (widget == g->strength_l) p->strength_l = val;
  else if (widget == g->affinity_l) p->affinity_l = val;
  else if (widget == g->preserve_l) p->preserve_l = val;
  else if (widget == g->priority_l) p->priority_l = val;
  else if (widget == g->preserve_texture_l) p->preserve_texture_l = val;
  // Adjustments
  else if (widget == g->offset_h) p->offset_h = val;
  else if (widget == g->offset_c) p->offset_c = val;
  else if (widget == g->offset_l) p->offset_l = val;

  dt_dev_add_history_item(self->dev, self, TRUE);
}

static void _preview_mode_changed_cb(GtkComboBox *widget, gpointer user_data) {
  if (darktable.gui->reset) return;
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  dt_iop_coloruniformityv2_params_t *p = (dt_iop_coloruniformityv2_params_t *)self->params;
  p->preview_mode = gtk_combo_box_get_active(widget);
  dt_dev_add_history_item(self->dev, self, TRUE);
}

static void _offsets_weighted_changed_cb(GtkComboBox *widget, gpointer user_data) {
  if (darktable.gui->reset) return;
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  dt_iop_coloruniformityv2_params_t *p = (dt_iop_coloruniformityv2_params_t *)self->params;
  p->offsets_weighted = gtk_combo_box_get_active(widget);
  dt_dev_add_history_item(self->dev, self, TRUE);
}

// Magic button: Copy Source -> Target
static void _match_source_to_target_cb(GtkWidget *widget, gpointer user_data) {
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  dt_iop_coloruniformityv2_params_t *p = (dt_iop_coloruniformityv2_params_t *)self->params;
  dt_iop_coloruniformityv2_gui_data_t *g = (dt_iop_coloruniformityv2_gui_data_t *)self->gui_data;

  p->target_hue = p->source_hue;
  p->target_chroma = p->source_chroma;
  p->target_lightness = p->source_lightness;

  ++darktable.gui->reset;
  dt_bauhaus_slider_set(g->target_hue_slider, p->target_hue);
  dt_bauhaus_slider_set(g->target_chroma, p->target_chroma);
  dt_bauhaus_slider_set(g->target_lightness, p->target_lightness);
  _paint_chroma_slider(g->target_chroma, p->target_hue);
  _paint_hue_priority_slider(g->priority_h, p->target_hue);
  --darktable.gui->reset;

  dt_dev_add_history_item(self->dev, self, TRUE);
}

static void _bypass_hue_toggled(GtkWidget *widget, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  dt_iop_coloruniformityv2_params_t *p = self->params;
  p->bypass_hue = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget)) ? 1 : 0;
  dt_dev_add_history_item(self->dev, self, TRUE);
}

static void _bypass_chroma_toggled(GtkWidget *widget, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  dt_iop_coloruniformityv2_params_t *p = self->params;
  p->bypass_chroma = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget)) ? 1 : 0;
  dt_dev_add_history_item(self->dev, self, TRUE);
}

static void _bypass_luma_toggled(GtkWidget *widget, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  dt_iop_coloruniformityv2_params_t *p = self->params;
  p->bypass_luma = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget)) ? 1 : 0;
  dt_dev_add_history_item(self->dev, self, TRUE);
}

void gui_update(dt_iop_module_t *self) {
  dt_iop_coloruniformityv2_gui_data_t *g = self->gui_data;
  dt_iop_coloruniformityv2_params_t *p = self->params;
  ++darktable.gui->reset;
  
  // Hue arc (temp sliders)
  dt_bauhaus_slider_set(g->hue_center, p->hue_center);
  dt_bauhaus_slider_set(g->hue_plateau, p->hue_plateau);
  dt_bauhaus_slider_set(g->hue_falloff, p->hue_falloff);
  // Chroma / lightness ranges (4-handle gradient sliders)
  double dc[4] = { p->chroma_h0, p->chroma_h1, p->chroma_h2, p->chroma_h3 };
  double dl[4] = { p->light_h0, p->light_h1, p->light_h2, p->light_h3 };
  dtgtk_gradient_slider_multivalue_set_values(DTGTK_GRADIENT_SLIDER_MULTIVALUE(g->chroma_slider), dc);
  dtgtk_gradient_slider_multivalue_set_values(DTGTK_GRADIENT_SLIDER_MULTIVALUE(g->light_slider), dl);
  // Correction
  dt_bauhaus_slider_set(g->target_hue_slider, p->target_hue);
  dt_bauhaus_slider_set(g->target_chroma, p->target_chroma);
  dt_bauhaus_slider_set(g->target_lightness, p->target_lightness);
  dt_bauhaus_slider_set(g->strength_h, p->strength_h);
  dt_bauhaus_slider_set(g->affinity_h, p->affinity_h);
  dt_bauhaus_slider_set(g->preserve_h, p->preserve_h);
  dt_bauhaus_slider_set(g->priority_h, p->priority_h);
  dt_bauhaus_slider_set(g->preserve_texture_h, p->preserve_texture_h);
  dt_bauhaus_slider_set(g->strength_c, p->strength_c);
  dt_bauhaus_slider_set(g->affinity_c, p->affinity_c);
  dt_bauhaus_slider_set(g->preserve_c, p->preserve_c);
  dt_bauhaus_slider_set(g->priority_c, p->priority_c);
  dt_bauhaus_slider_set(g->preserve_texture_c, p->preserve_texture_c);
  dt_bauhaus_slider_set(g->strength_l, p->strength_l);
  dt_bauhaus_slider_set(g->affinity_l, p->affinity_l);
  dt_bauhaus_slider_set(g->preserve_l, p->preserve_l);
  dt_bauhaus_slider_set(g->priority_l, p->priority_l);
  dt_bauhaus_slider_set(g->preserve_texture_l, p->preserve_texture_l);
  // Adjustments
  dt_bauhaus_slider_set(g->offset_h, p->offset_h);
  dt_bauhaus_slider_set(g->offset_c, p->offset_c);
  dt_bauhaus_slider_set(g->offset_l, p->offset_l);
  gtk_combo_box_set_active(GTK_COMBO_BOX(g->offsets_weighted_combo), p->offsets_weighted);

  gtk_combo_box_set_active(GTK_COMBO_BOX(g->preview_mode_combo), p->preview_mode);

  // Bypass buttons
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->btn_bypass_hue),    p->bypass_hue);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->btn_bypass_chroma), p->bypass_chroma);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->btn_bypass_luma),   p->bypass_luma);

  _paint_chroma_slider(g->target_chroma, p->target_hue);
  _paint_hue_priority_slider(g->priority_h, p->target_hue);

  dt_iop_color_picker_reset(self, TRUE);
  --darktable.gui->reset;
}

static GtkWidget *_create_manual_slider(dt_iop_module_t *self, const char *label, float min, float max, float step, float def, int digits, const char *format, float factor) {
  GtkWidget *slider = dt_bauhaus_slider_new_with_range(self, min, max, step, def, digits);
  dt_bauhaus_widget_set_label(slider, NULL, label);
  if (format) dt_bauhaus_slider_set_format(slider, format);
  if (factor != 1.0f) dt_bauhaus_slider_set_factor(slider, factor);
  g_signal_connect(G_OBJECT(slider), "value-changed", G_CALLBACK(_slider_changed_cb), self);
  return slider;
}

// --- Parametric range slider (darktable-style 4-handle gradient slider) ---
static void _range_changed_cb(GtkWidget *widget, gpointer user_data)
{
  dt_iop_module_t *self = user_data;
  if(darktable.gui->reset) return;
  dt_iop_coloruniformityv2_params_t *p = self->params;
  dt_iop_coloruniformityv2_gui_data_t *g = self->gui_data;
  double v[4];
  dtgtk_gradient_slider_multivalue_get_values(DTGTK_GRADIENT_SLIDER_MULTIVALUE(widget), v);
  if(widget == g->chroma_slider)
  { p->chroma_h0 = v[0]; p->chroma_h1 = v[1]; p->chroma_h2 = v[2]; p->chroma_h3 = v[3]; }
  else if(widget == g->light_slider)
  { p->light_h0 = v[0]; p->light_h1 = v[1]; p->light_h2 = v[2]; p->light_h3 = v[3]; }
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static GtkWidget *_create_range_slider(dt_iop_module_t *self)
{
  GtkWidget *w = dtgtk_gradient_slider_multivalue_new_with_name(4, "coloruniformityv2-range");
  GtkDarktableGradientSlider *s = DTGTK_GRADIENT_SLIDER_MULTIVALUE(w);
  // 2 outer open handles (falloff edges) + 2 inner filled handles (plateau)
  dtgtk_gradient_slider_multivalue_set_marker(s, GRADIENT_SLIDER_MARKER_LOWER_OPEN_BIG, 0);
  dtgtk_gradient_slider_multivalue_set_marker(s, GRADIENT_SLIDER_MARKER_UPPER_FILLED_BIG, 1);
  dtgtk_gradient_slider_multivalue_set_marker(s, GRADIENT_SLIDER_MARKER_UPPER_FILLED_BIG, 2);
  dtgtk_gradient_slider_multivalue_set_marker(s, GRADIENT_SLIDER_MARKER_LOWER_OPEN_BIG, 3);
  dtgtk_gradient_slider_multivalue_set_increment(s, 0.001);
  g_signal_connect(G_OBJECT(w), "value-changed", G_CALLBACK(_range_changed_cb), self);
  return w;
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_coloruniformityv2_gui_data_t *g = IOP_GUI_ALLOC(coloruniformityv2);

  static dt_action_def_t notebook_def = { 0 };
  g->notebook = dt_ui_notebook_new(&notebook_def);
  dt_action_define_iop(self, NULL, N_("page"), GTK_WIDGET(g->notebook), &notebook_def);

  // Main container: preview_box + notebook
  GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, DT_BAUHAUS_SPACE);

  // Preview -- ABOVE the notebook tabs so it stays visible on all pages
  GtkWidget *preview_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, DT_BAUHAUS_SPACE);
  gtk_box_pack_start(GTK_BOX(preview_box), gtk_label_new(_("Preview:")), FALSE, FALSE, 0);

  g->preview_mode_combo = gtk_combo_box_text_new();
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(g->preview_mode_combo), _("Image"));
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(g->preview_mode_combo), _("B&W Mask"));
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(g->preview_mode_combo), _("Colour Mask"));
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(g->preview_mode_combo), _("Checker (before)"));
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(g->preview_mode_combo), _("Checker (after)"));
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(g->preview_mode_combo), _("Correction"));
  g_signal_connect(G_OBJECT(g->preview_mode_combo), "changed",
                   G_CALLBACK(_preview_mode_changed_cb), self);
  gtk_box_pack_start(GTK_BOX(preview_box), g->preview_mode_combo, TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(vbox), preview_box, FALSE, FALSE, 0);

  gtk_box_pack_start(GTK_BOX(vbox), GTK_WIDGET(g->notebook), TRUE, TRUE, 0);

  self->widget = vbox;

  // =========================================================================
  // TAB 1: SELECTION
  // =========================================================================
  GtkWidget *page_sel = dt_ui_notebook_page(g->notebook,
                                            N_("selection"),
                                            _("source color selection"));

  dt_gui_box_add(page_sel, dt_ui_section_label_new(_("source color")));

  g->source_hue_picker = dt_color_picker_new(self, DT_COLOR_PICKER_AREA, NULL);
  gtk_widget_set_tooltip_text(g->source_hue_picker,
    _("pick a source color: sets the correction anchor and\n"
      "re-centers the hue / chroma / lightness selection ranges on it"));
  dt_gui_box_add(page_sel, g->source_hue_picker);

  dt_gui_box_add(page_sel, dt_ui_section_label_new(_("selection ranges")));

  // HUE: symmetric circular arc (center + plateau/falloff half-widths).
  // Temporary sliders -- a circular ring widget will replace them; the model
  // already selects wrapping arcs (e.g. reds around 0 deg) correctly.
  dt_gui_box_add(page_sel, dt_ui_label_new(_("hue (circular arc)")));
  g->hue_center  = _create_manual_slider(self, _("hue center"), 0.0f, 1.0f, 0.001f, 0.08f, 1, "°", 360.0f);
  _paint_hue_slider(g->hue_center);
  dt_gui_box_add(page_sel, g->hue_center);
  g->hue_plateau = _create_manual_slider(self, _("hue plateau"), 0.0f, 0.5f, 0.001f, 0.5f, 1, "°", 360.0f);
  gtk_widget_set_tooltip_text(g->hue_plateau, _("half-width of the full-weight hue arc"));
  dt_gui_box_add(page_sel, g->hue_plateau);
  g->hue_falloff = _create_manual_slider(self, _("hue falloff"), 0.0f, 0.5f, 0.001f, 0.0f, 1, "°", 360.0f);
  gtk_widget_set_tooltip_text(g->hue_falloff, _("extra half-width of the hue transition to zero"));
  dt_gui_box_add(page_sel, g->hue_falloff);

  // CHROMA: linear 4-handle range + gray->saturated gradient reference
  dt_gui_box_add(page_sel, dt_ui_label_new(_("chroma")));
  g->chroma_slider = _create_range_slider(self);
  {
    GtkDarktableGradientSlider *s = DTGTK_GRADIENT_SLIDER_MULTIVALUE(g->chroma_slider);
    GdkRGBA g0 = { 0.5, 0.5, 0.5, 1.0 }, g1 = { 0.9, 0.2, 0.2, 1.0 };
    dtgtk_gradient_slider_multivalue_set_stop(s, 0.0, g0);
    dtgtk_gradient_slider_multivalue_set_stop(s, 1.0, g1);
  }
  gtk_widget_set_tooltip_text(g->chroma_slider,
    _("chroma (saturation) selection range.\n"
      "the low handles also exclude neutral/gray pixels"));
  dt_gui_box_add(page_sel, g->chroma_slider);

  // LIGHTNESS: linear 4-handle range + black->white gradient reference
  dt_gui_box_add(page_sel, dt_ui_label_new(_("lightness")));
  g->light_slider = _create_range_slider(self);
  {
    GtkDarktableGradientSlider *s = DTGTK_GRADIENT_SLIDER_MULTIVALUE(g->light_slider);
    GdkRGBA b0 = { 0.0, 0.0, 0.0, 1.0 }, b1 = { 1.0, 1.0, 1.0, 1.0 };
    dtgtk_gradient_slider_multivalue_set_stop(s, 0.0, b0);
    dtgtk_gradient_slider_multivalue_set_stop(s, 1.0, b1);
  }
  gtk_widget_set_tooltip_text(g->light_slider,
    _("lightness selection range (perceptual)"));
  dt_gui_box_add(page_sel, g->light_slider);

  // =========================================================================
  // TAB 2: CORRECTION
  // =========================================================================
  GtkWidget *page_cor = dt_ui_notebook_page(g->notebook,
                                            N_("correction"),
                                            _("target color and convergence strength"));

  GtkWidget *target_hdr = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, DT_BAUHAUS_SPACE);
  gtk_box_pack_start(GTK_BOX(target_hdr),
                     dt_ui_section_label_new(_("Target Color T")),
                     TRUE, TRUE, 0);

  GtkWidget *btn_match = gtk_button_new_with_label(_("S -> T"));
  g_signal_connect(G_OBJECT(btn_match), "clicked",
                   G_CALLBACK(_match_source_to_target_cb), self);
  gtk_box_pack_end(GTK_BOX(target_hdr), btn_match, FALSE, FALSE, 0);
  dt_gui_box_add(page_cor, target_hdr);

  g->target_hue_slider = _create_manual_slider(self, _("hue"),
                                               0.0f, 1.0f, 0.001f, 0.12f, 2, "°", 360.0f);
  dt_bauhaus_slider_set_feedback(g->target_hue_slider, 0);
  _paint_hue_slider(g->target_hue_slider);
  g->target_hue_picker = dt_color_picker_new(self, DT_COLOR_PICKER_AREA, g->target_hue_slider);
  dt_gui_box_add(page_cor, g->target_hue_picker);

  g->target_chroma = _create_manual_slider(self, _("chroma"),
                                           0.0f, 1.5f, 0.001f, 0.20f, 2, "%", 100.0f);
  _paint_chroma_slider(g->target_chroma, 0.12f);
  dt_gui_box_add(page_cor, g->target_chroma);

  g->target_lightness = _create_manual_slider(self, _("lightness"),
                                              0.0f, 1.5f, 0.001f, 0.50f, 2, "%", 100.0f);
  _paint_luma_slider(g->target_lightness);
  dt_gui_box_add(page_cor, g->target_lightness);

  // === Bypass bar: Hue / Chroma / Lightness ===
  {
    GtkWidget *hb = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);

    g->btn_bypass_hue = gtk_toggle_button_new_with_label(_("hue"));
    gtk_widget_set_tooltip_text(g->btn_bypass_hue, _("disable hue correction"));
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->btn_bypass_hue),
      ((const dt_iop_coloruniformityv2_params_t *)self->params)->bypass_hue);
    g_signal_connect(G_OBJECT(g->btn_bypass_hue), "toggled", G_CALLBACK(_bypass_hue_toggled), self);
    gtk_box_pack_start(GTK_BOX(hb), g->btn_bypass_hue, TRUE, TRUE, 0);

    g->btn_bypass_chroma = gtk_toggle_button_new_with_label(_("chroma"));
    gtk_widget_set_tooltip_text(g->btn_bypass_chroma, _("disable chroma correction"));
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->btn_bypass_chroma),
      ((const dt_iop_coloruniformityv2_params_t *)self->params)->bypass_chroma);
    g_signal_connect(G_OBJECT(g->btn_bypass_chroma), "toggled", G_CALLBACK(_bypass_chroma_toggled), self);
    gtk_box_pack_start(GTK_BOX(hb), g->btn_bypass_chroma, TRUE, TRUE, 0);

    g->btn_bypass_luma = gtk_toggle_button_new_with_label(_("luma"));
    gtk_widget_set_tooltip_text(g->btn_bypass_luma, _("disable lightness correction"));
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->btn_bypass_luma),
      ((const dt_iop_coloruniformityv2_params_t *)self->params)->bypass_luma);
    g_signal_connect(G_OBJECT(g->btn_bypass_luma), "toggled", G_CALLBACK(_bypass_luma_toggled), self);
    gtk_box_pack_start(GTK_BOX(hb), g->btn_bypass_luma, TRUE, TRUE, 0);

    gtk_box_pack_start(GTK_BOX(page_cor), hb, FALSE, FALSE, 2);
  }

  // --- HUE section ---------------------------------------------
  dt_gui_box_add(page_cor, dt_ui_section_label_new(_("Hue")));

  g->strength_h = _create_manual_slider(self, _("strength"),
                                        -1.0f, 1.0f, 0.01f, 0.0f, 2, NULL, 1.0f);
  _paint_strength_slider(g->strength_h, 0.85f, 0.55f, 0.25f);
  dt_gui_box_add(page_cor, g->strength_h);

  // Hue advanced expander
  {
    GtkWidget *exp = gtk_expander_new(_("hue advanced"));
    GtkWidget *box = gtk_box_new(GTK_ORIENTATION_VERTICAL, DT_BAUHAUS_SPACE);
    gtk_container_add(GTK_CONTAINER(exp), box);

    g->affinity_h = _create_manual_slider(self, _("affinity"),
                                          -1.0f, 1.0f, 0.01f, 0.0f, 2, NULL, 1.0f);
    gtk_widget_set_tooltip_text(g->affinity_h, _("negative: donut (corrects flaws)\npositive: peak (smooths complexion)"));
    gtk_box_pack_start(GTK_BOX(box), g->affinity_h, FALSE, FALSE, 0);

    g->preserve_h = _create_manual_slider(self, _("neutral zone"),
                                          0.0f, 1.0f, 0.01f, 0.5f, 2, "%", 100.0f);
    gtk_widget_set_tooltip_text(g->preserve_h, _("neutral zone radius (donut) in negative affinity mode"));
    gtk_box_pack_start(GTK_BOX(box), g->preserve_h, FALSE, FALSE, 0);

    g->priority_h = _create_manual_slider(self, _("priority ↑↓"),
                                          -1.0f, 1.0f, 0.01f, 0.0f, 2, NULL, 1.0f);
    gtk_widget_set_tooltip_text(g->priority_h,
      _("positive: mostly corrects pixels shifted clockwise\n"
        "negative: mostly corrects those shifted counterclockwise\n"
        "0: symmetric correction\n"
        "slider colors show which side will be corrected"));
    _paint_hue_priority_slider(g->priority_h, 0.12f);
    gtk_box_pack_start(GTK_BOX(box), g->priority_h, FALSE, FALSE, 0);

    g->preserve_texture_h = _create_manual_slider(self, _("preserve texture"),
                                                  0.0f, 1.0f, 0.01f, 0.0f, 2, "%", 100.0f);
    gtk_widget_set_tooltip_text(g->preserve_texture_h,
      _("slows correction when pixel is already close to target\n"
        "protects micro-contrast and skin texture\n"
        "0% = linear correction / 100% = maximum braking"));
    gtk_box_pack_start(GTK_BOX(box), g->preserve_texture_h, FALSE, FALSE, 0);

    gtk_widget_show_all(exp);
    dt_gui_box_add(page_cor, exp);
  }

  // --- CHROMA section ---------------------------------------------
  dt_gui_box_add(page_cor, dt_ui_section_label_new(_("Chroma")));

  g->strength_c = _create_manual_slider(self, _("strength"),
                                        -1.0f, 1.0f, 0.01f, 0.0f, 2, NULL, 1.0f);
  _paint_strength_slider(g->strength_c, 0.75f, 0.35f, 0.55f);
  dt_gui_box_add(page_cor, g->strength_c);

  // Chroma advanced expander
  {
    GtkWidget *exp = gtk_expander_new(_("chroma advanced"));
    GtkWidget *box = gtk_box_new(GTK_ORIENTATION_VERTICAL, DT_BAUHAUS_SPACE);
    gtk_container_add(GTK_CONTAINER(exp), box);

    g->affinity_c = _create_manual_slider(self, _("affinity"),
                                          -1.0f, 1.0f, 0.01f, 0.0f, 2, NULL, 1.0f);
    gtk_widget_set_tooltip_text(g->affinity_c, _("negative: donut (corrects flaws)\npositive: peak (smooths complexion)"));
    gtk_box_pack_start(GTK_BOX(box), g->affinity_c, FALSE, FALSE, 0);

    g->preserve_c = _create_manual_slider(self, _("neutral zone"),
                                          0.0f, 1.0f, 0.01f, 0.5f, 2, "%", 100.0f);
    gtk_widget_set_tooltip_text(g->preserve_c, _("neutral zone radius (donut) in negative affinity mode"));
    gtk_box_pack_start(GTK_BOX(box), g->preserve_c, FALSE, FALSE, 0);

    g->priority_c = _create_manual_slider(self, _("priority ↑↓"),
                                          -1.0f, 1.0f, 0.01f, 0.0f, 2, NULL, 1.0f);
    gtk_widget_set_tooltip_text(g->priority_c,
      _("positive: mostly corrects pixels above target\n"
        "negative: mostly corrects those below\n"
        "0: symmetric correction"));
    gtk_box_pack_start(GTK_BOX(box), g->priority_c, FALSE, FALSE, 0);

    g->preserve_texture_c = _create_manual_slider(self, _("preserve texture"),
                                                  0.0f, 1.0f, 0.01f, 0.0f, 2, "%", 100.0f);
    gtk_widget_set_tooltip_text(g->preserve_texture_c,
      _("slows correction when pixel is already close to target\n"
        "protects micro-contrast and skin texture\n"
        "0% = linear correction / 100% = maximum braking"));
    gtk_box_pack_start(GTK_BOX(box), g->preserve_texture_c, FALSE, FALSE, 0);

    gtk_widget_show_all(exp);
    dt_gui_box_add(page_cor, exp);
  }

  // --- LIGHTNESS section -----------------------------------------
  dt_gui_box_add(page_cor, dt_ui_section_label_new(_("Lightness")));

  g->strength_l = _create_manual_slider(self, _("strength"),
                                        -1.0f, 1.0f, 0.01f, 0.0f, 2, NULL, 1.0f);
  _paint_strength_slider(g->strength_l, 0.80f, 0.80f, 0.60f);
  dt_gui_box_add(page_cor, g->strength_l);

  // Lightness advanced expander
  {
    GtkWidget *exp = gtk_expander_new(_("lightness advanced"));
    GtkWidget *box = gtk_box_new(GTK_ORIENTATION_VERTICAL, DT_BAUHAUS_SPACE);
    gtk_container_add(GTK_CONTAINER(exp), box);

    g->affinity_l = _create_manual_slider(self, _("affinity"),
                                          -1.0f, 1.0f, 0.01f, 0.0f, 2, NULL, 1.0f);
    gtk_widget_set_tooltip_text(g->affinity_l, _("negative: donut (corrects flaws)\npositive: peak (smooths complexion)"));
    gtk_box_pack_start(GTK_BOX(box), g->affinity_l, FALSE, FALSE, 0);

    g->preserve_l = _create_manual_slider(self, _("neutral zone"),
                                          0.0f, 1.0f, 0.01f, 0.5f, 2, "%", 100.0f);
    gtk_widget_set_tooltip_text(g->preserve_l, _("neutral zone radius (donut) in negative affinity mode"));
    gtk_box_pack_start(GTK_BOX(box), g->preserve_l, FALSE, FALSE, 0);

    g->priority_l = _create_manual_slider(self, _("priority ↑↓"),
                                          -1.0f, 1.0f, 0.01f, 0.0f, 2, NULL, 1.0f);
    gtk_widget_set_tooltip_text(g->priority_l,
      _("positive: mostly corrects pixels brighter than target\n"
        "negative: mostly corrects darker ones\n"
        "0: symmetric correction"));
    gtk_box_pack_start(GTK_BOX(box), g->priority_l, FALSE, FALSE, 0);

    g->preserve_texture_l = _create_manual_slider(self, _("preserve texture"),
                                                  0.0f, 1.0f, 0.01f, 0.0f, 2, "%", 100.0f);
    gtk_widget_set_tooltip_text(g->preserve_texture_l,
      _("slows correction when pixel is already close to target\n"
        "protects micro-contrast and skin texture\n"
        "0% = linear correction / 100% = maximum braking"));
    gtk_box_pack_start(GTK_BOX(box), g->preserve_texture_l, FALSE, FALSE, 0);

    gtk_widget_show_all(exp);
    dt_gui_box_add(page_cor, exp);
  }

  GtkWidget *auto_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, DT_BAUHAUS_SPACE);

  GtkWidget *btn_auto_h = gtk_button_new_with_label(_("hue shift"));
  g_signal_connect(G_OBJECT(btn_auto_h), "clicked", G_CALLBACK(_shift_hue_cb), self);
  gtk_widget_set_tooltip_text(btn_auto_h, _("Full shift towards target hue, applied in adjustments tab"));
  gtk_box_pack_start(GTK_BOX(auto_box), btn_auto_h, TRUE, TRUE, 0);
  gtk_widget_show_all(auto_box);
  dt_gui_box_add(page_cor, auto_box);
  
  // =========================================================================
  // TAB 3: ADJUSTMENTS
  // =========================================================================
  GtkWidget *page_adj = dt_ui_notebook_page(g->notebook,
                                            N_("adjustments"),
                                            _("global offsets and application mode"));

  g->offset_h = _create_manual_slider(self, _("hue offset"),
                                      -0.5f, 0.5f, 0.001f, 0.0f, 2, "°", 360.0f);
  _paint_hue_slider(g->offset_h);
  dt_gui_box_add(page_adj, g->offset_h);

  g->offset_c = _create_manual_slider(self, _("chroma offset"),
                                      -1.0f, 1.0f, 0.01f, 0.0f, 2, "%", 100.0f);
  dt_gui_box_add(page_adj, g->offset_c);

  g->offset_l = _create_manual_slider(self, _("luma offset"),
                                      -1.0f, 1.0f, 0.01f, 0.0f, 2, "%", 100.0f);
  _paint_luma_slider(g->offset_l);
  dt_gui_box_add(page_adj, g->offset_l);

  GtkWidget *offsets_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, DT_BAUHAUS_SPACE);
  gtk_box_pack_start(GTK_BOX(offsets_box), gtk_label_new(_("Apply mode:")), FALSE, FALSE, 0);

  g->offsets_weighted_combo = gtk_combo_box_text_new();
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(g->offsets_weighted_combo), _("uniform"));
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(g->offsets_weighted_combo), _("weighted by affinity"));
  g_signal_connect(G_OBJECT(g->offsets_weighted_combo), "changed",
                   G_CALLBACK(_offsets_weighted_changed_cb), self);
  gtk_box_pack_start(GTK_BOX(offsets_box), g->offsets_weighted_combo, TRUE, TRUE, 0);
  gtk_widget_show_all(offsets_box);
  dt_gui_box_add(page_adj, offsets_box);

}

void gui_focus(dt_iop_module_t *self, gboolean in)
{
  if(!in) dt_iop_color_picker_reset(self, TRUE);
}

void gui_cleanup(dt_iop_module_t *self) { dt_iop_default_cleanup(self);}
