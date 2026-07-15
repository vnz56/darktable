/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/

// ENGINE PROTOTYPE for a scene-referred creative grading tool ("attractor field
// in dt UCS"). One attractor, driven by sliders, to validate the RENDER (smooth,
// continuous, micro-variation preserved) before building the vectorscope UI.
//
// Pipeline per pixel:
//   RGB (working) -> dt UCS JCH   [perceptual, scene-referred]
//   selection weight = hue-angular Gaussian (around a selected hue) * saturation gate
//   (protects neutrals) * strength; the selected colours get a hue rotation +
//   chroma/lightness change, weighted by that continuous kernel -> no banding, no
//   seams, keeps micro-variation.
//   -> back to RGB.

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "bauhaus/bauhaus.h"
#include "common/chromatic_adaptation.h"
#include "common/colorspaces_inline_conversions.h"
#include "common/eigf.h"
#include "common/guided_filter.h"
#include "common/imagebuf.h"
#include "common/iop_profile.h"
#include "common/math.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/imageop_gui.h"
#include "dtgtk/drawingarea.h"
#include "gui/accelerators.h"
#include "gui/color_picker_proxy.h"
#include "gui/gtk.h"
#include "iop/iop_api.h"

#include <gtk/gtk.h>

DT_MODULE_INTROSPECTION(1, dt_iop_colorwarp_params_t)

#define CW_MAX_NODES 8

// one attractor node: everything that defines a selection + its directed move.
// (mirrors the flat "scratch" fields below, which are the live editor for the active node)
typedef struct dt_iop_colorwarp_node_t
{
  float strength;         // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0
  float center_hue;       // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.5
  float reach;            // $MIN: 0.05 $MAX: 1.0 $DEFAULT: 0.125
  float select_sat;       // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.5
  float sat_range;        // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 1.0
  float select_light;     // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.5
  float light_range;      // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 1.0
  float feather;          // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.5
  float neutral_protect;  // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0
  int invert;             // $DEFAULT: 0
  float shift_hue;        // $MIN: -0.5 $MAX: 0.5 $DEFAULT: 0.0
  float shift_chroma;     // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0
  float shift_lightness;  // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0
  float convergence;      // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0
  float affinity;         // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0
  float neutral_zone;     // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0
  float priority;         // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0
  int per_component;      // $DEFAULT: 0
  float conv_h;           // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0
  float aff_h;            // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0
  float nz_h;             // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0
  float prio_h;           // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0
  float conv_c;           // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0
  float aff_c;            // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0
  float nz_c;             // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0
  float prio_c;           // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0
  float conv_l;           // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0
  float aff_l;            // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0
  float nz_l;             // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0
  float prio_l;           // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0
  int absolute_target;    // $DEFAULT: 0
} dt_iop_colorwarp_node_t;

typedef struct dt_iop_colorwarp_params_t
{
  float strength;         // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "strength"
  // --- selection: one uniform "centre + range" band per axis ---
  float center_hue;       // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.5 $DESCRIPTION: "select hue"
  float reach;            // $MIN: 0.05 $MAX: 1.0 $DEFAULT: 0.125 $DESCRIPTION: "hue range"
  float select_sat;       // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.5 $DESCRIPTION: "select saturation"
  float sat_range;        // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 1.0 $DESCRIPTION: "saturation range"
  float select_light;     // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.5 $DESCRIPTION: "select lightness"
  float light_range;      // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 1.0 $DESCRIPTION: "lightness range"
  float feather;          // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.5 $DESCRIPTION: "feather"
  float neutral_protect;  // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "neutral protection"
  gboolean invert;        // $DEFAULT: FALSE $DESCRIPTION: "invert selection"
  // --- the H/S/L move applied to the selection ---
  float shift_hue;        // $MIN: -0.5 $MAX: 0.5 $DEFAULT: 0.0 $DESCRIPTION: "shift hue"
  float shift_chroma;     // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "shift chroma"
  float shift_lightness;  // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "shift lightness"
  // --- affinity, global ---
  float convergence;      // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "convergence"
  float affinity;         // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "affinity"
  float neutral_zone;     // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "neutral zone"
  float priority;         // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "priority"
  // --- affinity, per component (used when per_component is on) ---
  gboolean per_component; // $DEFAULT: FALSE $DESCRIPTION: "per-component affinity"
  float conv_h;           // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "convergence"
  float aff_h;            // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "affinity"
  float nz_h;             // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "neutral zone"
  float prio_h;           // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "priority"
  float conv_c;           // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "convergence"
  float aff_c;            // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "affinity"
  float nz_c;             // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "neutral zone"
  float prio_c;           // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "priority"
  float conv_l;           // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "convergence"
  float aff_l;            // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "affinity"
  float nz_l;             // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "neutral zone"
  float prio_l;           // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "priority"
  gboolean absolute_target; // $DEFAULT: FALSE $DESCRIPTION: "absolute target"
  float smoothing;        // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.25 $DESCRIPTION: "smoothing"
  float edge;             // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.3 $DESCRIPTION: "edge threshold"
  gboolean use_eigf;      // $DEFAULT: FALSE $DESCRIPTION: "exposure-independent filter"
  float corr_smooth;      // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "correction smoothing"
  float input_smooth;     // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "denoise selection input"
  gboolean polar_move;    // $DEFAULT: TRUE $DESCRIPTION: "hue rotation (polar)"
  // --- multi-node (field): the flat fields above are the live editor for node[active_node] ---
  int num_nodes;          // $MIN: 1 $MAX: 8 $DEFAULT: 1 $DESCRIPTION: "nodes"
  int active_node;        // $MIN: 0 $MAX: 7 $DEFAULT: 0
  dt_iop_colorwarp_node_t node[CW_MAX_NODES];
} dt_iop_colorwarp_params_t;

typedef struct dt_iop_colorwarp_gui_data_t
{
  GtkWidget *strength, *center_hue, *reach, *select_sat, *sat_range, *select_light, *light_range, *invert;
  GtkWidget *feather, *neutral_protect;
  GtkWidget *shift_hue, *shift_chroma, *shift_lightness, *convergence, *affinity;
  GtkWidget *neutral_zone, *priority, *absolute_target, *smoothing, *edge, *use_eigf, *corr_smooth, *input_smooth;
  GtkWidget *polar_move;
  GtkNotebook *aff_notebook;   // affinity: global + per-component pages
  GtkWidget *conv_h, *aff_h, *nz_h, *prio_h;
  GtkWidget *conv_c, *aff_c, *nz_c, *prio_c;
  GtkWidget *conv_l, *aff_l, *nz_l, *prio_l;
  GtkDrawingArea *area;   // selection canvas
  GtkWidget *mode_combo;  // which 2 axes the canvas shows
  GtkWidget *fine_toggle; // reveal the axis sliders already on the canvas
  GtkWidget *mask_combo;  // show mask: off / correction / selection
  GtkWidget *node_combo, *node_add, *node_remove;  // node selector
  int mode;               // 0=hue/sat (wheel), 1=hue/light (wheel), 2=sat/light (panel)
  int mask_mode;          // 0=off, 1=correction intensity, 2=selection (darkroom preview only)
  int view_all;           // "all nodes" preview: mask spans every node (union), not just active
  int drag;               // 0=none, 1=source, 2=target
} dt_iop_colorwarp_gui_data_t;

typedef struct dt_iop_colorwarp_nodedata_t
{
  float strength;
  float hc;              // selected hue centre (radians)
  float hw_h, sigma_h;   // hue plateau half-width + shoulder sigma (radians)
  float sat_center;      // selected saturation centre (dt UCS S)
  float hw_s, sat_sigma; // saturation plateau half-width + shoulder sigma
  float light_center;    // selected lightness centre (dt UCS J)
  float hw_l, light_sigma; // lightness plateau half-width + shoulder sigma
  float guard2;          // neutral-protection guard, squared (dt UCS S)
  int invert;            // invert the whole selection
  float sel_hue;         // selection centre hue (turns [0,1])
  float sel_sat;         // selection centre saturation (canvas [0,1])
  float shift_hue;       // in turns
  float shift_chroma;    // canvas-saturation offset
  float shift_lightness; // dt UCS J offset
  float convergence;     // -1 diverge .. 0 translate .. 1 converge (global)
  float affinity;        // boost (>0) / core-preserving donut depth (<0) (global)
  float neutral_zone;    // preserved-core radius [0,1] (with negative affinity) (global)
  float priority;        // per-axis asymmetry [-1,1] (global)
  int per_component;     // use the per-axis affinity sets instead of the global one
  float conv_h, aff_h, nz_h, prio_h;   // per-component affinity: hue
  float conv_c, aff_c, nz_c, prio_c;   //                         saturation
  float conv_l, aff_l, nz_l, prio_l;   //                         lightness
} dt_iop_colorwarp_nodedata_t;

typedef struct dt_iop_colorwarp_data_t
{
  int num_nodes;
  int active_node;                                // which node the mask preview isolates
  dt_iop_colorwarp_nodedata_t nd[CW_MAX_NODES];   // resolved, process-ready nodes
  float smoothing;       // spatial mask smoothing amount [0,1] (global)
  float edge_eps;        // guided-filter sqrt_eps (edge sensitivity) (global)
  int use_eigf;          // mask filter: 0=guided (image), 1=EIGF (exposure-independent) (global)
  float eigf_feather;    // EIGF feathering, derived from the edge slider (global)
  float corr_smooth;     // spatial smoothing of the accumulated move [0,1] (global)
  float input_smooth;    // pre-smoothing of the selection inputs (J + chromaticity) [0,1] (global)
  int polar_move;        // move mode: 1 = polar (hue rotation), 0 = Cartesian a/b slide (global)
} dt_iop_colorwarp_data_t;


const char *name()
{
  return _("color warp");
}

const char *aliases()
{
  return _("grading|harmony|hue");
}

int flags()
{
  // no ALLOW_TILING: the spatial mask blur needs the whole ROI (tile seams otherwise)
  return IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_SUPPORTS_BLENDING;
}

int default_group()
{
  return IOP_GROUP_COLOR | IOP_GROUP_GRADING;
}

dt_iop_colorspace_type_t default_colorspace(dt_iop_module_t *self,
                                            dt_dev_pixelpipe_t *pipe,
                                            dt_dev_pixelpipe_iop_t *piece)
{
  return IOP_CS_RGB;
}

const char **description(dt_iop_module_t *self)
{
  return dt_iop_set_description(self, _("creative color grading via a smooth attractor\n"
                                        "field in the perceptual dt UCS space (prototype)"),
                                      _("creative"),
                                      _("linear, RGB, scene-referred"),
                                      _("perceptual, dt UCS"),
                                      _("linear, RGB, scene-referred"));
}

// neutral-zone slider [0,1] -> preserved-core radius (canvas-distance units), progressive
// (fine control at the low end), spanning the 3D colour scope. Same idea as coloruniformity.
#define CW_RMAX 1.5f
// one selection axis: flat plateau of half-width hw, then a Gaussian shoulder.
static inline float _cw_band_w(const float ad, const float hw, const float inv_2sig2)
{
  const float e = fmaxf(ad - hw, 0.0f);
  return expf(-(e * e) * inv_2sig2);
}

// invert the dt UCS saturation metric S = C/(J*(C^1.336+1)) for C, given S and J.
// fixed-point C = S*J*(C^1.336+1) converges in a few steps; used per node (target chroma).
static inline float _cw_S_to_C(const float S, const float J)
{
  if(S <= 0.0f || J <= 0.0f) return 0.0f;
  float C = S * J;
  for(int i = 0; i < 6; i++) C = S * J * (powf(fmaxf(C, 0.0f), 1.33654221029386f) + 1.0f);
  return C;
}

static inline float _cw_neutral_radius(const float neutral)
{
  const float n = CLAMP(neutral, 0.0f, 1.0f);
  return CW_RMAX * (expf(3.0f * n) - 1.0f) / (expf(3.0f) - 1.0f);
}

// affinity weight (coloruniformity model): >=0 boost the correction; <0 preserve a core
// of radius r around the target, by depth |affinity|.
static inline float _cw_affinity_weight(const float affinity, const float neutral, const float dist)
{
  if(affinity >= 0.0f) return 1.0f + affinity;
  const float r = _cw_neutral_radius(neutral);
  float hole;
  if(dist <= r) hole = 1.0f;
  else
  {
    const float edge = fmaxf(r * 0.5f, 1e-4f);
    const float u = CLAMP((dist - r) / edge, 0.0f, 1.0f);
    hole = 1.0f - u * u * (3.0f - 2.0f * u);   // smoothstep down
  }
  return 1.0f - fabsf(affinity) * hole;
}

// one axis' move, using that axis' distance for the affinity weight (per-component path).
// dtc = target - centre (translate delta, pre-wrapped for hue); dt = target - value.
static inline float _cw_axis_move(const float dtc, const float dt, const float w_base,
                                  const float conv, const float aff, const float nz, const float prio)
{
  const float w = w_base * _cw_affinity_weight(aff, nz, fabsf(dt));
  const float wt = w * (1.0f - conv);
  const float wc = (conv > 0.0f) ? fminf(w * conv, 1.0f) : w * conv;
  const float p = CLAMP(1.0f - prio * tanhf(dt * 4.0f), 0.0f, 2.0f);
  return p * (wt * dtc + wc * dt);
}

void process(dt_iop_module_t *self,
             dt_dev_pixelpipe_iop_t *piece,
             const void *const ivoid,
             void *const ovoid,
             const dt_iop_roi_t *const roi_in,
             const dt_iop_roi_t *const roi_out)
{
  if(!dt_iop_have_required_input_format(4, self, piece->colors, ivoid, ovoid, roi_in, roi_out))
    return;

  const dt_iop_colorwarp_data_t *const d = piece->data;
  const dt_iop_order_iccprofile_info_t *const work_profile =
    dt_ioppr_get_pipe_work_profile_info(piece->pipe);
  const int W = roi_out->width, H = roi_out->height;
  const size_t npixels = (size_t)W * H;

  const float L_white = Y_to_dt_UCS_L_star(1.0f);   // scene white = 1.0 (prototype)

  // mask display is a darkroom-preview-only GUI state (never thumbnails/export)
  const dt_iop_colorwarp_gui_data_t *const gd = self->gui_data;
  const int mask_mode = (gd && dt_pipe_is_full(piece->pipe)) ? gd->mask_mode : 0;
  const int mask_all = (gd && gd->view_all) ? 1 : 0;   // "all nodes": mask spans every node

  gboolean any = FALSE;
  for(int n = 0; n < d->num_nodes; n++) if(d->nd[n].strength > 0.f) any = TRUE;
  if(!work_profile || (!any && !mask_mode))
  {
    dt_iop_image_copy_by_size(ovoid, ivoid, W, H, 4);
    return;
  }

  // RAM-safe field: one reusable mask buffer + move accumulators (never N masks at once)
  float *const restrict mask = dt_alloc_align_float(npixels);
  float *const restrict mask_f = dt_alloc_align_float(npixels);
  float *const restrict acc_h = dt_alloc_align_float(npixels);
  float *const restrict acc_s = dt_alloc_align_float(npixels);
  float *const restrict acc_l = dt_alloc_align_float(npixels);
  float *const restrict sel_acc = (mask_mode == 2) ? dt_alloc_align_float(npixels) : NULL;
  if(!mask || !mask_f || !acc_h || !acc_s || !acc_l || (mask_mode == 2 && !sel_acc))
  {
    dt_free_align(mask); dt_free_align(mask_f); dt_free_align(acc_h);
    dt_free_align(acc_s); dt_free_align(acc_l); dt_free_align(sel_acc);
    dt_iop_image_copy_by_size(ovoid, ivoid, W, H, 4);
    return;
  }

  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
  {
    acc_h[k] = acc_s[k] = acc_l[k] = 0.0f;
    if(sel_acc) sel_acc[k] = 0.0f;
  }

  const int gw = (int)(d->smoothing * fmaxf(1.5f, 8.0f * roi_in->scale / piece->iscale) + 0.5f);

  // optional: pre-smooth the selection inputs (J + chromaticity) so the mask is built from
  // denoised data -> clean selections in noisy low-chroma shadows. shared by all nodes (also
  // spares the per-node JCH recompute). opt-in: allocates 3 buffers (RAM), 0 = off.
  const float *restrict PJ = NULL, *restrict PH = NULL, *restrict PS = NULL;
  float *restrict bJ = NULL, *restrict bU = NULL, *restrict bV = NULL;
  const int pw = (int)(d->input_smooth * fmaxf(1.5f, 8.0f * roi_in->scale / piece->iscale) + 0.5f);
  if(d->input_smooth > 0.f && pw >= 1)
  {
    bJ = dt_alloc_align_float(npixels);
    bU = dt_alloc_align_float(npixels);
    bV = dt_alloc_align_float(npixels);
    if(bJ && bU && bV)
    {
      DT_OMP_FOR()
      for(size_t k = 0; k < npixels; k++)
      {
        const float *const restrict in = (const float *)ivoid + 4 * k;
        const dt_aligned_pixel_t px_rgb = { in[0], in[1], in[2], 0.f };
        dt_aligned_pixel_t JCH;
        dt_ioppr_rgb_matrix_to_dt_UCS_JCH(px_rgb, JCH, work_profile->matrix_in_transposed, L_white);
        bJ[k] = JCH[0];                          // lightness J
        bU[k] = JCH[1] * cosf(JCH[2]);           // chromaticity vector (blur-safe for hue,
        bV[k] = JCH[1] * sinf(JCH[2]);           //  averages out low-chroma hue noise)
      }
      float *const restrict bufs[3] = { bJ, bU, bV };
      for(int c = 0; c < 3; c++)
      {
        guided_filter(ivoid, bufs[c], mask_f, W, H, 4, pw, d->edge_eps, 1.0f, 0.0f, 1.0f);
        DT_OMP_FOR()
        for(size_t k = 0; k < npixels; k++) bufs[c][k] = mask_f[k];
      }
      // derive smoothed hue + saturation in place: bU <- hue, bV <- saturation
      DT_OMP_FOR()
      for(size_t k = 0; k < npixels; k++)
      {
        const float J = bJ[k];
        const float C = hypotf(bU[k], bV[k]);
        bU[k] = atan2f(bV[k], bU[k]);
        bV[k] = (J > 1e-6f) ? C / (J * (powf(C, 1.33654221029386f) + 1.0f)) : 0.0f;
      }
      PJ = bJ; PH = bU; PS = bV;
    }
  }

  // --- per node: build its mask, smooth it, accumulate its move (superposition field) ---
  for(int n = 0; n < d->num_nodes; n++)
  {
    const dt_iop_colorwarp_nodedata_t *const nd = &d->nd[n];
    if(mask_mode != 0 && !mask_all && n != d->active_node) continue;   // isolate active node, unless "all nodes"
    if(nd->strength <= 0.f && mask_mode != 2) continue;   // inactive nodes add no move (selection still shown)

    const float hc = nd->hc, hw_h = nd->hw_h;
    const float inv_2sh2 = 1.0f / (2.0f * nd->sigma_h * nd->sigma_h);
    const float sat_center = nd->sat_center, hw_s = nd->hw_s;
    const float inv_2ss2 = 1.0f / (2.0f * nd->sat_sigma * nd->sat_sigma);
    const float light_center = nd->light_center, hw_l = nd->hw_l;
    const float inv_2sl2 = 1.0f / (2.0f * nd->light_sigma * nd->light_sigma);
    const float guard2 = nd->guard2;
    const int invert = nd->invert;

    DT_OMP_FOR()
    for(size_t k = 0; k < npixels; k++)
    {
      float J, sat, h;
      if(PJ) { J = PJ[k]; h = PH[k]; sat = PS[k]; }   // pre-smoothed selection inputs
      else
      {
        const float *const restrict in = (const float *)ivoid + 4 * k;
        const dt_aligned_pixel_t px_rgb = { in[0], in[1], in[2], 0.f };
        dt_aligned_pixel_t JCH;
        dt_ioppr_rgb_matrix_to_dt_UCS_JCH(px_rgb, JCH, work_profile->matrix_in_transposed, L_white);
        J = JCH[0]; h = JCH[2];
        sat = (J > 1e-6f) ? JCH[1] / (J * (powf(JCH[1], 1.33654221029386f) + 1.0f)) : 0.0f;
      }
      const float dh = fabsf(atan2f(sinf(h - hc), cosf(h - hc)));
      const float hue_w = _cw_band_w(dh, hw_h, inv_2sh2);
      const float sat_w = _cw_band_w(fabsf(sat - sat_center), hw_s, inv_2ss2);
      const float light_w = _cw_band_w(fabsf(J - light_center), hw_l, inv_2sl2);
      float sel = hue_w * sat_w * light_w;
      if(invert) sel = 1.0f - sel;
      // neutral protection: fade near-neutral pixels (noisy hue); guard2==0 -> off
      mask[k] = (guard2 > 0.0f) ? sel * (sat * sat) / (sat * sat + guard2) : sel;
    }

    const float *restrict selbuf = mask;
    if(d->smoothing > 0.f && gw >= 1)
    {
      float *restrict fb;
      if(d->use_eigf)
      {
        // exposure-independent, self-guided (in-place on mask)
        fast_eigf_surface_blur(mask, W, H, (float)gw, d->eigf_feather, 1,
                               DT_GF_BLENDING_LINEAR, 1.0f, 0.0f, exp2f(-14.0f), 4.0f);
        fb = mask;
      }
      else
      {
        guided_filter(ivoid, mask, mask_f, W, H, 4, gw, d->edge_eps, 1.0f, 0.0f, 1.0f);
        fb = mask_f;
      }
      // both filters can overshoot; a mask must stay in [0,1] (negatives -> black speckle)
      DT_OMP_FOR()
      for(size_t k = 0; k < npixels; k++) fb[k] = CLAMP(fb[k], 0.0f, 1.0f);
      selbuf = fb;
    }

    const float conv = nd->convergence, affinity = nd->affinity, neutral = nd->neutral_zone;
    const float strength = nd->strength;

    if(d->polar_move)
    {
      // POLAR: move in (hue turns, saturation, lightness); chroma rebuilt by scaling.
      // acc_h/acc_s/acc_l hold the hue / saturation / lightness deltas.
      const float prio = nd->priority;
      const float c_hue = nd->sel_hue,      t_hue = nd->sel_hue + nd->shift_hue;
      const float c_sat = nd->sel_sat,      t_sat = nd->sel_sat + nd->shift_chroma;
      const float c_lgt = nd->light_center, t_lgt = nd->light_center + nd->shift_lightness;

      DT_OMP_FOR()
      for(size_t k = 0; k < npixels; k++)
      {
        if(sel_acc) sel_acc[k] = fmaxf(sel_acc[k], selbuf[k]);
        if(strength <= 0.f) continue;

        const float *const restrict in = (const float *)ivoid + 4 * k;
        const dt_aligned_pixel_t px_rgb = { in[0], in[1], in[2], 0.f };
        dt_aligned_pixel_t JCH;
        dt_ioppr_rgb_matrix_to_dt_UCS_JCH(px_rgb, JCH, work_profile->matrix_in_transposed, L_white);
        const float J = JCH[0], C = JCH[1], h = JCH[2];
        const float hue_p = (h + M_PI_F) / (2.0f * M_PI_F);
        const float S_in = (J > 1e-6f) ? C / (J * (powf(C, 1.33654221029386f) + 1.0f)) : 0.0f;
        const float sat_p = sqrtf(fmaxf(S_in, 0.0f) / 0.1f);
        const float lgt_p = J;

        float dth = t_hue - hue_p; dth -= roundf(dth);
        const float dts = t_sat - sat_p, dtl = t_lgt - lgt_p;
        float dct_h = t_hue - c_hue; dct_h -= roundf(dct_h);
        const float dct_s = t_sat - c_sat, dct_l = t_lgt - c_lgt;
        const float w_base = strength * selbuf[k];

        if(nd->per_component)
        {
          acc_h[k] += _cw_axis_move(dct_h, dth, w_base, nd->conv_h, nd->aff_h, nd->nz_h, nd->prio_h);
          acc_s[k] += _cw_axis_move(dct_s, dts, w_base, nd->conv_c, nd->aff_c, nd->nz_c, nd->prio_c);
          acc_l[k] += _cw_axis_move(dct_l, dtl, w_base, nd->conv_l, nd->aff_l, nd->nz_l, nd->prio_l);
        }
        else
        {
          const float dist = sqrtf(dth * dth + dts * dts + dtl * dtl);
          const float w = w_base * _cw_affinity_weight(affinity, neutral, dist);
          const float wt = w * (1.0f - conv);
          const float wc = (conv > 0.0f) ? fminf(w * conv, 1.0f) : w * conv;
          const float ph = CLAMP(1.0f - prio * tanhf(dth * 4.0f), 0.0f, 2.0f);
          const float ps = CLAMP(1.0f - prio * tanhf(dts * 4.0f), 0.0f, 2.0f);
          const float pl = CLAMP(1.0f - prio * tanhf(dtl * 4.0f), 0.0f, 2.0f);
          acc_h[k] += ph * (wt * dct_h + wc * dth);
          acc_s[k] += ps * (wt * dct_s + wc * dts);
          acc_l[k] += pl * (wt * dct_l + wc * dtl);
        }
      }
    }
    else
    {
      // CARTESIAN a/b: absolute chroma plane (a = C*cos h, b = C*sin h) + lightness.
      // acc_h/acc_s hold the (a, b) delta, acc_l the lightness delta.
      const float th_ang = (nd->sel_hue + nd->shift_hue) * (2.0f * M_PI_F) - M_PI_F;
      const float ch_ang = nd->sel_hue * (2.0f * M_PI_F) - M_PI_F;
      const float t_lgt = nd->light_center + nd->shift_lightness;
      const float c_lgt = nd->light_center;
      const float ts = CLAMP(nd->sel_sat + nd->shift_chroma, 0.0f, 2.0f);
      const float cs = CLAMP(nd->sel_sat, 0.0f, 2.0f);
      const float Ct = _cw_S_to_C(ts * ts * 0.1f, fmaxf(t_lgt, 1e-4f));
      const float Cc = _cw_S_to_C(cs * cs * 0.1f, fmaxf(c_lgt, 1e-4f));
      const float a_t = Ct * cosf(th_ang), b_t = Ct * sinf(th_ang);
      const float a_c = Cc * cosf(ch_ang), b_c = Cc * sinf(ch_ang);

      DT_OMP_FOR()
      for(size_t k = 0; k < npixels; k++)
      {
        if(sel_acc) sel_acc[k] = fmaxf(sel_acc[k], selbuf[k]);
        if(strength <= 0.f) continue;

        const float *const restrict in = (const float *)ivoid + 4 * k;
        const dt_aligned_pixel_t px_rgb = { in[0], in[1], in[2], 0.f };
        dt_aligned_pixel_t JCH;
        dt_ioppr_rgb_matrix_to_dt_UCS_JCH(px_rgb, JCH, work_profile->matrix_in_transposed, L_white);
        const float J = JCH[0], C = JCH[1], h = JCH[2];
        const float a_p = C * cosf(h), b_p = C * sinf(h);
        const float w_base = strength * selbuf[k];

        if(nd->per_component)
        {
          const float da = a_t - a_p, db = b_t - b_p;
          const float distc = sqrtf(da * da + db * db);            // chroma plane uses the "saturation" set
          const float w = w_base * _cw_affinity_weight(nd->aff_c, nd->nz_c, distc);
          const float wt = w * (1.0f - nd->conv_c);
          const float wc = (nd->conv_c > 0.0f) ? fminf(w * nd->conv_c, 1.0f) : w * nd->conv_c;
          acc_h[k] += wt * (a_t - a_c) + wc * da;
          acc_s[k] += wt * (b_t - b_c) + wc * db;
          acc_l[k] += _cw_axis_move(t_lgt - c_lgt, t_lgt - J, w_base, nd->conv_l, nd->aff_l, nd->nz_l, nd->prio_l);
        }
        else
        {
          // convergence acts on the CHROMA plane only; lightness just translates by its shift.
          const float da = a_t - a_p, db = b_t - b_p;
          const float dist = sqrtf(da * da + db * db);
          const float w = w_base * _cw_affinity_weight(affinity, neutral, dist);
          const float wt = w * (1.0f - conv);
          const float wc = (conv > 0.0f) ? fminf(w * conv, 1.0f) : w * conv;
          acc_h[k] += wt * (a_t - a_c) + wc * da;
          acc_s[k] += wt * (b_t - b_c) + wc * db;
          acc_l[k] += w * (t_lgt - c_lgt);
        }
      }
    }
  }

  if(mask_mode) piece->pipe->mask_display = DT_DEV_PIXELPIPE_DISPLAY_PASSTHRU;

  // optional: edge-aware smoothing of the accumulated move, to cut the noise that the
  // per-pixel converge / priority / affinity terms pick up in low-chroma shadows.
  // one pass per axis on the summed field (independent of node count); mask/mask_f are free.
  const int cw = (int)(d->corr_smooth * fmaxf(1.5f, 8.0f * roi_in->scale / piece->iscale) + 0.5f);
  if(d->corr_smooth > 0.f && cw >= 1 && mask_mode == 0)
  {
    float *const restrict accs[3] = { acc_h, acc_s, acc_l };
    for(int c = 0; c < 3; c++)
    {
      guided_filter(ivoid, accs[c], mask_f, W, H, 4, cw, d->edge_eps, 1.0f, 0.0f, 1.0f);
      DT_OMP_FOR()
      for(size_t k = 0; k < npixels; k++) accs[c][k] = mask_f[k];
    }
  }

  // --- apply the accumulated move (or write the grayscale mask) ---
  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
  {
    const float *const restrict in = (const float *)ivoid + 4 * k;
    float *const restrict out = (float *)ovoid + 4 * k;

    if(mask_mode == 2)
    {
      const float grey = CLAMP(sel_acc[k], 0.0f, 1.0f);
      out[0] = out[1] = out[2] = grey; out[3] = in[3];
      continue;
    }

    const dt_aligned_pixel_t px_rgb = { in[0], in[1], in[2], 0.f };
    dt_aligned_pixel_t JCH;
    dt_ioppr_rgb_matrix_to_dt_UCS_JCH(px_rgb, JCH, work_profile->matrix_in_transposed, L_white);
    const float J = JCH[0], C = JCH[1], h = JCH[2];

    float J2, C2, h2;
    if(d->polar_move)
    {
      // polar: rebuild from shifted (hue, saturation), chroma scaled by the saturation ratio
      const float hue_p = (h + M_PI_F) / (2.0f * M_PI_F);
      const float S_in = (J > 1e-6f) ? C / (J * (powf(C, 1.33654221029386f) + 1.0f)) : 0.0f;
      const float sat_p = sqrtf(fmaxf(S_in, 0.0f) / 0.1f);
      const float hue_o = hue_p + acc_h[k];
      const float sat_o = fmaxf(sat_p + acc_s[k], 0.0f);
      J2 = fmaxf(J + acc_l[k], 0.0f);
      if(mask_mode == 1)
      {
        float dhue = hue_o - hue_p; dhue -= roundf(dhue);
        const float mh = 2.0f * dhue * sat_p;
        const float mag = sqrtf(mh * mh + (sat_o - sat_p) * (sat_o - sat_p) + (J2 - J) * (J2 - J));
        const float grey = CLAMP(mag * 1.5f, 0.0f, 1.0f);
        out[0] = out[1] = out[2] = grey; out[3] = in[3];
        continue;
      }
      const float S_out = sat_o * sat_o * 0.1f;
      C2 = fmaxf(C * (S_out / fmaxf(S_in, 1e-6f)), 0.0f);
      h2 = hue_o * (2.0f * M_PI_F) - M_PI_F;
    }
    else
    {
      // Cartesian: apply the (a, b) chroma delta; chroma & hue rebuilt directly, no S ratio
      const float a_o = C * cosf(h) + acc_h[k];
      const float b_o = C * sinf(h) + acc_s[k];
      J2 = fmaxf(J + acc_l[k], 0.0f);
      if(mask_mode == 1)
      {
        const float mag = sqrtf(acc_h[k] * acc_h[k] + acc_s[k] * acc_s[k] + acc_l[k] * acc_l[k]);
        const float grey = CLAMP(mag * 4.0f, 0.0f, 1.0f);
        out[0] = out[1] = out[2] = grey; out[3] = in[3];
        continue;
      }
      C2 = hypotf(a_o, b_o);
      h2 = atan2f(b_o, a_o);
    }
    dt_aligned_pixel_t JCH2 = { J2, C2, h2, 0.f }, xyY, xyz65, xyz50, rgb_out;
    dt_UCS_JCH_to_xyY(JCH2, L_white, xyY);
    dt_xyY_to_XYZ(xyY, xyz65);
    XYZ_D65_to_D50(xyz65, xyz50);
    dt_apply_transposed_color_matrix(xyz50, work_profile->matrix_out_transposed, rgb_out);
    out[0] = rgb_out[0]; out[1] = rgb_out[1]; out[2] = rgb_out[2]; out[3] = in[3];
  }

  dt_free_align(mask); dt_free_align(mask_f);
  dt_free_align(acc_h); dt_free_align(acc_s); dt_free_align(acc_l); dt_free_align(sel_acc);
  dt_free_align(bJ); dt_free_align(bU); dt_free_align(bV);
}

// resolve a raw params node into process-ready values (band sigmas, centre coords, ...)
static void _cw_resolve_node(const dt_iop_colorwarp_node_t *n, dt_iop_colorwarp_nodedata_t *nd)
{
  nd->strength = n->strength;
  nd->hc = n->center_hue * (2.0f * M_PI_F) - M_PI_F;   // [0,1] -> [-pi,pi]
  // each axis: a flat plateau of half-width range*span (range=1 -> covers the whole axis,
  // weight 1 everywhere; range=0 -> nothing), plus a Gaussian shoulder whose width is a
  // fraction of the PLATEAU (feather*hw/2) so a tight range stays tight regardless of
  // feather. feather=0 -> hard edge. span = max axis distance.
  const float f = CLAMP(n->feather, 0.0f, 1.0f);
  nd->hw_h = n->reach * M_PI_F;            nd->sigma_h = fmaxf(f * nd->hw_h * 0.5f, 1e-3f);
  nd->sat_center = n->select_sat * n->select_sat * 0.1f;
  nd->hw_s = n->sat_range * 0.3f;          nd->sat_sigma = fmaxf(f * nd->hw_s * 0.5f, 1e-4f);
  nd->light_center = n->select_light;
  nd->hw_l = n->light_range * 1.5f;        nd->light_sigma = fmaxf(f * nd->hw_l * 0.5f, 0.02f);
  // neutral protection now fully user-controlled: 0 = off (can select even neutrals)
  const float guard = CLAMP(n->neutral_protect, 0.0f, 1.0f) * 0.06f;
  nd->guard2 = guard * guard;
  nd->invert = n->invert ? 1 : 0;
  nd->sel_hue = n->center_hue;
  nd->sel_sat = n->select_sat;
  nd->shift_hue = n->shift_hue;
  nd->shift_chroma = n->shift_chroma;
  nd->shift_lightness = n->shift_lightness;
  nd->convergence = n->convergence;
  nd->affinity = n->affinity;
  nd->neutral_zone = n->neutral_zone;
  nd->priority = n->priority;
  nd->per_component = n->per_component ? 1 : 0;
  nd->conv_h = n->conv_h; nd->aff_h = n->aff_h; nd->nz_h = n->nz_h; nd->prio_h = n->prio_h;
  nd->conv_c = n->conv_c; nd->aff_c = n->aff_c; nd->nz_c = n->nz_c; nd->prio_c = n->prio_c;
  nd->conv_l = n->conv_l; nd->aff_l = n->aff_l; nd->nz_l = n->nz_l; nd->prio_l = n->prio_l;
}

// copy the flat "scratch" fields (the live editor for the active node) into a node struct
static void _cw_scratch_node(const dt_iop_colorwarp_params_t *p, dt_iop_colorwarp_node_t *n)
{
  n->strength = p->strength; n->center_hue = p->center_hue; n->reach = p->reach;
  n->select_sat = p->select_sat; n->sat_range = p->sat_range;
  n->select_light = p->select_light; n->light_range = p->light_range;
  n->feather = p->feather; n->neutral_protect = p->neutral_protect;
  n->invert = p->invert ? 1 : 0;
  n->shift_hue = p->shift_hue; n->shift_chroma = p->shift_chroma; n->shift_lightness = p->shift_lightness;
  n->convergence = p->convergence; n->affinity = p->affinity;
  n->neutral_zone = p->neutral_zone; n->priority = p->priority;
  n->per_component = p->per_component ? 1 : 0;
  n->conv_h = p->conv_h; n->aff_h = p->aff_h; n->nz_h = p->nz_h; n->prio_h = p->prio_h;
  n->conv_c = p->conv_c; n->aff_c = p->aff_c; n->nz_c = p->nz_c; n->prio_c = p->prio_c;
  n->conv_l = p->conv_l; n->aff_l = p->aff_l; n->nz_l = p->nz_l; n->prio_l = p->prio_l;
  n->absolute_target = p->absolute_target ? 1 : 0;
}

void commit_params(dt_iop_module_t *self, dt_iop_params_t *p1, dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  const dt_iop_colorwarp_params_t *p = (dt_iop_colorwarp_params_t *)p1;
  dt_iop_colorwarp_data_t *d = piece->data;

  dt_iop_colorwarp_node_t scratch;
  _cw_scratch_node(p, &scratch);          // the flat fields are the active node's live values
  const int nn = CLAMP(p->num_nodes, 1, CW_MAX_NODES);
  const int act = CLAMP(p->active_node, 0, nn - 1);
  d->num_nodes = nn;
  d->active_node = act;
  for(int i = 0; i < nn; i++)
    _cw_resolve_node((i == act) ? &scratch : &p->node[i], &d->nd[i]);

  d->smoothing = p->smoothing;
  d->edge_eps = fmaxf(p->edge * p->edge * 0.3f, 1e-4f);
  d->use_eigf = p->use_eigf ? 1 : 0;
  // EIGF feathering: high edge slider = preserve edges (low feathering), low = smooth across
  d->eigf_feather = CLAMP(powf(10.0f, (0.5f - p->edge) * 3.0f), 0.02f, 100.0f);
  d->corr_smooth = p->corr_smooth;
  d->input_smooth = p->input_smooth;
  d->polar_move = p->polar_move ? 1 : 0;
}

void init_pipe(dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = calloc(1, sizeof(dt_iop_colorwarp_data_t));
}

void cleanup_pipe(dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  free(piece->data);
  piece->data = NULL;
}

// paint the "select hue" slider with the actual dt UCS hue circle, so the user
// sees which colour family a slider position grabs.
// vivid sRGB (display) for an HSV hue [0,1], full saturation
static void _cw_hsv_vivid(const float h, float rgb[3])
{
  const float h6 = fmodf(h, 1.0f) * 6.0f;
  const float x = 1.0f - fabsf(fmodf(h6, 2.0f) - 1.0f);
  if      (h6 < 1.0f) { rgb[0] = 1; rgb[1] = x; rgb[2] = 0; }
  else if (h6 < 2.0f) { rgb[0] = x; rgb[1] = 1; rgb[2] = 0; }
  else if (h6 < 3.0f) { rgb[0] = 0; rgb[1] = 1; rgb[2] = x; }
  else if (h6 < 4.0f) { rgb[0] = 0; rgb[1] = x; rgb[2] = 1; }
  else if (h6 < 5.0f) { rgb[0] = x; rgb[1] = 0; rgb[2] = 1; }
  else                { rgb[0] = 1; rgb[1] = 0; rgb[2] = x; }
}

// Build a table mapping dt UCS hue position [0,1] -> vivid sRGB colour, so that a
// position on the slider/canvas shows the colour the engine actually selects there
// (no blue-shows-as-yellow mismatch).
#define CW_HUE_SAMPLES 360
typedef struct { float pos; float r, g, b; } cw_hs_t;

static int _cw_hs_cmp(const void *a, const void *b)
{
  const float pa = ((const cw_hs_t *)a)->pos, pb = ((const cw_hs_t *)b)->pos;
  return (pa < pb) ? -1 : (pa > pb) ? 1 : 0;
}

// vivid sRGB hues placed at their true dt UCS hue position, sorted by position
static void _cw_hue_samples(cw_hs_t s[CW_HUE_SAMPLES])
{
  const float L_white = Y_to_dt_UCS_L_star(1.0f);
  for(int j = 0; j < CW_HUE_SAMPLES; j++)
  {
    float disp[3];
    _cw_hsv_vivid((float)j / (float)CW_HUE_SAMPLES, disp);
    const float lr = powf(disp[0], 2.2f), lg = powf(disp[1], 2.2f), lb = powf(disp[2], 2.2f);
    dt_aligned_pixel_t XYZ = { 0.4124f * lr + 0.3576f * lg + 0.1805f * lb,
                               0.2126f * lr + 0.7152f * lg + 0.0722f * lb,
                               0.0193f * lr + 0.1192f * lg + 0.9505f * lb, 0.f };
    dt_aligned_pixel_t xyY, JCH;
    dt_D65_XYZ_to_xyY(XYZ, xyY);
    xyY_to_dt_UCS_JCH(xyY, L_white, JCH);
    s[j].pos = (JCH[2] + M_PI_F) / (2.0f * M_PI_F);
    s[j].r = disp[0]; s[j].g = disp[1]; s[j].b = disp[2];
  }
  qsort(s, CW_HUE_SAMPLES, sizeof(cw_hs_t), _cw_hs_cmp);
}

// interpolated colour at hue position p (linear between bracketing samples, wraps)
static void _cw_color_at(float p, const cw_hs_t s[CW_HUE_SAMPLES], float out[3])
{
  p -= floorf(p);
  int i, j; float t;
  if(p < s[0].pos || p >= s[CW_HUE_SAMPLES - 1].pos)  // wrap segment across the seam
  {
    i = CW_HUE_SAMPLES - 1; j = 0;
    const float span = s[0].pos + 1.0f - s[i].pos;
    const float pp = (p >= s[i].pos) ? (p - s[i].pos) : (p + 1.0f - s[i].pos);
    t = (span > 1e-6f) ? pp / span : 0.f;
  }
  else
  {
    i = 0;
    while(i < CW_HUE_SAMPLES - 2 && s[i + 1].pos <= p) i++;
    j = i + 1;
    const float span = s[j].pos - s[i].pos;
    t = (span > 1e-6f) ? (p - s[i].pos) / span : 0.f;
  }
  out[0] = s[i].r + t * (s[j].r - s[i].r);
  out[1] = s[i].g + t * (s[j].g - s[i].g);
  out[2] = s[i].b + t * (s[j].b - s[i].b);
}

static void _cw_paint_hue_slider(GtkWidget *w)
{
  cw_hs_t s[CW_HUE_SAMPLES];
  _cw_hue_samples(s);
  for(int i = 0; i < DT_BAUHAUS_SLIDER_MAX_STOPS; i++)
  {
    const float stop = (float)i / (float)(DT_BAUHAUS_SLIDER_MAX_STOPS - 1);
    float rgb[3];
    _cw_color_at(stop, s, rgb);
    dt_bauhaus_slider_set_stop(w, stop, rgb[0], rgb[1], rgb[2]);
  }
  gtk_widget_queue_draw(w);
}

// ---- selection canvas, 3 modes (2 axes shown of hue/saturation/lightness) ----
// Each mode maps two axes to (a, b) in [0,1]: a = angle-axis (hue on wheels, sat on the
// panel), b = radial/vertical axis. hue-bearing modes are wheels; sat/light is a panel.

// (a,b) -> screen point
static void _cw_ab_pt(int mode, double a, double b, double cx, double cy, double R,
                      double *X, double *Y)
{
  if(mode == 2) { *X = cx + (a * 2.0 - 1.0) * R; *Y = cy - (b * 2.0 - 1.0) * R; }  // panel
  else { const double ang = -2.0 * M_PI * a; *X = cx + cos(ang) * R * b; *Y = cy + sin(ang) * R * b; }
}
// screen point -> (a,b)
static void _cw_xy_ab(int mode, double x, double y, double cx, double cy, double R,
                      double *a, double *b)
{
  if(mode == 2)
  {
    *a = CLAMP((x - cx) / R * 0.5 + 0.5, 0.0, 1.0);
    *b = CLAMP(-(y - cy) / R * 0.5 + 0.5, 0.0, 1.0);
  }
  else
  {
    const double dx = x - cx, dy = y - cy;
    double ang = atan2(-dy, dx);
    if(ang < 0) ang += 2.0 * M_PI;
    *a = ang / (2.0 * M_PI);
    *b = CLAMP(hypot(dx, dy) / R, 0.0, 1.0);
  }
}
static void _cw_src_ab(int mode, const dt_iop_colorwarp_params_t *p, double *a, double *b)
{
  if(mode == 0) { *a = p->center_hue; *b = p->select_sat; }
  else if(mode == 1) { *a = p->center_hue; *b = p->select_light; }
  else { *a = p->select_sat; *b = p->select_light; }
}
static void _cw_tgt_ab(int mode, const dt_iop_colorwarp_params_t *p, double *a, double *b)
{
  if(mode == 0) { *a = p->center_hue + p->shift_hue; *b = CLAMP(p->select_sat + p->shift_chroma, 0.0, 1.0); }
  else if(mode == 1) { *a = p->center_hue + p->shift_hue; *b = CLAMP(p->select_light + p->shift_lightness, 0.0, 1.0); }
  else { *a = CLAMP(p->select_sat + p->shift_chroma, 0.0, 1.0); *b = CLAMP(p->select_light + p->shift_lightness, 0.0, 1.0); }
}

static void _cw_switch_node(dt_iop_module_t *self, int newnode);

// same as _cw_src_ab/_cw_tgt_ab but reading a stored node (used to draw the inactive nodes)
static void _cw_node_src_ab(int mode, const dt_iop_colorwarp_node_t *n, double *a, double *b)
{
  if(mode == 0) { *a = n->center_hue; *b = n->select_sat; }
  else if(mode == 1) { *a = n->center_hue; *b = n->select_light; }
  else { *a = n->select_sat; *b = n->select_light; }
}
static void _cw_node_tgt_ab(int mode, const dt_iop_colorwarp_node_t *n, double *a, double *b)
{
  if(mode == 0) { *a = n->center_hue + n->shift_hue; *b = CLAMP(n->select_sat + n->shift_chroma, 0.0, 1.0); }
  else if(mode == 1) { *a = n->center_hue + n->shift_hue; *b = CLAMP(n->select_light + n->shift_lightness, 0.0, 1.0); }
  else { *a = CLAMP(n->select_sat + n->shift_chroma, 0.0, 1.0); *b = CLAMP(n->select_light + n->shift_lightness, 0.0, 1.0); }
}

static gboolean _cw_draw(GtkWidget *widget, cairo_t *cr, dt_iop_module_t *self)
{
  dt_iop_colorwarp_gui_data_t *g = self->gui_data;
  dt_iop_colorwarp_params_t *p = self->params;
  const int mode = g->mode;
  GtkAllocation al;
  gtk_widget_get_allocation(widget, &al);
  const double cx = al.width / 2.0, cy = al.height / 2.0;
  const double R = fmin(al.width, al.height) / 2.0 - DT_PIXEL_APPLY_DPI(8.0);
  if(R < 4.0) return FALSE;

  cw_hs_t s[CW_HUE_SAMPLES];
  _cw_hue_samples(s);

  if(mode != 2)   // WHEEL: hue = angle; radius = saturation (mode 0) or lightness (mode 1)
  {
    // mode 1 fades to black at the centre (lightness), mode 0 to neutral grey (saturation)
    const double c_centre = (mode == 1) ? 0.0 : 0.5;
    const int NW = 120;
    cairo_pattern_t *mesh = cairo_pattern_create_mesh();
    for(int i = 0; i < NW; i++)
    {
      const double a0 = -2.0 * M_PI * i / NW, a1 = -2.0 * M_PI * (i + 1) / NW;
      float c0[3], c1[3];
      _cw_color_at((float)i / NW, s, c0);
      _cw_color_at((float)(i + 1) / NW, s, c1);
      cairo_mesh_pattern_begin_patch(mesh);
      cairo_mesh_pattern_move_to(mesh, cx, cy);
      cairo_mesh_pattern_line_to(mesh, cx + cos(a0) * R, cy + sin(a0) * R);
      cairo_mesh_pattern_line_to(mesh, cx + cos(a1) * R, cy + sin(a1) * R);
      cairo_mesh_pattern_line_to(mesh, cx, cy);
      cairo_mesh_pattern_set_corner_color_rgb(mesh, 0, c_centre, c_centre, c_centre);
      cairo_mesh_pattern_set_corner_color_rgb(mesh, 1, c0[0], c0[1], c0[2]);
      cairo_mesh_pattern_set_corner_color_rgb(mesh, 2, c1[0], c1[1], c1[2]);
      cairo_mesh_pattern_set_corner_color_rgb(mesh, 3, c_centre, c_centre, c_centre);
      cairo_mesh_pattern_end_patch(mesh);
    }
    cairo_arc(cr, cx, cy, R, 0, 2.0 * M_PI); cairo_set_source(cr, mesh); cairo_fill(cr);
    cairo_pattern_destroy(mesh);
    cairo_arc(cr, cx, cy, R, 0, 2.0 * M_PI); cairo_set_source_rgba(cr, 0.1, 0.1, 0.1, 0.35); cairo_fill(cr);
    cairo_arc(cr, cx, cy, R, 0, 2.0 * M_PI); cairo_set_source_rgba(cr, 1, 1, 1, 0.25);
    cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.0)); cairo_stroke(cr);
  }
  else            // PANEL: fixed hue, x = saturation, y = lightness
  {
    float hue[3];
    _cw_color_at(p->center_hue, s, hue);
    cairo_pattern_t *mesh = cairo_pattern_create_mesh();
    cairo_mesh_pattern_begin_patch(mesh);
    cairo_mesh_pattern_move_to(mesh, cx - R, cy + R);   // 0: S=0,L=0
    cairo_mesh_pattern_line_to(mesh, cx + R, cy + R);   // 1: S=1,L=0
    cairo_mesh_pattern_line_to(mesh, cx + R, cy - R);   // 2: S=1,L=1
    cairo_mesh_pattern_line_to(mesh, cx - R, cy - R);   // 3: S=0,L=1
    cairo_mesh_pattern_set_corner_color_rgb(mesh, 0, 0, 0, 0);
    cairo_mesh_pattern_set_corner_color_rgb(mesh, 1, 0, 0, 0);
    cairo_mesh_pattern_set_corner_color_rgb(mesh, 2, hue[0], hue[1], hue[2]);
    cairo_mesh_pattern_set_corner_color_rgb(mesh, 3, 0.82, 0.82, 0.82);
    cairo_mesh_pattern_end_patch(mesh);
    cairo_rectangle(cr, cx - R, cy - R, 2 * R, 2 * R);
    cairo_set_source(cr, mesh); cairo_fill(cr); cairo_pattern_destroy(mesh);
    cairo_rectangle(cr, cx - R, cy - R, 2 * R, 2 * R); cairo_set_source_rgba(cr, 0.1, 0.1, 0.1, 0.3); cairo_fill(cr);
    cairo_rectangle(cr, cx - R, cy - R, 2 * R, 2 * R); cairo_set_source_rgba(cr, 1, 1, 1, 0.25);
    cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.0)); cairo_stroke(cr);
  }

  // source & target handles + shift vector
  double sa, sb, ta, tb, sx, sy, tx, ty;
  _cw_src_ab(mode, p, &sa, &sb); _cw_ab_pt(mode, sa, sb, cx, cy, R, &sx, &sy);
  _cw_tgt_ab(mode, p, &ta, &tb); _cw_ab_pt(mode, ta, tb, cx, cy, R, &tx, &ty);

  // selection frame
  const double dash[] = { DT_PIXEL_APPLY_DPI(3), DT_PIXEL_APPLY_DPI(3) };
  cairo_set_dash(cr, dash, 2, 0);
  cairo_set_source_rgba(cr, 1, 1, 1, 0.8);
  cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.2));
  if(mode != 2)   // annular sector
  {
    const double dh = CLAMP(p->reach * 0.5, 0.0, 0.5) * 2.0 * M_PI;
    const double bc = (mode == 0) ? p->select_sat : p->select_light;
    const double br = (mode == 0) ? p->sat_range : p->light_range;
    const double r0 = CLAMP(bc - br * 0.5, 0.0, 1.0) * R;
    const double r1 = CLAMP(bc + br * 0.5, 0.0, 1.0) * R;
    const double ac = -2.0 * M_PI * p->center_hue;
    cairo_new_path(cr);
    cairo_arc(cr, cx, cy, r1, ac - dh, ac + dh);
    cairo_arc_negative(cr, cx, cy, r0, ac + dh, ac - dh);
    cairo_close_path(cr);
    cairo_stroke(cr);
  }
  else            // rectangle
  {
    double x0, y0, x1, y1;
    _cw_ab_pt(2, CLAMP(p->select_sat - p->sat_range * 0.5, 0.0, 1.0),
              CLAMP(p->select_light - p->light_range * 0.5, 0.0, 1.0), cx, cy, R, &x0, &y0);
    _cw_ab_pt(2, CLAMP(p->select_sat + p->sat_range * 0.5, 0.0, 1.0),
              CLAMP(p->select_light + p->light_range * 0.5, 0.0, 1.0), cx, cy, R, &x1, &y1);
    cairo_rectangle(cr, fmin(x0, x1), fmin(y0, y1), fabs(x1 - x0), fabs(y1 - y0));
    cairo_stroke(cr);
  }
  cairo_set_dash(cr, NULL, 0, 0);

  // other nodes: small dim source dots + faint move vector, so the whole field is visible
  const int nn = CLAMP(p->num_nodes, 1, CW_MAX_NODES);
  for(int i = 0; i < nn; i++)
  {
    if(i == p->active_node) continue;
    double na, nb, nx, ny, mta, mtb, mtx, mty;
    _cw_node_src_ab(mode, &p->node[i], &na, &nb); _cw_ab_pt(mode, na, nb, cx, cy, R, &nx, &ny);
    _cw_node_tgt_ab(mode, &p->node[i], &mta, &mtb); _cw_ab_pt(mode, mta, mtb, cx, cy, R, &mtx, &mty);
    cairo_move_to(cr, nx, ny); cairo_line_to(cr, mtx, mty);
    cairo_set_source_rgba(cr, 1, 1, 1, 0.35); cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.0)); cairo_stroke(cr);
    cairo_arc(cr, nx, ny, DT_PIXEL_APPLY_DPI(3.0), 0, 2.0 * M_PI);
    cairo_set_source_rgba(cr, 1, 1, 1, 0.55); cairo_fill_preserve(cr);
    cairo_set_source_rgba(cr, 0, 0, 0, 0.6); cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.0)); cairo_stroke(cr);
  }

  cairo_move_to(cr, sx, sy); cairo_line_to(cr, tx, ty);
  cairo_set_source_rgb(cr, 1, 1, 1); cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.5)); cairo_stroke(cr);
  cairo_arc(cr, sx, sy, DT_PIXEL_APPLY_DPI(4.0), 0, 2.0 * M_PI); cairo_set_source_rgb(cr, 1, 1, 1); cairo_fill_preserve(cr);
  cairo_set_source_rgb(cr, 0, 0, 0); cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.0)); cairo_stroke(cr);
  cairo_arc(cr, tx, ty, DT_PIXEL_APPLY_DPI(5.0), 0, 2.0 * M_PI); cairo_set_source_rgb(cr, 0.12, 0.12, 0.12); cairo_fill_preserve(cr);
  cairo_set_source_rgb(cr, 1, 1, 1); cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.5)); cairo_stroke(cr);

  return FALSE;
}

static void _cw_set_from_xy(dt_iop_module_t *self, double x, double y, int which)
{
  dt_iop_colorwarp_gui_data_t *g = self->gui_data;
  dt_iop_colorwarp_params_t *p = self->params;
  const int mode = g->mode;
  GtkAllocation al;
  gtk_widget_get_allocation(GTK_WIDGET(g->area), &al);
  const double cx = al.width / 2.0, cy = al.height / 2.0;
  const double R = fmin(al.width, al.height) / 2.0 - DT_PIXEL_APPLY_DPI(8.0);
  double a, b;
  _cw_xy_ab(mode, x, y, cx, cy, R, &a, &b);

  if(which == 1)   // source: move the selection centre(s); with absolute target, keep target fixed
  {
    const gboolean abso = p->absolute_target;
    if(mode == 0)
    {
      const double dc_h = a - p->center_hue, dc_s = b - p->select_sat;
      dt_bauhaus_slider_set(g->center_hue, a);
      dt_bauhaus_slider_set(g->select_sat, b);
      if(abso)
      {
        double sh = p->shift_hue - dc_h; sh -= round(sh);
        dt_bauhaus_slider_set(g->shift_hue, CLAMP(sh, -0.5, 0.5));
        dt_bauhaus_slider_set(g->shift_chroma, CLAMP(p->shift_chroma - dc_s, -1.0, 1.0));
      }
    }
    else if(mode == 1)
    {
      const double dc_h = a - p->center_hue, dc_l = b - p->select_light;
      dt_bauhaus_slider_set(g->center_hue, a);
      dt_bauhaus_slider_set(g->select_light, b);
      if(abso)
      {
        double sh = p->shift_hue - dc_h; sh -= round(sh);
        dt_bauhaus_slider_set(g->shift_hue, CLAMP(sh, -0.5, 0.5));
        dt_bauhaus_slider_set(g->shift_lightness, CLAMP(p->shift_lightness - dc_l, -1.0, 1.0));
      }
    }
    else
    {
      const double dc_s = a - p->select_sat, dc_l = b - p->select_light;
      dt_bauhaus_slider_set(g->select_sat, a);
      dt_bauhaus_slider_set(g->select_light, b);
      if(abso)
      {
        dt_bauhaus_slider_set(g->shift_chroma, CLAMP(p->shift_chroma - dc_s, -1.0, 1.0));
        dt_bauhaus_slider_set(g->shift_lightness, CLAMP(p->shift_lightness - dc_l, -1.0, 1.0));
      }
    }
  }
  else             // target: set the two shifts (relative to the source centre)
  {
    if(mode == 0 || mode == 1)   // a is hue (angular, wraps)
    {
      double dhue = a - p->center_hue;
      while(dhue > 0.5) dhue -= 1.0;
      while(dhue < -0.5) dhue += 1.0;
      dt_bauhaus_slider_set(g->shift_hue, CLAMP(dhue, -0.5, 0.5));
      if(mode == 0) dt_bauhaus_slider_set(g->shift_chroma, CLAMP(b - p->select_sat, -1.0, 1.0));
      else          dt_bauhaus_slider_set(g->shift_lightness, CLAMP(b - p->select_light, -1.0, 1.0));
    }
    else                          // sat/light panel
    {
      dt_bauhaus_slider_set(g->shift_chroma, CLAMP(a - p->select_sat, -1.0, 1.0));
      dt_bauhaus_slider_set(g->shift_lightness, CLAMP(b - p->select_light, -1.0, 1.0));
    }
  }
}

static gboolean _cw_press(GtkWidget *widget, GdkEventButton *e, dt_iop_module_t *self)
{
  dt_iop_colorwarp_gui_data_t *g = self->gui_data;
  dt_iop_colorwarp_params_t *p = self->params;
  if(e->button != GDK_BUTTON_PRIMARY) return FALSE;
  GtkAllocation al;
  gtk_widget_get_allocation(widget, &al);
  const double cx = al.width / 2.0, cy = al.height / 2.0;
  const double R = fmin(al.width, al.height) / 2.0 - DT_PIXEL_APPLY_DPI(8.0);
  double sa, sb, ta, tb, sx, sy, tx, ty;
  _cw_src_ab(g->mode, p, &sa, &sb); _cw_ab_pt(g->mode, sa, sb, cx, cy, R, &sx, &sy);
  _cw_tgt_ab(g->mode, p, &ta, &tb); _cw_ab_pt(g->mode, ta, tb, cx, cy, R, &tx, &ty);
  const double eps = DT_PIXEL_APPLY_DPI(12.0);
  g->drag = 0;
  if(hypot(e->x - tx, e->y - ty) < eps) g->drag = 2;                 // grab target
  else if(hypot(e->x - sx, e->y - sy) < eps) g->drag = 1;            // grab source
  if(!g->drag)
  {
    // click near another node's source dot -> select that node
    const int nn = CLAMP(p->num_nodes, 1, CW_MAX_NODES);
    for(int i = 0; i < nn; i++)
    {
      if(i == p->active_node) continue;
      double na, nb, nx, ny;
      _cw_node_src_ab(g->mode, &p->node[i], &na, &nb); _cw_ab_pt(g->mode, na, nb, cx, cy, R, &nx, &ny);
      if(hypot(e->x - nx, e->y - ny) < eps) { _cw_switch_node(self, i); return TRUE; }
    }
    g->drag = 1; _cw_set_from_xy(self, e->x, e->y, 1);               // click empty -> move source
  }
  return TRUE;
}

static gboolean _cw_motion(GtkWidget *widget, GdkEventMotion *e, dt_iop_module_t *self)
{
  dt_iop_colorwarp_gui_data_t *g = self->gui_data;
  if(g->drag) { _cw_set_from_xy(self, e->x, e->y, g->drag); return TRUE; }
  return FALSE;
}

static gboolean _cw_release(GtkWidget *widget, GdkEventButton *e, dt_iop_module_t *self)
{
  dt_iop_colorwarp_gui_data_t *g = self->gui_data;
  if(e->button == GDK_BUTTON_PRIMARY) { g->drag = 0; return TRUE; }
  return FALSE;
}

// declutter: show only the complementary axis's sliders by default (the two axes on
// the canvas are dragged there); "show all controls" reveals the rest for fine tuning.
static void _cw_update_visibility(dt_iop_module_t *self)
{
  dt_iop_colorwarp_gui_data_t *g = self->gui_data;
  const gboolean all = g->fine_toggle && gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(g->fine_toggle));
  const gboolean hue = all || (g->mode == 2);   // hue is complementary in sat/light view
  const gboolean sat = all || (g->mode == 1);   // saturation complementary in hue/light view
  const gboolean lgt = all || (g->mode == 0);   // lightness complementary in hue/sat view
  gtk_widget_set_visible(g->center_hue, hue);
  gtk_widget_set_visible(g->reach, hue);
  gtk_widget_set_visible(g->shift_hue, hue);
  gtk_widget_set_visible(g->select_sat, sat);
  gtk_widget_set_visible(g->sat_range, sat);
  gtk_widget_set_visible(g->shift_chroma, sat);
  gtk_widget_set_visible(g->select_light, lgt);
  gtk_widget_set_visible(g->light_range, lgt);
  gtk_widget_set_visible(g->shift_lightness, lgt);
}

static void _cw_fine_toggled(GtkWidget *w, dt_iop_module_t *self)
{
  _cw_update_visibility(self);
}

static void _cw_mode_changed(GtkWidget *w, dt_iop_module_t *self)
{
  dt_iop_colorwarp_gui_data_t *g = self->gui_data;
  g->mode = CLAMP(dt_bauhaus_combobox_get(w), 0, 2);
  dt_conf_set_int("plugins/darkroom/colorwarp/canvasmode", g->mode);
  _cw_update_visibility(self);
  if(g->area) gtk_widget_queue_draw(GTK_WIDGET(g->area));
}

static void _cw_mask_changed(GtkWidget *w, dt_iop_module_t *self)
{
  dt_iop_colorwarp_gui_data_t *g = self->gui_data;
  g->mask_mode = CLAMP(dt_bauhaus_combobox_get(w), 0, 2);
  dt_dev_reprocess_center(self->dev);   // re-run the pipe so process() picks up the flag
}

// affinity notebook: page 0 = global, pages 1-3 = per-component. The selected page IS the
// mode: on "global" the per-component sets are ignored, on any component page the global is.
static void _cw_aff_page(GtkNotebook *nb, GtkWidget *page, guint page_num, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  dt_iop_colorwarp_params_t *p = self->params;
  const int per = (page_num > 0) ? 1 : 0;
  if(p->per_component != per)
  {
    p->per_component = per;
    dt_dev_add_history_item(darktable.develop, self, TRUE);
  }
}

// ---- multi-node management (the flat fields are the live editor for the active node) ----
static void _cw_default_node(dt_iop_colorwarp_node_t *n)
{
  memset(n, 0, sizeof(*n));
  n->strength = 1.0f;
  n->center_hue = 0.5f; n->reach = 0.125f;   // ~45 degrees total hue band
  n->select_sat = 0.5f; n->sat_range = 1.0f;
  n->select_light = 0.5f; n->light_range = 1.0f;
  n->feather = 0.5f; n->neutral_protect = 0.0f;
}

// copy a node struct into the flat editor fields
static void _cw_node_to_flat(dt_iop_colorwarp_params_t *p, const dt_iop_colorwarp_node_t *n)
{
  p->strength = n->strength; p->center_hue = n->center_hue; p->reach = n->reach;
  p->select_sat = n->select_sat; p->sat_range = n->sat_range;
  p->select_light = n->select_light; p->light_range = n->light_range;
  p->feather = n->feather; p->neutral_protect = n->neutral_protect;
  p->invert = n->invert; p->shift_hue = n->shift_hue; p->shift_chroma = n->shift_chroma;
  p->shift_lightness = n->shift_lightness; p->convergence = n->convergence; p->affinity = n->affinity;
  p->neutral_zone = n->neutral_zone; p->priority = n->priority; p->per_component = n->per_component;
  p->conv_h = n->conv_h; p->aff_h = n->aff_h; p->nz_h = n->nz_h; p->prio_h = n->prio_h;
  p->conv_c = n->conv_c; p->aff_c = n->aff_c; p->nz_c = n->nz_c; p->prio_c = n->prio_c;
  p->conv_l = n->conv_l; p->aff_l = n->aff_l; p->nz_l = n->nz_l; p->prio_l = n->prio_l;
  p->absolute_target = n->absolute_target;
}

// push the flat editor fields into the widgets (callbacks suppressed)
static void _cw_sync_sliders(dt_iop_module_t *self)
{
  dt_iop_colorwarp_gui_data_t *g = self->gui_data;
  dt_iop_colorwarp_params_t *p = self->params;
  ++darktable.gui->reset;
  dt_bauhaus_slider_set(g->strength, p->strength);
  dt_bauhaus_slider_set(g->center_hue, p->center_hue);
  dt_bauhaus_slider_set(g->reach, p->reach);
  dt_bauhaus_slider_set(g->select_sat, p->select_sat);
  dt_bauhaus_slider_set(g->sat_range, p->sat_range);
  dt_bauhaus_slider_set(g->select_light, p->select_light);
  dt_bauhaus_slider_set(g->light_range, p->light_range);
  dt_bauhaus_slider_set(g->feather, p->feather);
  dt_bauhaus_slider_set(g->neutral_protect, p->neutral_protect);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->invert), p->invert);
  dt_bauhaus_slider_set(g->shift_hue, p->shift_hue);
  dt_bauhaus_slider_set(g->shift_chroma, p->shift_chroma);
  dt_bauhaus_slider_set(g->shift_lightness, p->shift_lightness);
  dt_bauhaus_slider_set(g->convergence, p->convergence);
  dt_bauhaus_slider_set(g->affinity, p->affinity);
  dt_bauhaus_slider_set(g->neutral_zone, p->neutral_zone);
  dt_bauhaus_slider_set(g->priority, p->priority);
  dt_bauhaus_slider_set(g->conv_h, p->conv_h); dt_bauhaus_slider_set(g->aff_h, p->aff_h);
  dt_bauhaus_slider_set(g->nz_h, p->nz_h); dt_bauhaus_slider_set(g->prio_h, p->prio_h);
  dt_bauhaus_slider_set(g->conv_c, p->conv_c); dt_bauhaus_slider_set(g->aff_c, p->aff_c);
  dt_bauhaus_slider_set(g->nz_c, p->nz_c); dt_bauhaus_slider_set(g->prio_c, p->prio_c);
  dt_bauhaus_slider_set(g->conv_l, p->conv_l); dt_bauhaus_slider_set(g->aff_l, p->aff_l);
  dt_bauhaus_slider_set(g->nz_l, p->nz_l); dt_bauhaus_slider_set(g->prio_l, p->prio_l);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->absolute_target), p->absolute_target);
  gtk_notebook_set_current_page(g->aff_notebook, p->per_component ? 1 : 0);
  --darktable.gui->reset;
}

static void _cw_rebuild_node_combo(dt_iop_module_t *self)
{
  dt_iop_colorwarp_gui_data_t *g = self->gui_data;
  dt_iop_colorwarp_params_t *p = self->params;
  if(p->num_nodes <= 1) g->view_all = FALSE;   // no "all nodes" entry when a single node
  ++darktable.gui->reset;
  dt_bauhaus_combobox_clear(g->node_combo);
  for(int i = 0; i < p->num_nodes; i++)
  {
    char s[24];
    snprintf(s, sizeof(s), _("node %d"), i + 1);
    dt_bauhaus_combobox_add(g->node_combo, s);
  }
  if(p->num_nodes > 1) dt_bauhaus_combobox_add(g->node_combo, _("all nodes"));   // mask overview
  dt_bauhaus_combobox_set(g->node_combo, g->view_all ? p->num_nodes : p->active_node);
  --darktable.gui->reset;
}

static void _cw_switch_node(dt_iop_module_t *self, int newnode)
{
  dt_iop_colorwarp_params_t *p = self->params;
  dt_iop_colorwarp_gui_data_t *g = self->gui_data;
  g->view_all = FALSE;
  newnode = CLAMP(newnode, 0, p->num_nodes - 1);
  _cw_scratch_node(p, &p->node[p->active_node]);   // save the current edits into the array
  p->active_node = newnode;
  _cw_node_to_flat(p, &p->node[newnode]);           // load the new node into the editor
  _cw_sync_sliders(self);
  if(g->node_combo)
  {
    ++darktable.gui->reset;
    dt_bauhaus_combobox_set(g->node_combo, newnode);
    --darktable.gui->reset;
  }
  if(g->area) gtk_widget_queue_draw(GTK_WIDGET(g->area));
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void _cw_node_combo_changed(GtkWidget *w, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  dt_iop_colorwarp_gui_data_t *g = self->gui_data;
  dt_iop_colorwarp_params_t *p = self->params;
  const int sel = dt_bauhaus_combobox_get(w);
  if(sel >= p->num_nodes)   // "all nodes" overview: mask spans every node, keep editing the active one
  {
    g->view_all = TRUE;
    dt_dev_reprocess_center(darktable.develop);
    return;
  }
  g->view_all = FALSE;
  if(sel != p->active_node) _cw_switch_node(self, sel);
  else dt_dev_reprocess_center(darktable.develop);
}

static void _cw_node_add(GtkWidget *w, dt_iop_module_t *self)
{
  dt_iop_colorwarp_params_t *p = self->params;
  if(p->num_nodes >= CW_MAX_NODES) return;
  ((dt_iop_colorwarp_gui_data_t *)self->gui_data)->view_all = FALSE;
  _cw_scratch_node(p, &p->node[p->active_node]);
  _cw_default_node(&p->node[p->num_nodes]);
  p->active_node = p->num_nodes;
  p->num_nodes++;
  _cw_node_to_flat(p, &p->node[p->active_node]);
  _cw_sync_sliders(self);
  _cw_rebuild_node_combo(self);
  dt_iop_colorwarp_gui_data_t *g = self->gui_data;
  if(g->area) gtk_widget_queue_draw(GTK_WIDGET(g->area));
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void _cw_node_remove(GtkWidget *w, dt_iop_module_t *self)
{
  dt_iop_colorwarp_params_t *p = self->params;
  if(p->num_nodes <= 1) return;
  for(int i = p->active_node; i < p->num_nodes - 1; i++) p->node[i] = p->node[i + 1];
  p->num_nodes--;
  p->active_node = CLAMP(p->active_node, 0, p->num_nodes - 1);
  _cw_node_to_flat(p, &p->node[p->active_node]);
  _cw_sync_sliders(self);
  _cw_rebuild_node_combo(self);
  dt_iop_colorwarp_gui_data_t *g = self->gui_data;
  if(g->area) gtk_widget_queue_draw(GTK_WIDGET(g->area));
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

void gui_update(dt_iop_module_t *self)
{
  _cw_sync_sliders(self);
  _cw_rebuild_node_combo(self);
  dt_iop_colorwarp_gui_data_t *g = self->gui_data;
  if(g && g->area) gtk_widget_queue_draw(GTK_WIDGET(g->area));
}

void gui_changed(dt_iop_module_t *self, GtkWidget *w, void *previous)
{
  dt_iop_colorwarp_gui_data_t *g = self->gui_data;
  if(g && g->area) gtk_widget_queue_draw(GTK_WIDGET(g->area));
}

// eyedropper: grab the selected hue directly from an image pixel/area
void color_picker_apply(dt_iop_module_t *self, GtkWidget *picker, dt_dev_pixelpipe_t *pipe)
{
  dt_iop_colorwarp_gui_data_t *g = self->gui_data;
  dt_iop_colorwarp_params_t *p = self->params;
  if(picker != g->center_hue) return;

  const dt_iop_order_iccprofile_info_t *const wp = dt_ioppr_get_pipe_work_profile_info(pipe);
  if(!wp) return;

  const float L_white = Y_to_dt_UCS_L_star(1.0f);
  dt_aligned_pixel_t JCH;
  dt_ioppr_rgb_matrix_to_dt_UCS_JCH((const float *)self->picked_color, JCH, wp->matrix_in_transposed, L_white);

  const float old_ch = p->center_hue;
  const float new_ch = (JCH[2] + M_PI_F) / (2.0f * M_PI_F);   // [-pi,pi] -> [0,1]
  DT_ENTER_GUI_UPDATE();
  p->center_hue = new_ch;
  dt_bauhaus_slider_set(g->center_hue, p->center_hue);
  if(p->absolute_target)   // keep the target fixed when re-grabbing a hue
  {
    float sh = p->shift_hue - (new_ch - old_ch); sh -= roundf(sh);
    p->shift_hue = CLAMP(sh, -0.5f, 0.5f);
    dt_bauhaus_slider_set(g->shift_hue, p->shift_hue);
  }
  DT_LEAVE_GUI_UPDATE();

  if(g->area) gtk_widget_queue_draw(GTK_WIDGET(g->area));
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_colorwarp_gui_data_t *g = IOP_GUI_ALLOC(colorwarp);

  // chroma-plane canvas on top of the panel
  self->widget = dt_gui_vbox();
  // recover a canvas height that got persisted as a sliver (accidental drag/scroll to ~0)
  if(dt_conf_get_int("plugins/darkroom/colorwarp/canvasheight") < 120)
    dt_conf_set_int("plugins/darkroom/colorwarp/canvasheight", 320);
  g->area = GTK_DRAWING_AREA(dt_ui_resize_wrap(NULL, 120, "plugins/darkroom/colorwarp/canvasheight"));
  gtk_widget_set_can_focus(GTK_WIDGET(g->area), TRUE);
  gtk_widget_set_tooltip_text(GTK_WIDGET(g->area),
                              _("chroma plane (hue = angle, saturation = radius).\n"
                                "drag the white dot to grab a colour family;\n"
                                "drag the dark dot to set the hue/saturation shift."));
  gtk_widget_add_events(GTK_WIDGET(g->area),
                        GDK_BUTTON_PRESS_MASK | GDK_POINTER_MOTION_MASK | GDK_BUTTON_RELEASE_MASK);
  g_signal_connect(G_OBJECT(g->area), "draw", G_CALLBACK(_cw_draw), self);
  g_signal_connect(G_OBJECT(g->area), "button-press-event", G_CALLBACK(_cw_press), self);
  g_signal_connect(G_OBJECT(g->area), "button-release-event", G_CALLBACK(_cw_release), self);
  g_signal_connect(G_OBJECT(g->area), "motion-notify-event", G_CALLBACK(_cw_motion), self);
  gtk_box_pack_start(GTK_BOX(self->widget), GTK_WIDGET(g->area), TRUE, TRUE, 0);

  // which two axes the canvas shows
  g->mode = CLAMP(dt_conf_get_int("plugins/darkroom/colorwarp/canvasmode"), 0, 2);
  g->mode_combo = dt_bauhaus_combobox_new(self);
  dt_bauhaus_widget_set_label(g->mode_combo, NULL, N_("canvas view"));
  dt_bauhaus_combobox_add(g->mode_combo, _("hue / saturation"));
  dt_bauhaus_combobox_add(g->mode_combo, _("hue / lightness"));
  dt_bauhaus_combobox_add(g->mode_combo, _("saturation / lightness"));
  dt_bauhaus_combobox_set(g->mode_combo, g->mode);
  gtk_widget_set_tooltip_text(g->mode_combo, _("which two selection axes the canvas shows.\n"
                                              "the third axis stays on its sliders."));
  g_signal_connect(G_OBJECT(g->mode_combo), "value-changed", G_CALLBACK(_cw_mode_changed), self);
  gtk_box_pack_start(GTK_BOX(self->widget), g->mode_combo, FALSE, FALSE, 0);

  // reveal the sliders for the axes already dragged on the canvas
  g->fine_toggle = gtk_check_button_new_with_label(_("show all controls"));
  gtk_widget_set_tooltip_text(g->fine_toggle,
                              _("also show the sliders for the two axes already on the canvas,\n"
                                "for precise numeric tuning"));
  g_signal_connect(G_OBJECT(g->fine_toggle), "toggled", G_CALLBACK(_cw_fine_toggled), self);
  gtk_box_pack_start(GTK_BOX(self->widget), g->fine_toggle, FALSE, FALSE, 0);

  // show the selection / correction as a grayscale mask (darkroom preview only)
  g->mask_mode = 0;
  g->mask_combo = dt_bauhaus_combobox_new(self);
  dt_bauhaus_widget_set_label(g->mask_combo, NULL, N_("show mask"));
  dt_bauhaus_combobox_add(g->mask_combo, _("off"));
  dt_bauhaus_combobox_add(g->mask_combo, _("correction intensity"));
  dt_bauhaus_combobox_add(g->mask_combo, _("selection"));
  dt_bauhaus_combobox_set(g->mask_combo, 0);
  gtk_widget_set_tooltip_text(g->mask_combo,
                              _("show a grayscale mask over the darkroom preview:\n"
                                "correction intensity = how much each pixel actually changes (affinity-aware),\n"
                                "selection = where the tool acts, before the move.\n"
                                "never affects thumbnails or export."));
  g_signal_connect(G_OBJECT(g->mask_combo), "value-changed", G_CALLBACK(_cw_mask_changed), self);
  gtk_box_pack_start(GTK_BOX(self->widget), g->mask_combo, FALSE, FALSE, 0);

  // node selector: the panel below edits the chosen attractor; all nodes act together (field)
  g->node_combo = dt_bauhaus_combobox_new(self);
  dt_bauhaus_widget_set_label(g->node_combo, NULL, N_("node"));
  gtk_widget_set_tooltip_text(g->node_combo, _("which attractor node the panel below edits.\n"
                                              "all nodes act together as one smooth field."));
  g_signal_connect(G_OBJECT(g->node_combo), "value-changed", G_CALLBACK(_cw_node_combo_changed), self);
  gtk_box_pack_start(GTK_BOX(self->widget), g->node_combo, FALSE, FALSE, 0);
  g->node_add = gtk_button_new_with_label(_("+ add node"));
  g_signal_connect(G_OBJECT(g->node_add), "clicked", G_CALLBACK(_cw_node_add), self);
  g->node_remove = gtk_button_new_with_label(_("− remove node"));
  g_signal_connect(G_OBJECT(g->node_remove), "clicked", G_CALLBACK(_cw_node_remove), self);
  GtkWidget *nrow = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, DT_BAUHAUS_SPACE);
  gtk_box_pack_start(GTK_BOX(nrow), g->node_add, TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(nrow), g->node_remove, TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(self->widget), nrow, FALSE, FALSE, 0);

  // master dose on top: it multiplies the whole H/S/L move below.
  g->strength = dt_bauhaus_slider_from_params(self, "strength");
  gtk_widget_set_tooltip_text(g->strength, _("master intensity: scales the whole shift below.\n"
                                             "set your hue/saturation/lightness shifts first,\n"
                                             "then use this to dose or A/B the effect.\n"
                                             "at 0 (or with all shifts at 0) the module does nothing."));

  g->center_hue = dt_color_picker_new(self, DT_COLOR_PICKER_AREA,
                                      dt_bauhaus_slider_from_params(self, "center_hue"));
  gtk_widget_set_tooltip_text(g->center_hue, _("hue of the colour family to grab.\n"
                                              "use the picker to grab it straight from the image,\n"
                                              "or right-click for the colour wheel."));
  // format "°" + factor 360 over a [0,1] range -> triggers the bauhaus colour-wheel popup
  dt_bauhaus_slider_set_format(g->center_hue, "°");
  dt_bauhaus_slider_set_factor(g->center_hue, 360.0f);
  dt_bauhaus_slider_set_feedback(g->center_hue, 0);   // no left-fill; show the hue bar only
  _cw_paint_hue_slider(g->center_hue);
  g->reach = dt_bauhaus_slider_from_params(self, "reach");
  gtk_widget_set_tooltip_text(g->reach, _("width of the hue band around the selected hue.\n"
                                         "max = all hues."));

  g->select_sat = dt_bauhaus_slider_from_params(self, "select_sat");
  gtk_widget_set_tooltip_text(g->select_sat, _("saturation to target (dull on the left, vivid on the\n"
                                              "right). paired with the range below."));
  g->sat_range = dt_bauhaus_slider_from_params(self, "sat_range");
  gtk_widget_set_tooltip_text(g->sat_range, _("how wide a saturation band around the target is affected.\n"
                                             "max = all saturations (neutrals always protected)."));

  g->select_light = dt_bauhaus_slider_from_params(self, "select_light");
  gtk_widget_set_tooltip_text(g->select_light, _("lightness to target (shadows on the left,\n"
                                                "highlights on the right). paired with the range below."));
  for(int i = 0; i < DT_BAUHAUS_SLIDER_MAX_STOPS; i++)   // black -> white reference ramp
  {
    const float s = (float)i / (float)(DT_BAUHAUS_SLIDER_MAX_STOPS - 1);
    dt_bauhaus_slider_set_stop(g->select_light, s, s, s, s);
  }
  dt_bauhaus_slider_set_feedback(g->select_light, 0);
  g->light_range = dt_bauhaus_slider_from_params(self, "light_range");
  gtk_widget_set_tooltip_text(g->light_range, _("how wide a lightness band around the target is affected.\n"
                                               "max = affect all lightnesses (no restriction)."));

  g->feather = dt_bauhaus_slider_from_params(self, "feather");
  gtk_widget_set_tooltip_text(g->feather, _("transition softness beyond the selected band.\n"
                                           "0 = hard edge; higher = softer Gaussian shoulder.\n"
                                           "the 'range' sliders set the fully-selected zone\n"
                                           "(range = max selects everything)."));

  g->neutral_protect = dt_bauhaus_slider_from_params(self, "neutral_protect");
  gtk_widget_set_tooltip_text(g->neutral_protect, _("fade the selection out at low chroma, where hue is noisy\n"
                                                   "(RAW shadows). raise to keep near-neutral pixels untouched."));

  g->invert = dt_bauhaus_toggle_from_params(self, "invert");
  gtk_widget_set_tooltip_text(g->invert, _("invert the selection: affect everything EXCEPT the\n"
                                          "selected hue/saturation/lightness region (neutrals stay protected)."));

  g->shift_hue = dt_bauhaus_slider_from_params(self, "shift_hue");
  gtk_widget_set_tooltip_text(g->shift_hue, _("rotate the grabbed colours' hue (scaled by strength)"));
  g->shift_chroma = dt_bauhaus_slider_from_params(self, "shift_chroma");
  gtk_widget_set_tooltip_text(g->shift_chroma, _("change the grabbed colours' saturation (scaled by strength)"));
  g->shift_lightness = dt_bauhaus_slider_from_params(self, "shift_lightness");
  gtk_widget_set_tooltip_text(g->shift_lightness, _("change the grabbed colours' lightness (scaled by strength)"));

  g->absolute_target = dt_bauhaus_toggle_from_params(self, "absolute_target");
  gtk_widget_set_tooltip_text(g->absolute_target, _("keep the target fixed when you re-grab a colour\n"
                                                   "(converge different families to the same target).\n"
                                                   "off = the target follows the selection (a grade)."));

  g->polar_move = dt_bauhaus_toggle_from_params(self, "polar_move");
  gtk_widget_set_tooltip_text(g->polar_move, _("how the move is applied.\n"
                                              "on (polar): ROTATE hue around neutral, keeping each\n"
                                              "pixel's saturation — natural for grading existing colours.\n"
                                              "off (a/b slide): move in the absolute chroma plane — tints\n"
                                              "neutrals cleanly, uniform tints, split toning."));

  // --- affinity: a notebook. page 0 = global (one affinity for all axes); pages 1-3 =
  //     per component. the selected page is the mode (global page vs per-component). ---
  GtkWidget *box = self->widget;
  static dt_action_def_t aff_nb_def = { 0 };
  g->aff_notebook = dt_ui_notebook_new(&aff_nb_def);
  dt_action_define_iop(self, NULL, N_("affinity page"), GTK_WIDGET(g->aff_notebook), &aff_nb_def);

  self->widget = dt_ui_notebook_page(g->aff_notebook, N_("global"), _("one affinity for all three axes"));
  g->convergence = dt_bauhaus_slider_from_params(self, "convergence");
  gtk_widget_set_tooltip_text(g->convergence, _("how the move is applied to the selected family:\n"
                                            "0 = translate (shift the whole family, keep its spread) — a grade.\n"
                                            "+ = converge toward the target (make the family uniform).\n"
                                            "− = diverge from the target (expand the family's variation)."));
  g->affinity = dt_bauhaus_slider_from_params(self, "affinity");
  gtk_widget_set_tooltip_text(g->affinity, _("shape the correction by distance to the target:\n"
                                            "+ = boost it; − = preserve a core already at the target,\n"
                                            "act only on the drift (core size = neutral zone)."));
  g->neutral_zone = dt_bauhaus_slider_from_params(self, "neutral_zone");
  gtk_widget_set_tooltip_text(g->neutral_zone, _("radius of the preserved core (with negative affinity)."));
  g->priority = dt_bauhaus_slider_from_params(self, "priority");
  gtk_widget_set_tooltip_text(g->priority, _("bias the move toward one side of the target. 0 = symmetric."));

  self->widget = dt_ui_notebook_page(g->aff_notebook, N_("hue"), _("affinity for the hue axis only"));
  g->conv_h = dt_bauhaus_slider_from_params(self, "conv_h");
  g->aff_h = dt_bauhaus_slider_from_params(self, "aff_h");
  g->nz_h = dt_bauhaus_slider_from_params(self, "nz_h");
  g->prio_h = dt_bauhaus_slider_from_params(self, "prio_h");

  self->widget = dt_ui_notebook_page(g->aff_notebook, N_("saturation"), _("affinity for the saturation axis only"));
  g->conv_c = dt_bauhaus_slider_from_params(self, "conv_c");
  g->aff_c = dt_bauhaus_slider_from_params(self, "aff_c");
  g->nz_c = dt_bauhaus_slider_from_params(self, "nz_c");
  g->prio_c = dt_bauhaus_slider_from_params(self, "prio_c");

  self->widget = dt_ui_notebook_page(g->aff_notebook, N_("lightness"), _("affinity for the lightness axis only"));
  g->conv_l = dt_bauhaus_slider_from_params(self, "conv_l");
  g->aff_l = dt_bauhaus_slider_from_params(self, "aff_l");
  g->nz_l = dt_bauhaus_slider_from_params(self, "nz_l");
  g->prio_l = dt_bauhaus_slider_from_params(self, "prio_l");

  self->widget = box;   // restore the main container
  gtk_box_pack_start(GTK_BOX(box), GTK_WIDGET(g->aff_notebook), FALSE, FALSE, 0);
  g_signal_connect(G_OBJECT(g->aff_notebook), "switch-page", G_CALLBACK(_cw_aff_page), self);

  g->smoothing = dt_bauhaus_slider_from_params(self, "smoothing");
  gtk_widget_set_tooltip_text(g->smoothing, _("edge-aware smoothing of the selection to remove\n"
                                              "speckle from noisy hues, while following image\n"
                                              "edges (no halo). raise it if a tight selection\n"
                                              "produces blotches."));
  g->edge = dt_bauhaus_slider_from_params(self, "edge");
  gtk_widget_set_tooltip_text(g->edge, _("how strongly the smoothing follows image edges.\n"
                                         "low: hug even faint edges (crisp selection, but\n"
                                         "speckle may survive near edges).\n"
                                         "high: smooth across edges (toward a plain blur, halos)."));

  g->use_eigf = dt_bauhaus_toggle_from_params(self, "use_eigf");
  gtk_widget_set_tooltip_text(g->use_eigf, _("exposure-independent guided filter for the mask.\n"
                                             "the default guided filter uses the scene-linear image\n"
                                             "as a guide, which is noisy in shadows; this variant is\n"
                                             "self-guided and behaves consistently from shadows to\n"
                                             "highlights. try it when low-chroma shadows stay noisy."));

  g->corr_smooth = dt_bauhaus_slider_from_params(self, "corr_smooth");
  gtk_widget_set_tooltip_text(g->corr_smooth, _("edge-aware smoothing of the applied correction itself\n"
                                                "(not just the selection). cuts the noise that converge,\n"
                                                "priority and affinity pick up from per-pixel values in\n"
                                                "low-chroma shadows. 0 = off (no extra cost)."));

  g->input_smooth = dt_bauhaus_slider_from_params(self, "input_smooth");
  gtk_widget_set_tooltip_text(g->input_smooth, _("denoise the data the selection is built from (lightness\n"
                                                 "and chromaticity) before thresholding, so a partial\n"
                                                 "selection in noisy low-chroma shadows stays clean at\n"
                                                 "its edges. best fix for shadow speckle; 0 = off\n"
                                                 "(otherwise allocates working buffers)."));

  // start on the affinity page that matches the saved mode
  gtk_notebook_set_current_page(g->aff_notebook,
                                ((dt_iop_colorwarp_params_t *)self->params)->per_component ? 1 : 0);
  _cw_rebuild_node_combo(self);

  // the 9 axis sliders are shown/hidden per canvas view -> keep show_all from forcing them
  GtkWidget *managed[] = { g->center_hue, g->reach, g->shift_hue,
                           g->select_sat, g->sat_range, g->shift_chroma,
                           g->select_light, g->light_range, g->shift_lightness };
  for(int i = 0; i < 9; i++) gtk_widget_set_no_show_all(managed[i], TRUE);
  _cw_update_visibility(self);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
