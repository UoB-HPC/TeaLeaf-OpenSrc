#include "kokkos_shared.hpp"
#include "set_chunk_data.hpp"
#include "set_chunk_state.hpp"
#include "cg.hpp"
#include "cheby.hpp"
#include "ppcg.hpp"
#include "jacobi.hpp"
#include "field_summary.hpp"
#include "solver_methods.hpp"
#include "local_halos.hpp"
#include "store_energy.hpp"
#include "pack_halos.hpp"
#include "../../kernel_interface.h"
#include "../../shared.h"

using namespace Kokkos;

void kernel_initialise(
    Settings* settings, int x, int y, KView* density0, 
    KView* density, KView* energy0, KView* energy, KView* u, 
    KView* u0, KView* p, KView* r, KView* mi, 
    KView* w, KView* kx, KView* ky, KView* sd, 
    KView* volume, KView* x_area, KView* y_area, KView* cell_x, 
    KView* cell_y, KView* cell_dx, KView* cell_dy, KView* vertex_dx, 
    KView* vertex_dy, KView* vertex_x, KView* vertex_y, KView* comms_buffer,
    Kokkos::View<double*>::HostMirror* host_comms_mirror, 
    double** cg_alphas, double** cg_betas, double** cheby_alphas, 
    double** cheby_betas);

void kernel_finalise(
    double* cg_alphas, double* cg_betas, double* cheby_alphas, 
    double* cheby_betas);

void run_set_chunk_data(Chunk* chunk, Settings* settings)
{
  START_PROFILING(settings->kernel_profile);

  double x_min = settings->grid_x_min + settings->dx*(double)chunk->left;
  double y_min = settings->grid_y_min + settings->dy*(double)chunk->bottom;

  SetChunkDataVertices<DEVICE> set_chunk_data_vertices(
      chunk->x, chunk->y, settings->halo_depth, chunk->vertex_x, 
      chunk->vertex_y, x_min, y_min, settings->dx, settings->dy);

  parallel_for(MAX(chunk->x,chunk->y)+1, set_chunk_data_vertices);

  SetChunkData<DEVICE> set_chunk_data(
      chunk->x, chunk->y, settings->halo_depth, chunk->vertex_x, 
      chunk->vertex_y, chunk->cell_x, chunk->cell_y, chunk->volume, 
      chunk->x_area, chunk->y_area, x_min, y_min, 
      settings->dx, settings->dy);

  parallel_for(chunk->x*chunk->y, set_chunk_data);

  STOP_PROFILING(settings->kernel_profile, __func__);
}

void run_set_chunk_state(Chunk* chunk, Settings* settings, State* states)
{
  START_PROFILING(settings->kernel_profile);

  SetChunkInitialState<DEVICE> set_chunk_initial_state(
      states[0].energy, states[0].density, chunk->energy0, chunk->density);

  parallel_for(chunk->x*chunk->y, set_chunk_initial_state);

  for(int ii = 1; ii < settings->num_states; ++ii)
  {
    SetChunkState<DEVICE> set_chunk_state(
        chunk->x, chunk->y, settings->halo_depth, states[ii], 
        chunk->energy0, chunk->density, chunk->u, chunk->cell_x, 
        chunk->cell_y, chunk->vertex_x, chunk->vertex_y);

    parallel_for(chunk->x*chunk->y, set_chunk_state);
  }

  STOP_PROFILING(settings->kernel_profile, __func__);
}

void run_kernel_initialise(Chunk* chunk, Settings* settings)
{
  kernel_initialise(settings, chunk->x, chunk->y, &(chunk->density0), 
      &(chunk->density), &(chunk->energy0), &(chunk->energy), 
      &(chunk->u), &(chunk->u0), &(chunk->p), &(chunk->r), 
      &(chunk->mi), &(chunk->w), &(chunk->kx), &(chunk->ky), 
      &(chunk->sd), &(chunk->volume), &(chunk->x_area), &(chunk->y_area),
      &(chunk->cell_x), &(chunk->cell_y), &(chunk->cell_dx), 
      &(chunk->cell_dy), &(chunk->vertex_dx), &(chunk->vertex_dy), 
      &(chunk->vertex_x), &(chunk->vertex_y), &(chunk->ext->comms_buffer),
      &(chunk->ext->host_comms_mirror), &(chunk->cg_alphas), 
      &(chunk->cg_betas), &(chunk->cheby_alphas), &(chunk->cheby_betas));
}

void run_kernel_finalise(Chunk* chunk, Settings* settings)
{    
  kernel_finalise(
      chunk->cg_alphas, chunk->cg_betas, chunk->cheby_alphas, 
      chunk->cheby_betas);
}

// Solver-wide kernels
void run_local_halos(
    Chunk* chunk, Settings* settings, int depth)
{
#define UPDATE_FACE(face, buffer, n) \
  if(chunk->neighbours[face] == EXTERNAL_FACE) \
  { \
    START_PROFILING(settings->kernel_profile); \
    LocalHalos<DEVICE> local_halos( \
        chunk->x, chunk->y, settings->halo_depth, buffer, face, depth); \
    parallel_for(n, local_halos); \
    STOP_PROFILING(settings->kernel_profile, #face); \
  }

#define LAUNCH_UPDATE(index, buffer) \
  if(settings->fields_to_exchange[index]) \
  {\
    UPDATE_FACE(CHUNK_LEFT, buffer, chunk->y*depth); \
    UPDATE_FACE(CHUNK_RIGHT, buffer, chunk->y*depth); \
    UPDATE_FACE(CHUNK_TOP, buffer, chunk->x*depth); \
    UPDATE_FACE(CHUNK_BOTTOM, buffer, chunk->x*depth); \
  }

  LAUNCH_UPDATE(FIELD_DENSITY, chunk->density);
  LAUNCH_UPDATE(FIELD_P, chunk->p);
  LAUNCH_UPDATE(FIELD_ENERGY0, chunk->energy0);
  LAUNCH_UPDATE(FIELD_ENERGY1, chunk->energy);
  LAUNCH_UPDATE(FIELD_U, chunk->u);
  LAUNCH_UPDATE(FIELD_SD, chunk->sd);
}

void run_pack_or_unpack(
    Chunk* chunk, Settings* settings, int depth,
    int face, bool pack, KView field, double* buffer)
{
  START_PROFILING(settings->kernel_profile);

  const int buffer_length = (face == CHUNK_LEFT || face == CHUNK_RIGHT)
    ? chunk->y*depth : chunk->x*depth;

  if(!pack)
  {
    KokkosHelper::PackMirror<double>(
        chunk->ext->host_comms_mirror, buffer, buffer_length); 
    Kokkos::deep_copy(chunk->ext->comms_buffer, chunk->ext->host_comms_mirror); 
  }

  HaloPacker<DEVICE> halo_packer(
      chunk->x, chunk->y, settings->halo_depth, 
      chunk->ext->comms_buffer, field, depth, face, pack);

  parallel_for(buffer_length, halo_packer);

  if(pack)
  {
    Kokkos::deep_copy(chunk->ext->host_comms_mirror, chunk->ext->comms_buffer);
    KokkosHelper::UnpackMirror<double>(
        buffer, chunk->ext->host_comms_mirror, buffer_length);
  }

  STOP_PROFILING(settings->kernel_profile, __func__);
}

void run_store_energy(Chunk* chunk, Settings* settings)
{
  START_PROFILING(settings->kernel_profile);

  StoreEnergy<DEVICE> store_energy(chunk->energy, chunk->energy0);

  parallel_for(chunk->x*chunk->y, store_energy);

  STOP_PROFILING(settings->kernel_profile, __func__);
}

void run_field_summary(
    Chunk* chunk, Settings* settings, 
    double* vol, double* mass, double* ie, double* temp)
{
  START_PROFILING(settings->kernel_profile);

  FieldSummary<DEVICE> field_summary(
      chunk->x, chunk->y, settings->halo_depth, chunk->u, chunk->density, 
      chunk->energy0, chunk->volume);
  FieldSummary<DEVICE>::value_type result;

  parallel_reduce(chunk->x*chunk->y, field_summary, result);

  *vol = result.vol;
  *mass = result.mass;
  *ie = result.ie;
  *temp = result.temp;

  STOP_PROFILING(settings->kernel_profile, __func__);
}

// CG solver kernels
void run_cg_init(
    Chunk* chunk, Settings* settings, 
    double rx, double ry, double* rro)
{
  START_PROFILING(settings->kernel_profile);

  CGInitU<DEVICE> cg_init_u(
      chunk->x, chunk->y, settings->coefficient, chunk->p, chunk->r, chunk->u, 
      chunk->w, chunk->density, chunk->energy);

  parallel_for(chunk->x*chunk->y, cg_init_u);

  CGInitK<DEVICE> cg_init_k(
      chunk->x, chunk->y, settings->halo_depth, chunk->w, chunk->kx, 
      chunk->ky, rx, ry);

  parallel_for(chunk->x*chunk->y, cg_init_k);

  CGInitOthers<DEVICE> cg_init_others(
      chunk->x, chunk->y, settings->halo_depth, chunk->kx, chunk->ky, 
      chunk->p, chunk->r, chunk->u, chunk->w);

  parallel_reduce(chunk->x*chunk->y, cg_init_others, *rro);

  STOP_PROFILING(settings->kernel_profile, __func__);
}

void run_cg_calc_w(Chunk* chunk, Settings* settings, double* pw)
{
  START_PROFILING(settings->kernel_profile);

  CGCalcW<DEVICE> cg_calc_w(
      chunk->x, chunk->y, settings->halo_depth, chunk->w, 
      chunk->p, chunk->kx, chunk->ky);

  parallel_reduce(
      TeamPolicy<DEVICE>(chunk->y - 2*settings->halo_depth, AUTO), 
      cg_calc_w, *pw);

  STOP_PROFILING(settings->kernel_profile, __func__);
}

void run_cg_calc_ur(
    Chunk* chunk, Settings* settings, double alpha, double* rrn)
{
  START_PROFILING(settings->kernel_profile);

  CGCalcUR<DEVICE> cg_calc_ur(
      chunk->x, chunk->y, settings->halo_depth, chunk->u, chunk->r, 
      chunk->p, chunk->w, alpha);

  parallel_reduce(
      TeamPolicy<DEVICE>(chunk->y - 2*settings->halo_depth, AUTO), 
      cg_calc_ur, *rrn);

  STOP_PROFILING(settings->kernel_profile, __func__);
}

void run_cg_calc_p(Chunk* chunk, Settings* settings, double beta)
{
  START_PROFILING(settings->kernel_profile);

  CGCalcP<DEVICE> cg_calc_p(
      chunk->x, chunk->y, settings->halo_depth, beta, chunk->p, chunk->r);

  parallel_for(
      TeamPolicy<DEVICE>(chunk->y - 2*settings->halo_depth, AUTO), 
      cg_calc_p);

  STOP_PROFILING(settings->kernel_profile, __func__);
}

// Chebyshev solver kernels
void run_cheby_init(Chunk* chunk, Settings* settings)
{
  START_PROFILING(settings->kernel_profile);

  ChebyInit<DEVICE> cheby_init(
      chunk->x, chunk->y, settings->halo_depth, chunk->theta, 
      chunk->p, chunk->r, chunk->u, chunk->u0, chunk->w, 
      chunk->kx, chunk->ky);

  parallel_for(chunk->x*chunk->y, cheby_init);

  STOP_PROFILING(settings->kernel_profile, __func__);
}

void run_cheby_iterate(
    Chunk* chunk, Settings* settings, double alpha, double beta)
{
  START_PROFILING(settings->kernel_profile);

  ChebyIterate<DEVICE> cheby_iterate(
      chunk->x, chunk->y, settings->halo_depth, alpha, beta, chunk->p, 
      chunk->r, chunk->u, chunk->u0, chunk->w, 
      chunk->kx, chunk->ky);

  parallel_for(
      TeamPolicy<DEVICE>(chunk->y - 2*settings->halo_depth, AUTO), 
      cheby_iterate);

  ChebyCalcU<DEVICE> cheby_calc_u(
      chunk->x, chunk->y, settings->halo_depth, chunk->p, chunk->u);

  parallel_for(
      TeamPolicy<DEVICE>(chunk->y - 2*settings->halo_depth, AUTO), 
      cheby_calc_u);

  STOP_PROFILING(settings->kernel_profile, __func__);
}

// Jacobi solver kernels
void run_jacobi_init(
    Chunk* chunk, Settings* settings, double rx, double ry)
{
  START_PROFILING(settings->kernel_profile);

  JacobiInit<DEVICE> jacobi_init(
      chunk->x, chunk->y, settings->halo_depth, 
      settings->coefficient, rx, ry, chunk->u, chunk->u0, 
      chunk->density, chunk->energy, chunk->kx, chunk->ky);

  parallel_for(chunk->x*chunk->y, jacobi_init);

  STOP_PROFILING(settings->kernel_profile, __func__);
}

void run_jacobi_iterate(
    Chunk* chunk, Settings* settings, double* error)
{
  START_PROFILING(settings->kernel_profile);

  JacobiCopyU<DEVICE> jacobi_copy_u(chunk->r, chunk->u);

  parallel_for(chunk->x*chunk->y, jacobi_copy_u);

  JacobiIterate<DEVICE> jacobi_iterate(
      chunk->x, chunk->y, settings->halo_depth, chunk->u, 
      chunk->u0, chunk->r, chunk->kx, chunk->ky);

  parallel_reduce(chunk->x*chunk->y, jacobi_iterate, *error);

  STOP_PROFILING(settings->kernel_profile, __func__);
}

// PPCG solver kernels
void run_ppcg_init(Chunk* chunk, Settings* settings)
{
  START_PROFILING(settings->kernel_profile);

  PPCGInit<DEVICE> ppcg_init(
      chunk->x, chunk->y, settings->halo_depth, chunk->theta, 
      chunk->sd, chunk->r);

  parallel_for(chunk->x*chunk->y, ppcg_init);

  STOP_PROFILING(settings->kernel_profile, __func__);
}

void run_ppcg_inner_iteration(
    Chunk* chunk, Settings* settings, double alpha, double beta)
{
  START_PROFILING(settings->kernel_profile);

  PPCGCalcUR<DEVICE> ppcg_calc_ur(
      chunk->x, chunk->y, settings->halo_depth, chunk->sd, chunk->r, 
      chunk->u, chunk->kx, chunk->ky);

  parallel_for(
      TeamPolicy<DEVICE>(chunk->y - 2*settings->halo_depth, AUTO), 
      ppcg_calc_ur);

  PPCGCalcSd<DEVICE> ppcg_calc_sd(
      chunk->x, chunk->y, settings->halo_depth, chunk->theta, alpha, beta, 
      chunk->sd, chunk->r);

  parallel_for(
      TeamPolicy<DEVICE>(chunk->y - 2*settings->halo_depth, AUTO), 
      ppcg_calc_sd);

  STOP_PROFILING(settings->kernel_profile, __func__);
}

// Shared solver kernels
void run_copy_u(Chunk* chunk, Settings* settings)
{
  START_PROFILING(settings->kernel_profile);

  CopyU<DEVICE> copy_u(
      chunk->x, chunk->y, settings->halo_depth, chunk->u, chunk->u0);

  parallel_for(chunk->x*chunk->y, copy_u);

  STOP_PROFILING(settings->kernel_profile, __func__);
}

void run_calculate_residual(Chunk* chunk, Settings* settings)
{
  START_PROFILING(settings->kernel_profile);

  CalculateResidual<DEVICE> calculate_residual(
      chunk->x, chunk->y, settings->halo_depth, chunk->u, 
      chunk->u0, chunk->r, chunk->kx, chunk->ky);

  parallel_for(chunk->x*chunk->y, calculate_residual);

  STOP_PROFILING(settings->kernel_profile, __func__);
}

void run_calculate_2norm(
    Chunk* chunk, Settings* settings, KView buffer, double* norm)
{
  START_PROFILING(settings->kernel_profile);

  Calculate2Norm<DEVICE> calculate_2norm(
      chunk->x, chunk->y, settings->halo_depth, buffer);

  parallel_reduce(chunk->x*chunk->y, calculate_2norm, *norm);

  STOP_PROFILING(settings->kernel_profile, __func__);
}

void run_finalise(Chunk* chunk, Settings* settings)
{
  START_PROFILING(settings->kernel_profile);

  Finalise<DEVICE> finalise(
      chunk->x, chunk->y, settings->halo_depth, chunk->u, 
      chunk->density, chunk->energy);

  parallel_for(chunk->x*chunk->y, finalise);

  STOP_PROFILING(settings->kernel_profile, __func__);
}


