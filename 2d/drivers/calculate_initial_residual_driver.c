#include <math.h>
#include <float.h>
#include "../comms.h"
#include "drivers.h"
#include "../kernel_interface.h"
#include "../chunk.h"


void calculate_initial_error(
        Chunk* chunks, Settings* settings, double* error)
{
  for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
  {
      double tile_error = 0;
      if(settings->kernel_language == C)
      {
          run_copy_u(&(chunks[cc]), settings);
          run_calculate_residual(&(chunks[cc]), settings);
          run_calculate_2norm(
                  &(chunks[cc]), settings, chunks[cc].r, &tile_error);
      }
      else if(settings->kernel_language == FORTRAN)
      {
      }
      *error += tile_error;
  }
  sum_over_ranks(settings, error);
}

void calculate_initial_residual_driver(
  Chunk* chunks, Settings* settings)
{
  double error = 0;
  calculate_initial_error(chunks, settings, &error);
  settings->initial_residual = sqrt(fabs(error));
}
