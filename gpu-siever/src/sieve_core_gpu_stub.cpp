/* Stands in for sieve_core_gpu.cu in builds without CUDA. */

#include "sieve_core.hpp"

#include <cstdio>
#include <cstdlib>

void gpu_warmup()
{
}

bool gpu_init(std::vector<fb_entry> const *, int, uint32_t,
              tiny_table const &, filter_params const &)
{
    return false;
}

int gpu_slots()
{
    return 0;
}

void sieve_one_gpu(sq_task const &, std::vector<fb_entry> const *,
                   tiny_table const &, chunk_sink const &)
{
    fprintf(stderr, "glas was built without CUDA\n");
    abort();
}

void gpu_cofactor(std::vector<cofac_pair> const &,
                  std::vector<cofac_result> &, int const[2], bool)
{
    fprintf(stderr, "glas was built without CUDA\n");
    abort();
}
