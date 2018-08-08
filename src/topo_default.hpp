//TODO -- this file should be updated to not rely on environment variables
//        but can I determine node from rank, local rank from rank, 
//        or global rank from local rank and node, without ordering?
#ifndef NAPCOMM_TOPO_DEFAULT_HPP
#define NAPCOMM_TOPO_DEFAULT_HPP

#include <mpi.h>

#define SMP 1

int get_ordering()
{
    return SMP;
}

int get_node(const int proc, const int rank_ordering, const int num_nodes, const int ppn)
{
    return proc / ppn;
}

int get_local_proc(const int proc, const int rank_ordering, const int num_nodes, const int ppn)
{
    return proc % ppn;
}

int get_global_proc(const int node, const int local_proc, const int rank_ordering, 
        const int num_nodes, const int ppn)
{
    return local_proc + (node * ppn);
}

#endif
