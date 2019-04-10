//TODO -- this file should be updated to not rely on environment variables
//        but can I determine node from rank, local rank from rank,
//        or global rank from local rank and node, without ordering?
#ifndef NAPCOMM_TOPO_DATA_HPP
#define NAPCOMM_TOPO_DATA_HPP

#include <mpi.h>
#ifdef USING_CRAY
#include "topo_cray.hpp"
#else
#include "topo_default.hpp"
#endif

struct topo_data{
    int rank_ordering;
    int ppn;
    int num_nodes;
    int rank_node;
    MPI_Comm local_comm;

    topo_data(MPI_Comm mpi_comm)
    {
        int rank, num_procs;
        MPI_Comm_rank(mpi_comm, &rank);
        MPI_Comm_size(mpi_comm, &num_procs);

#ifdef USING_MPI3
        MPI_Comm_split_type(mpi_comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);

        // Determine processes per node
        MPI_Comm_size(local_comm, &ppn);

        // Determine number of nodes
        // Assuming num_nodes divides num_procs evenly
        num_nodes = num_procs / ppn;
#else
        char* ppn_char = getenv("PPN");
        ppn = 16;
        if (ppn_char)
        {
            ppn = atoi(ppn_char);
        }
        num_nodes = num_procs / ppn;
#endif

        rank_ordering = get_ordering();  // If not Cray or BGQ, assuming SMP style order
        rank_node = get_node(rank, rank_ordering, num_nodes, ppn);

#ifndef USING_MPI3
        MPI_Comm_split(mpi_comm, rank_node, rank, &local_comm);
#endif
    }

    ~topo_data()
    {
        MPI_Comm_free(&local_comm);
    }
};

static int get_node(const int proc, const topo_data* topo_info)
{
    return get_node(proc, topo_info->rank_ordering, topo_info->num_nodes,
            topo_info->ppn);
}
static int get_local_proc(const int proc, const topo_data* topo_info)
{
    return get_local_proc(proc, topo_info->rank_ordering, topo_info->num_nodes,
            topo_info->ppn);
}
static int get_global_proc(const int node, const int local_proc, const topo_data* topo_info)
{
    return get_global_proc(node, local_proc, topo_info->rank_ordering,
            topo_info->num_nodes, topo_info->ppn);
}

#endif
