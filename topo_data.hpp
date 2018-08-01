//TODO -- this file should be updated to not rely on environment variables
// USE MPI_Comm_split_type??

#ifndef NAPCOMM_TOPO_DATA_HPP
#define NAPCOMM_TOPO_DATA_HPP

#include <mpi.h>

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

        char* proc_layout_c = getenv("MPICH_RANK_REORDER_METHOD");
        char* ppn_char = getenv("PPN");

        rank_ordering = 1;
        if (proc_layout_c)
        {
            rank_ordering = atoi(proc_layout_c);
        }
        ppn = 16;
        if (ppn_char)
        {
            ppn = atoi(ppn_char);
        }

        num_nodes = num_procs / ppn;
        if (num_procs % ppn) num_nodes++;

        if (rank_ordering == 0)
        {
            rank_node = rank % num_nodes;
        }
        else if (rank_ordering == 1)
        {
            rank_node = rank / ppn;
        }
        else if (rank_ordering == 2)
        {
            if ((rank / num_nodes) % 2 == 0)
            {
                rank_node = rank % num_nodes;
            }
            else
            {
                rank_node = num_nodes - (rank % num_nodes) - 1;
            }
        }
        MPI_Comm_split(mpi_comm, rank_node, rank, &local_comm);   
    }

    ~topo_data()
    {
        MPI_Comm_free(&local_comm);
    }
};

int get_node(const int proc, const topo_data* topo_info)
{
    if (topo_info->rank_ordering == 0)
    {
        return proc % topo_info->num_nodes;
    }
    else if (topo_info->rank_ordering == 1)
    {
        return proc / topo_info->ppn;
    }
    else if (topo_info->rank_ordering == 2)
    {
        if ((proc / topo_info->num_nodes) % 2 == 0)
        {
            return proc % topo_info->num_nodes;
        }
        else
        {
            return topo_info->num_nodes - (proc % topo_info->num_nodes) - 1;
        }
    }
    else
    { 
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
        {
            printf("This MPI rank ordering is not supported!\n");
        }
        return -1;
    }
}

int get_local_proc(const int proc, const topo_data* topo_info)
{
    if (topo_info->rank_ordering == 0 || topo_info->rank_ordering == 2)
    {
        return proc / topo_info->num_nodes;
    }
    else if (topo_info->rank_ordering == 1)
    {
        return proc % topo_info->ppn;
    }
    else
    { 
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
        {
            printf("This MPI rank ordering is not supported!\n");
        }
        return -1;
    }
}

int get_global_proc(const int node, const int local_proc, const topo_data* topo_info)
{
    if (topo_info->rank_ordering == 0)
    {
        return local_proc * topo_info->num_nodes + node;
    }
    else if (topo_info->rank_ordering == 1)
    {
        return local_proc + (node * topo_info->ppn);
    }
    else if (topo_info->rank_ordering == 2)
    {
        if (local_proc % 2 == 0)
        {
            return local_proc * topo_info->num_nodes + node;
        }
        else
        {
            return local_proc * topo_info->num_nodes + topo_info->num_nodes - node - 1;                
        }
    }
    else
    { 
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
        {
            printf("This MPI rank ordering is not supported!\n");
        }
        return -1;
    }
}


#endif
