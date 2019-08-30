#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include "src/allreduce/nap_allreduce.hpp"

//using namespace raptor;
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int ppn = 16;
    double t0, tfinal;

    MPI_Comm local_comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank / ppn, rank, &local_comm);

    std::vector<int> step_sizes;
    int num_nodes = num_procs / ppn;
    int size = num_nodes;
    while (size > ppn)
    {
        size /= ppn;
        step_sizes.push_back(ppn);
    }
    if (size > 1) step_sizes.push_back(size);
    if (rank == 0) printf("NumNodes %d, NumSteps %d:\n", num_nodes, step_sizes.size());
    if (rank == 0) for (int i = 0; i < step_sizes.size(); i++) printf("Size[%d] = %d\n", i, step_sizes[i]);

    int n_masters = num_procs / ppn;
    int masters[n_masters];
    for (int i = 0; i < n_masters; i++)
        masters[i] = i*ppn;

    MPI_Group world_group;
    MPI_Group master_group;
    MPI_Comm master_comm;

    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_incl(world_group, n_masters, masters, &master_group);

    if (rank % ppn == 0)
    {
        MPI_Comm_create_group(MPI_COMM_WORLD, master_group, 0, &master_comm);
    }

    int min_i = 0;
    int max_i = 17;
    int max_size = pow(2, max_i);
    std::vector<double> vals(max_size, 0.005);
    std::vector<double> ref_sol(max_size);
    std::vector<double> sol(max_size);

    int max_n_tests = 10000;
    int n_tests = max_n_tests;


    // Run Tests: 
    if (rank == 0) printf("MPI Allreduce:");
    MPI_Allreduce(vals.data(), ref_sol.data(), size, MPI_DOUBLE, 
            MPI_SUM, MPI_COMM_WORLD);
    for (int i = min_i; i < max_i; i++)
    {
        if (rank == 0) printf("\t");
        size = pow(2,i);
        if (size > 10000) n_tests = 100;
        else n_tests = max_n_tests;

        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int test = 0; test < n_tests; test++)
            MPI_Allreduce(vals.data(), ref_sol.data(), size, MPI_DOUBLE, 
                    MPI_SUM, MPI_COMM_WORLD);
        tfinal = (MPI_Wtime() - t0) / n_tests;
        MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0) printf("%e", t0);
    }
    if (rank == 0) printf("\n");

    if (rank == 0) printf("RD Allreduce:");
    MPIX_Allreduce_RD(vals.data(), sol.data(), size, MPI_DOUBLE, 
           MPI_COMM_WORLD);
    for (int i = min_i; i < max_i; i++)
    {
        if (rank == 0) printf("\t");
        size = pow(2,i);
        if (size > 10000) n_tests = 100;
        else n_tests = max_n_tests;

        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int test = 0; test < n_tests; test++)
            MPIX_Allreduce_RD(vals.data(), sol.data(), size, MPI_DOUBLE, 
               MPI_COMM_WORLD);
        tfinal = (MPI_Wtime() - t0) / n_tests;
        MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0) printf("%e", t0);
    }
    if (rank == 0) printf("\n");

    if (rank == 0) printf("SMP Allreduce:");
    MPIX_Allreduce_SMP(vals.data(), sol.data(), size, MPI_DOUBLE,
        local_comm, master_comm);
    for (int i = min_i; i < max_i; i++)
    {
        if (rank == 0) printf("\t");
        size = pow(2,i);
        if (size > 10000) n_tests = 100;
        else n_tests = max_n_tests;

        MPI_Barrier(MPI_COMM_WORLD); 
        t0 = MPI_Wtime();
        for (int test = 0; test < n_tests; test++)
            MPIX_Allreduce_SMP(vals.data(), sol.data(), size, MPI_DOUBLE,
                local_comm, master_comm);
        tfinal = (MPI_Wtime() - t0) / n_tests;
        MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0) printf("%e", t0);
    }
    if (rank == 0) printf("\n");

    if (rank == 0) printf("NAP Allreduce:");
    MPIX_Allreduce_NAP(vals.data(), sol.data(), size, MPI_DOUBLE,
        local_comm, ppn, step_sizes.data(), step_sizes.size());
    for (int i = min_i; i < max_i; i++)
    {
        if (rank == 0) printf("\t");
        size = pow(2,i);
        if (size > 10000) n_tests = 100;
        else n_tests = max_n_tests;

        MPI_Barrier(MPI_COMM_WORLD); 
        t0 = MPI_Wtime();
        for (int test = 0; test < n_tests; test++)
            MPIX_Allreduce_NAP(vals.data(), sol.data(), size, MPI_DOUBLE,
                local_comm, ppn, step_sizes.data(), step_sizes.size());
        tfinal = (MPI_Wtime() - t0) / n_tests;
        MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0) printf("%e", t0);
    }
    if (rank == 0) printf("\n");
   
    MPI_Comm_free(&local_comm);
    MPI_Group_free(&world_group);
    MPI_Group_free(&master_group);
    if (rank % ppn == 0) MPI_Comm_free(&master_comm);

    MPI_Finalize();
    return 0;
}
