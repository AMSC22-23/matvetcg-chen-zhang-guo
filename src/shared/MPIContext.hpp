#ifndef MPICONTEXT_HPP
#define MPICONTEXT_HPP

#include <mpi.h>

class MPIContext {
public:
  MPIContext(MPI_Comm* mpi_comm, const int mpi_rank)
      : m_mpi_comm(mpi_comm), m_mpi_rank(mpi_rank) {}
  
  MPI_Comm* mpi_comm() const {
    return m_mpi_comm;
  }

  int mpi_rank() const {
    return m_mpi_rank;
  } 

private:
  MPI_Comm*   m_mpi_comm;
  int         m_mpi_rank;
};

#endif //MPICONTEXT_HPP
