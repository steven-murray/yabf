"""Module to keep track of MPI things.

Most functions copied from cobaya.mpi.
"""

# Vars to keep track of MPI parameters
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

_mpi_size = -1
_mpi_comm = -1
_mpi_rank = -1

mpi_comm = getattr(MPI, "COMM_WORLD", None)
mpi_size = getattr(mpi_comm, "Get_size", lambda: 0)()
mpi_rank = getattr(mpi_comm, "Get_rank", lambda: None)()
am_single_or_primary_process = not bool(mpi_rank)
more_than_one_process = mpi_size > 1


def sync_processes():
    if more_than_one_process:
        mpi_comm.barrier()
