package edu.coursera.distributed;

import edu.coursera.distributed.util.MPI;
import edu.coursera.distributed.util.MPI.MPIException;

class MatrixMult {

    static void parallelMatrixMultiply(Matrix a, Matrix b, Matrix c,
                                       final MPI mpi) throws MPIException {


        //identify rank of current process
        final int myRank = mpi.MPI_Comm_rank(mpi.MPI_COMM_WORLD);
        //number of ranks we are working with
        final int size = mpi.MPI_Comm_size(mpi.MPI_COMM_WORLD);


        int averow, dest, extra, source;
        int[] offset = new int[]{0};
        int[] rows = new int[]{0};

        if (myRank == 0) {
            averow = a.getNRows() / (size - 1);
            extra = a.getNRows() % (size - 1);
            offset[0] = 0;

            for (dest = 1; dest <= size - 1; dest++) {
                rows[0] = (dest <= extra) ? averow + 1 : averow;

                mpi.MPI_Send(offset, 0, 1, dest, 1, mpi.MPI_COMM_WORLD);
                mpi.MPI_Send(rows, 0, 1, dest, 1, mpi.MPI_COMM_WORLD);
                mpi.MPI_Send(a.getValues(), 0, rows[0] * a.getNCols(), dest, 1, mpi.MPI_COMM_WORLD);
                mpi.MPI_Send(b.getValues(), 0, b.getNRows() * b.getNCols(), dest,
                        1, mpi.MPI_COMM_WORLD);

                offset[0] = offset[0] + rows[0];
            }

            for (source = 1; source <= size - 1; source++) {
                mpi.MPI_Recv(offset, 0, 1, source, 2, mpi.MPI_COMM_WORLD);
                mpi.MPI_Recv(rows, 0, 1, source, 2, mpi.MPI_COMM_WORLD);
                mpi.MPI_Recv(c.getValues(), 0, c.getNCols() * c.getNRows(), source, 2, mpi.MPI_COMM_WORLD);
            }

        } else {
            mpi.MPI_Recv(offset, 0, 1, 0, 1, mpi.MPI_COMM_WORLD);
            mpi.MPI_Recv(rows, 0, 1, 0, 1, mpi.MPI_COMM_WORLD);
            mpi.MPI_Recv(a.getValues(), 0, rows[0] * a.getNCols(), 0, 1, mpi.MPI_COMM_WORLD);
            mpi.MPI_Recv(b.getValues(), 0, b.getNRows() * b.getNCols(), 0,
                    1, mpi.MPI_COMM_WORLD);

            for (int i = 0; i < b.getNCols(); i++) {
                for (int j = 0; j < rows[0]; j++) {
                    c.set(i, j, 0.0);

                    for (int k = 0; k < a.getNCols(); k++) {
                        c.incr(i, j, a.get(i, k) * b.get(k, j));
                    }
                }
            }

            mpi.MPI_Send(offset, 0, 1, 0, 2, mpi.MPI_COMM_WORLD);
            mpi.MPI_Send(rows, 0, 1, 0, 2, mpi.MPI_COMM_WORLD);
            mpi.MPI_Send(c.getValues(), 0, c.getNCols() * c.getNRows(), 0, 2, mpi.MPI_COMM_WORLD);
        }
    }
}

