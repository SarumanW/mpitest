package edu.coursera.distributed;

import edu.coursera.distributed.util.MPI;
import edu.coursera.distributed.util.MPI.MPIException;

public class CollectiveMatrixMult {

    public static void parallelMatrixMultiply(Matrix a, Matrix b, Matrix c,
                                              final MPI mpi) throws MPIException {


        //identify rank of current process
        final int myRank = mpi.MPI_Comm_rank(mpi.MPI_COMM_WORLD);
        //number of ranks we are working with
        final int size = mpi.MPI_Comm_size(mpi.MPI_COMM_WORLD);

        //number of rows we want to compute
        final int nRows = c.getNRows();

        //how many rows each process will have
        final int rowChunk = (nRows + size - 1) / size;

        //start and end row for the current rank
        final int startRow = myRank * rowChunk;
        int endRow = (myRank + 1) * rowChunk;

        if (endRow > nRows) {
            endRow = nRows;
        }

        if (myRank == 0) {
            mpi.MPI_Bcast(a.getValues(), 0, a.getNRows() * a.getNCols(),
                    0, mpi.MPI_COMM_WORLD);
            mpi.MPI_Bcast(b.getValues(), 0, b.getNRows() * b.getNCols(),
                    0, mpi.MPI_COMM_WORLD);


            MPI.MPI_Request[] requests = new MPI.MPI_Request[size - 1];

            for (int i = 1; i < size; i++) {
                final int rankStartRow = i * rowChunk;
                int rankEndRow = (i + 1) * rowChunk;

                if (rankEndRow > nRows) {
                    rankEndRow = nRows;
                }

                final int rowOffset = rankStartRow * c.getNCols();
                final int nElements = (rankEndRow - rankStartRow) * c.getNCols();

                requests[i - 1] = mpi.MPI_Irecv(c.getValues(), rowOffset, nElements,
                        i, i, mpi.MPI_COMM_WORLD);
            }

            mpi.MPI_Waitall(requests);

        } else {

            for (int i = startRow; i < endRow; i++) {
                for (int j = 0; j < c.getNCols(); j++) {
                    c.set(i, j, 0.0);

                    for (int k = 0; k < b.getNRows(); k++) {
                        c.incr(i, j, a.get(i, k) * b.get(k, j));
                    }
                }
            }

            mpi.MPI_Send(c.getValues(), startRow * c.getNCols(),
                    (endRow - startRow) * c.getNCols(), 0, myRank,
                    mpi.MPI_COMM_WORLD);
        }
    }
}

