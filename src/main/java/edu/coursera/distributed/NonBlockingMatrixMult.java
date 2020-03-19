package edu.coursera.distributed;

import edu.coursera.distributed.util.MPI;
import edu.coursera.distributed.util.MPI.MPIException;

import java.util.ArrayList;
import java.util.List;

class NonBlockingMatrixMult {

    static void parallelMatrixMultiply(Matrix a, Matrix b, Matrix c,
                                       final MPI mpi) throws MPIException {


        //identify rank of current process
        final int myRank = mpi.MPI_Comm_rank(mpi.MPI_COMM_WORLD);
        //number of ranks we are working with
        final int size = mpi.MPI_Comm_size(mpi.MPI_COMM_WORLD);


        int averow, dest, extra, source;
        double[] offset = new double[]{0};
        double[] rows = new double[]{0};

        if (myRank == 0) {
            averow = a.getNRows() / (size - 1);
            extra = a.getNRows() % (size - 1);
            offset[0] = 0;

            List<MPI.MPI_Request> sendRequests = new ArrayList<>();

            for (dest = 1; dest <= size - 1; dest++) {
                rows[0] = (dest <= extra) ? averow + 1 : averow;

                sendRequests.add(mpi.MPI_Isend(offset, 0, 1, dest, 1, mpi.MPI_COMM_WORLD));
                sendRequests.add(mpi.MPI_Isend(rows, 0, 1, dest, 1, mpi.MPI_COMM_WORLD));
                sendRequests.add(mpi.MPI_Isend(a.getValues(), 0, (int) rows[0] * a.getNCols(), dest, 1, mpi.MPI_COMM_WORLD));
                sendRequests.add(mpi.MPI_Isend(b.getValues(), 0, b.getNRows() * b.getNCols(), dest,
                        1, mpi.MPI_COMM_WORLD));

                offset[0] = (int) offset[0] + (int) rows[0];
            }

            MPI.MPI_Request[] sendRequestsArray = new MPI.MPI_Request[sendRequests.size()];
            sendRequests.toArray(sendRequestsArray);
            mpi.MPI_Waitall(sendRequestsArray);

            MPI.MPI_Request[] recvRequests = new MPI.MPI_Request[size - 1];

            for (source = 1; source <= size - 1; source++) {
                recvRequests[source - 1] = mpi.MPI_Irecv(c.getValues(), 0, c.getNCols() * c.getNRows(),
                        source, 2, mpi.MPI_COMM_WORLD);
            }

            mpi.MPI_Waitall(recvRequests);

            //System.out.println("final");
        } else {
            MPI.MPI_Request[] requests = new MPI.MPI_Request[4];

            requests[0] = mpi.MPI_Irecv(offset, 0, 1, 0, 1, mpi.MPI_COMM_WORLD);
            requests[1] = mpi.MPI_Irecv(rows, 0, 1, 0, 1, mpi.MPI_COMM_WORLD);
            requests[2] = mpi.MPI_Irecv(a.getValues(), 0, (int) rows[0] * a.getNCols(), 0, 1, mpi.MPI_COMM_WORLD);
            requests[3] = mpi.MPI_Irecv(b.getValues(), 0, b.getNRows() * b.getNCols(), 0,
                    1, mpi.MPI_COMM_WORLD);

            mpi.MPI_Waitall(requests);

            for (int i = 0; i < b.getNCols(); i++) {
                for (int j = 0; j < (int) rows[0]; j++) {
                    c.set(i, j, 0.0);

                    for (int k = 0; k < a.getNCols(); k++) {
                        c.incr(i, j, a.get(i, k) * b.get(k, j));
                    }
                }
            }

            MPI.MPI_Request mpi_request = mpi.MPI_Isend(c.getValues(), 0, c.getNCols() * c.getNRows(), 0, 2, mpi.MPI_COMM_WORLD);
            mpi.MPI_Wait(mpi_request);
        }
    }
}

