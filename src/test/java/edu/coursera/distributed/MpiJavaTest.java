package edu.coursera.distributed;

import edu.coursera.distributed.util.MPI;
import edu.coursera.distributed.util.MPI.MPIException;
import junit.extensions.TestSetup;
import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;
import org.junit.FixMethodOrder;
import org.junit.runners.MethodSorters;

import java.util.Random;

@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public class MpiJavaTest extends TestCase {

    private Matrix createRandomMatrix(final int rows, final int cols) {
        Matrix matrix = new Matrix(rows, cols);
        final Random rand = new Random(314);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix.set(i, j, rand.nextInt(100));
            }
        }

        return matrix;
    }

    private Matrix copyMatrix(Matrix input) {
        return new Matrix(input);
    }

    private void seqMatrixMultiply(Matrix a, Matrix b, Matrix c) {
        for (int i = 0; i < c.getNRows(); i++) {
            for (int j = 0; j < c.getNCols(); j++) {
                c.set(i, j, 0.0);

                for (int k = 0; k < b.getNRows(); k++) {
                    c.incr(i, j, a.get(i, k) * b.get(k, j));
                }
            }
        }
    }

    private static MPI mpi = null;

    public static Test suite() {
        TestSetup setup = new TestSetup(new TestSuite(MpiJavaTest.class)) {
            protected void setUp() throws Exception {
                assert (mpi == null);
                mpi = new MPI();
                mpi.MPI_Init();
            }

            protected void tearDown() throws Exception {
                assert (mpi != null);
                mpi.MPI_Finalize();
            }
        };
        return setup;
    }

    private void testDriver(final int M, final int N, final int P)
            throws MPIException {
        final int myrank = mpi.MPI_Comm_rank(mpi.MPI_COMM_WORLD);

        Matrix a, b, c;
        if (myrank == 0) {
            a = createRandomMatrix(M, N);
            b = createRandomMatrix(N, P);
            c = createRandomMatrix(M, P);
        } else {
            a = new Matrix(M, N);
            b = new Matrix(N, P);
            c = new Matrix(M, P);
        }

        Matrix copy_a = copyMatrix(a);
        Matrix copy_b = copyMatrix(b);
        Matrix copy_c = copyMatrix(c);

        if (myrank == 0) {
            System.err.println("Testing matrix multiply: [" + M + " x " + N +
                    "] * [" + N + " x " + P + "] = [" + M + " x " + P + "]");
        }

        final long seqStart = System.currentTimeMillis();
        seqMatrixMultiply(copy_a, copy_b, copy_c);
        final long seqElapsed = System.currentTimeMillis() - seqStart;

        if (myrank == 0) {
            System.err.println("Sequential implementation ran in " +
                    seqElapsed + " ms");
        }

        mpi.MPI_Barrier(mpi.MPI_COMM_WORLD);

        final long parallelStart = System.currentTimeMillis();
        NonBlockingMatrixMult.parallelMatrixMultiply(a, b, c, mpi);
        final long parallelElapsed = System.currentTimeMillis() - parallelStart;

        final long parallel2Start = System.currentTimeMillis();
        MatrixMult.parallelMatrixMultiply(a, b, c, mpi);
        final long parallel2Elapsed = System.currentTimeMillis() - parallel2Start;


        if (myrank == 0) {
            final double speedup = (double)seqElapsed / (double)parallelElapsed;
            System.err.println("Non-blocking MPI implementation ran in " + parallelElapsed +
                    " ms, yielding a speedup of " + speedup + "x");
            System.err.println();

            final double speedup2 = (double)seqElapsed / (double)parallel2Elapsed;
            System.err.println("Blocking MPI implementation ran in " + parallel2Elapsed +
                    " ms, yielding a speedup of " + speedup2 + "x");
            System.err.println();
        }
    }

    public void testMatrixMultiplySquareSmall() throws MPIException {
        testDriver(800, 800, 800);
    }

    public void testMatrixMultiplySquareLarge() throws MPIException {
        testDriver(1200, 1200, 1200);
    }

    public void testMatrixMultiplyRectangular1Small() throws MPIException {
        testDriver(800, 1600, 500);
    }

    public void testMatrixMultiplyRectangularLarge() throws MPIException {
        testDriver(1800, 1400, 1000);
    }

    public void testMatrixMultiply2000() throws MPIException {
        testDriver(2000, 2000, 2000);
    }

    public void testMatrixMultiply3000() throws MPIException {
        testDriver(3000, 3000, 3000);
    }
}
