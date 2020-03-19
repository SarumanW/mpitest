package edu.coursera.distributed;

public final class Matrix {

    private final double[] values;

    private final int nrows;

    private final int ncols;

    public Matrix(final int setNrows, final int setNcols) {
        this.nrows = setNrows;
        this.ncols = setNcols;
        this.values = new double[nrows * ncols];
    }

    public Matrix(final Matrix other) {
        this.nrows = other.nrows;
        this.ncols = other.ncols;
        this.values = new double[nrows * ncols];
        System.arraycopy(other.values, 0, values, 0, values.length);
    }

    public void set(final int row, final int col, final double val) {
        values[row * ncols + col] = val;
    }

    public void incr(final int row, final int col, final double val) {
        values[row * ncols + col] += val;
    }

    public double get(final int row, final int col) {
        return values[row * ncols + col];
    }

    public int getNRows() {
        return nrows;
    }

    public int getNCols() {
        return ncols;
    }

    public double[] getValues() {
        return values;
    }
}
