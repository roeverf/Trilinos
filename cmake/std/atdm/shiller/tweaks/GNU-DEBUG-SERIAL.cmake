# Disable test that times out after 10 minutes (#2455)
ATDM_SET_ENABLE(Anasazi_Epetra_BlockDavidson_auxtest_MPI_4_DISABLE ON)

# Disable test takes a long time to complete for some reason (#2455)
ATDM_SET_ENABLE(Anasazi_Epetra_LOBPCG_auxtest_MPI_4_DISABLE ON)
