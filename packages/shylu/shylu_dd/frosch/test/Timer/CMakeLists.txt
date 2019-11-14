TRIBITS_ADD_EXECUTABLE(
timer_test
SOURCES main.cpp
)


MESSAGE(STATUS "FROSch tests running with Tpetra_DefaultNode type ${Tpetra_DefaultNode}")
TRIBITS_ADD_TEST(
timer_test
NAME timer_test
COMM mpi
NUM_MPI_PROCS 4
)
