# Speed settings and the extra engines, shared by fet (src.pro) and fet-cl (src-cl.pro).
# Release builds get -O3 and link-time optimisation with GCC/Clang (whole-program optimisation with MSVC).
#
# Optional:
#   qmake "CONFIG+=fet_fast"    turn off FET's internal assert() checks. Faster; the finished
#                               timetable is still checked against every 100% constraint.
#   qmake "CONFIG+=fet_native"  tune for this computer's CPU. The program may not start on older CPUs,
#                               so do not use it for builds you give to other people.
#   Profile-guided optimisation (GCC; about 15% faster, a seed gives the same timetable as without it):
#     1. qmake "CONFIG+=fet_pgo_generate", build, then generate a few timetables with the program
#     2. delete the object files (*.o, keep the *.gcda files), qmake "CONFIG+=fet_pgo_use", build again

*-g++*|*clang* {
	QMAKE_CXXFLAGS_RELEASE -= -O2
	QMAKE_CXXFLAGS_RELEASE += -O3 -flto
	QMAKE_LFLAGS_RELEASE += -O3 -flto
	# keep floating point rounding identical to a normal build, so the same seed gives the same timetable
	fet_native: QMAKE_CXXFLAGS_RELEASE += -march=native -ffp-contract=off
	*-g++* {
		fet_pgo_generate {
			QMAKE_CXXFLAGS_RELEASE += -fprofile-generate -fprofile-update=atomic
			QMAKE_LFLAGS_RELEASE += -fprofile-generate -fprofile-update=atomic
		}
		fet_pgo_use {
			QMAKE_CXXFLAGS_RELEASE += -fprofile-use -fprofile-correction
			QMAKE_LFLAGS_RELEASE += -fprofile-use -fprofile-correction
		}
	}
}
msvc {
	QMAKE_CXXFLAGS_RELEASE += /GL
	QMAKE_LFLAGS_RELEASE += /LTCG
}

fet_fast: DEFINES += FET_NO_ASSERTS

# Several searches at once (threads) and the GPU engine (OpenCL, loaded at run time)
SOURCES += engine/gpusolver.cpp engine/parallelgenerate.cpp
HEADERS += engine/gpusolver.h engine/parallelgenerate.h
RESOURCES += engine/gpusolver.qrc
