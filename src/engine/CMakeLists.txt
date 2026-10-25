cmake_minimum_required(VERSION 3.28)

find_package(Qt6 6.10 REQUIRED COMPONENTS Core)

qt_add_library(fetengine STATIC
    activity.cpp activity.h
    activitytag.cpp activitytag.h
    building.cpp building.h
    generate.cpp generate.h
    generate_pre.cpp generate_pre.h
    gpusolver.cpp gpusolver.h
    parallelgenerate.cpp parallelgenerate.h
    lockunlock.cpp lockunlock.h
    matrix.h
    messageboxes.h
    room.cpp room.h
    rules.cpp rules.h
    solution.cpp solution.h
    spaceconstraint.cpp spaceconstraint.h
    studentsset.cpp studentsset.h
    subject.cpp subject.h
    teacher.cpp teacher.h
    timeconstraint.cpp timeconstraint.h
    timetable.cpp timetable.h
    timetable_defs.cpp timetable_defs.h
    timetableexport.cpp timetableexport.h
)

# the OpenCL source of the GPU engine, read at run time as :/engine/gpusolver_kernel.cl
qt_add_resources(fetengine "gpusolverkernel"
    PREFIX "/engine"
    FILES gpusolver_kernel.cl
)

target_include_directories(fetengine PUBLIC
    .
)

target_link_libraries(fetengine PRIVATE
    Qt::Core
)
