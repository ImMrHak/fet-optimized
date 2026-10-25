/*
File gpusolver.h - GPU (OpenCL) timetable engine.

The GPU engine is an alternative to the CPU engine in generate.cpp. It runs many independent
local searches on the graphics card and returns the first timetable that respects every 100%
constraint, as checked by FET's own constraint code. It does not reproduce the CPU engine's
timetable for a given seed; it finds another valid one.

OpenCL is loaded at run time, so FET works normally on computers without it.
*/

/***************************************************************************
 *                                                                         *
 *   This program is free software: you can redistribute it and/or modify  *
 *   it under the terms of the GNU Affero General Public License as        *
 *   published by the Free Software Foundation, either version 3 of the    *
 *   License, or (at your option) any later version.                       *
 *                                                                         *
 ***************************************************************************/

#ifndef GPUSOLVER_H
#define GPUSOLVER_H

#include <QString>
#include <QStringList>
#include <QVector>

#include <atomic>

class Solution;

namespace GpuSolver
{
	enum Result{
		Solved,          //sol holds a valid timetable
		TimeExceeded,
		Aborted,
		Impossible,      //some activity has no allowed slot or room at all
		NotAvailable,    //no OpenCL device, or the kernel could not be built
		Unsupported      //the data uses constraints the GPU engine does not handle
	};

	//Names of the usable OpenCL devices, best first. Empty if none (error says why).
	QStringList devices(QString* error=nullptr);

	//Empty if the GPU engine can handle the loaded data, otherwise the reason it cannot.
	//It handles official mode only, and only the constraint types it models (others are refused).
	//Call after Generate::precompute().
	QString unsupportedReason();

	//Search for a timetable. Call after Generate::precompute(). device indexes devices().
	//chains=0 picks a number of parallel searches that suits the device.
	//abort may be NULL; set *abort to true from another thread to stop.
	//message receives a short human-readable explanation of the result.
	//start (may be nullptr) is a partial timetable most searches begin from: nA times (-1 = not placed) then nA rooms
	//(-1 = no room, -2 = not chosen); values the GPU model does not allow are replaced by random ones.
	Result solve(int maxSeconds, int device, int chains, unsigned int seed, Solution& sol, QString& message, std::atomic<bool>* abort=nullptr,
	 const QVector<int>* start=nullptr);
}

#endif
