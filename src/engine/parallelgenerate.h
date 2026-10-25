/*
File parallelgenerate.h - run several timetable searches at once and keep the first valid result.

Each CPU search is a separate Generate object with its own random seed, on its own thread (as in
"generate multiple timetables"). The GPU engine can run as one more search at the same time.
*/

/***************************************************************************
 *                                                                         *
 *   This program is free software: you can redistribute it and/or modify  *
 *   it under the terms of the GNU Affero General Public License as        *
 *   published by the Free Software Foundation, either version 3 of the    *
 *   License, or (at your option) any later version.                       *
 *                                                                         *
 ***************************************************************************/

#ifndef PARALLELGENERATE_H
#define PARALLELGENERATE_H

#include <QString>

#include "solution.h"

#include <atomic>
#include <mutex>

class MRG32k3a;

namespace ParallelGenerate
{
	enum Outcome{
		Success,
		Impossible,
		TimeExceeded,
		Aborted,
		Failed
	};

	struct Options{
		int cpuSearches=1;    //number of CPU searches (0 = none)
		bool gpu=false;       //also run the GPU engine
		int gpuDevice=0;
		int gpuChains=0;      //0 = automatic
		int maxSeconds=2000000000;
	};

	//what the CPU searches have reached so far, updated about every 20 ms while run() works
	struct Progress{
		std::atomic<int> placed{0};           //activities placed now, in the furthest CPU search
		std::atomic<int> maxPlaced{0};        //most activities placed so far by one CPU search
		std::atomic<int> maxPlacedSeconds{0}; //when that happened
		std::mutex highestMutex;
		Solution highest;                     //the partial timetable with maxPlaced activities (valid if maxPlaced>0)
	};

	//one CPU search per physical processor core
	int defaultCpuSearches();

	//Generate::precompute() must have succeeded on the current data. The searches get seeds derived
	//from baseRng. On Success, sol holds the first valid timetable; message says which search found it
	//(with its random seed, so the classic engine can reproduce it). Set *abort to stop (may be nullptr).
	//If progress is given, it must have been reset (maxPlaced=0) and is kept up to date, also after stopping.
	Outcome run(const Options& options, const MRG32k3a& baseRng, Solution& sol, QString& message, std::atomic<bool>* abort=nullptr,
	 Progress* progress=nullptr);
}

#endif
