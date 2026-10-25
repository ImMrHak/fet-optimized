/*
File parallelgenerate.cpp - see parallelgenerate.h
*/

/***************************************************************************
 *                                                                         *
 *   This program is free software: you can redistribute it and/or modify  *
 *   it under the terms of the GNU Affero General Public License as        *
 *   published by the Free Software Foundation, either version 3 of the    *
 *   License, or (at your option) any later version.                       *
 *                                                                         *
 ***************************************************************************/

#include "parallelgenerate.h"

#include "timetable_defs.h"
#include "timetable.h"
#include "rules.h"
#include "solution.h"
#include "generate.h"
#include "gpusolver.h"

#include <QThread>
#include <QElapsedTimer>
#include <QByteArray>
#include <QVector>
#include <QFile>
#include <QSet>

#if defined(Q_OS_WIN)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(Q_OS_MACOS)
#include <sys/sysctl.h>
#endif

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

extern Timetable gt;

//One search per physical core: in benchmarks, two searches sharing a core (hyper-threading) found fewer timetables.
int ParallelGenerate::defaultCpuSearches()
{
	int cores=0;
#if defined(Q_OS_WIN)
	DWORD len=0;
	GetLogicalProcessorInformation(nullptr, &len);
	QVector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> info(int(len/sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)));
	if(!info.isEmpty() && GetLogicalProcessorInformation(info.data(), &len)){
		for(int i=0; i<int(len/sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)); i++)
			if(info[i].Relationship==RelationProcessorCore)
				cores++;
	}
#elif defined(Q_OS_LINUX)
	//each core lists its hardware threads once, as the same thread_siblings_list
	QSet<QByteArray> siblings;
	for(int cpu=0; cpu<QThread::idealThreadCount(); cpu++){
		QFile f(QString("/sys/devices/system/cpu/cpu%1/topology/thread_siblings_list").arg(cpu));
		if(f.open(QIODevice::ReadOnly))
			siblings.insert(f.readAll().trimmed());
	}
	cores=siblings.count();
#elif defined(Q_OS_MACOS)
	int n=0;
	size_t size=sizeof(n);
	if(sysctlbyname("hw.physicalcpu", &n, &size, nullptr, 0)==0)
		cores=n;
#endif
	if(cores<=0 || cores>QThread::idealThreadCount())
		cores=QThread::idealThreadCount();
	return qMax(1, cores);
}

namespace
{
enum State{ Running=0, Succeeded, WasImpossible, Stopped };

struct Search
{
	std::unique_ptr<Generate> gen;
	qint64 seeds[6];
	std::atomic<int> state{Running};
	std::thread thread;
};
}

ParallelGenerate::Outcome ParallelGenerate::run(const Options& options, const MRG32k3a& baseRng, Solution& sol, QString& message, std::atomic<bool>* abort, Progress* progress)
{
	QElapsedTimer clock;
	clock.start();

	//CPU searches, each with its own seed drawn from baseRng
	MRG32k3a seedSource=baseRng;
	std::vector<std::unique_ptr<Search>> searches;
	for(int t=0; t<options.cpuSearches; t++){
		std::unique_ptr<Search> s(new Search);
		for(int k=0; k<6; k++){
			qint64 m=(k<3) ? MRG32k3a::m1 : MRG32k3a::m2;
			s->seeds[k]=1+qint64(seedSource.uiMRG32k3a())%(m-1); //never all zero
		}
		s->gen.reset(new Generate());
		s->gen->nThread=t;
		s->gen->rng.initializeMRG32k3a(s->seeds[0], s->seeds[1], s->seeds[2], s->seeds[3], s->seeds[4], s->seeds[5]);
		s->gen->abortOptimization=false;
		s->gen->restart=false;
		s->gen->paused=false;
		s->gen->pausedTime=0;
		//read by updateProgress before the search sets them
		s->gen->maxActivitiesPlaced=0;
		s->gen->nPlacedActivities=0;
		s->gen->timeToHighestStage=0;
		searches.push_back(std::move(s));
	}
	for(std::unique_ptr<Search>& sp : searches){
		Search* s=sp.get();
		s->thread=std::thread([s, &options](){
			bool restarted, impossible, timeExceeded;
			//not threaded: the search does not wait for a window to show its progress
			s->gen->generate(options.maxSeconds, restarted, impossible, timeExceeded, false, false);
			if(impossible)
				s->state=WasImpossible;
			else if(timeExceeded || restarted || s->gen->abortOptimization)
				s->state=Stopped;
			else
				s->state=Succeeded;
		});
	}

	//GPU search
	Solution gpuSol;
	QString gpuMessage;
	std::atomic<bool> gpuAbort{false};
	std::atomic<int> gpuState{Running};
	std::atomic<int> gpuResult{-1};
	std::thread gpuThread;
	QVector<int> gpuStart; //partial timetable from the best CPU search, for the GPU to finish
	const unsigned int gpuSeed=options.gpu ? seedSource.uiMRG32k3a() : 0;
	auto startGpu=[&](int seconds, bool fromCpu){
		gpuThread=std::thread([&, seconds, fromCpu](){
			GpuSolver::Result r=GpuSolver::solve(seconds, options.gpuDevice, options.gpuChains, gpuSeed, gpuSol, gpuMessage, &gpuAbort,
			 fromCpu ? &gpuStart : nullptr);
			gpuResult=int(r);
			gpuState=(r==GpuSolver::Solved) ? Succeeded : ((r==GpuSolver::Impossible) ? WasImpossible : Stopped);
		});
	};
	//With CPU searches, the GPU waits gpuDelayMs (most data is solved by then) and then starts from the furthest CPU
	//partial timetable: the CPU is good at placing most activities, the GPU at repairing the last clashes.
	const qint64 gpuDelayMs=5000;
	bool gpuStarted=false;
	if(options.gpu && searches.empty()){
		startGpu(options.maxSeconds, false);
		gpuStarted=true;
	}

	//copy the furthest partial timetable (Generate changes it only under myMutex when not threaded)
	auto updateProgress=[&](){
		if(progress==nullptr)
			return;
		int placed=0;
		for(std::unique_ptr<Search>& s : searches){
			placed=qMax(placed, s->gen->nPlacedActivities); //shortcut: unlocked read of an int, only for display
			std::lock_guard<std::mutex> lock(s->gen->myMutex);
			if(s->gen->maxActivitiesPlaced>progress->maxPlaced){
				std::lock_guard<std::mutex> lockHighest(progress->highestMutex);
				progress->highest.copy(gt.rules, s->gen->highestStageSolution);
				progress->maxPlaced=s->gen->maxActivitiesPlaced;
				progress->maxPlacedSeconds=s->gen->timeToHighestStage;
			}
		}
		progress->placed=placed;
	};

	Outcome outcome=TimeExceeded;
	int winner=-1; //CPU search index, or -2 for the GPU
	for(;;){
		if(abort!=nullptr && *abort){
			outcome=Aborted;
			break;
		}
		updateProgress();
		bool running=false;
		const qint64 nowMs=clock.elapsed();
		for(int t=0; t<int(searches.size()); t++){
			int st=searches[t]->state;
			if(st==Running)
				running=true;
			else if(st==Succeeded && winner==-1)
				winner=t;
			else if(st==WasImpossible && outcome!=Impossible){
				//the CPU engine finds an activity impossible to place whatever the seed
				outcome=Impossible;
				message=QString("impossible to solve (found by CPU search %1)").arg(t+1);
			}
		}
		if(options.gpu && !gpuStarted && nowMs>=gpuDelayMs){
			int best=-1;
			int bestPlaced=0;
			for(int t=0; t<int(searches.size()); t++){
				std::lock_guard<std::mutex> lock(searches[t]->gen->myMutex);
				if(searches[t]->state==Running && searches[t]->gen->maxActivitiesPlaced>bestPlaced){
					best=t;
					bestPlaced=searches[t]->gen->maxActivitiesPlaced;
				}
			}
			int remaining=options.maxSeconds-int(nowMs/1000);
			if(remaining>1){
				const int nA=gt.rules.nInternalActivities;
				if(best>=0){
					std::lock_guard<std::mutex> lock(searches[best]->gen->myMutex);
					Solution& h=searches[best]->gen->highestStageSolution;
					gpuStart.resize(2*nA);
					for(int a=0; a<nA; a++){
						int tm=h.times[a], rm=h.rooms[a];
						gpuStart[a]=(tm>=0 && tm<gt.rules.nHoursPerWeek) ? tm : -1;
						gpuStart[nA+a]=(tm<0 || tm>=gt.rules.nHoursPerWeek || rm==UNALLOCATED_SPACE) ? -2 : ((rm==UNSPECIFIED_ROOM) ? -1 : rm);
					}
				}
				startGpu(remaining, best>=0);
			}
			else
				gpuState=Stopped;
			gpuStarted=true;
		}
		if(options.gpu){
			if(gpuState==Running)
				running=true;
			else if(gpuState==Succeeded && winner==-1)
				winner=-2;
			else if(gpuState==WasImpossible && searches.empty()){
				outcome=Impossible;
				message=QString("impossible to solve: %1").arg(gpuMessage);
			}
		}
		if(winner!=-1){
			outcome=Success;
			break;
		}
		if(outcome==Impossible || !running)
			break;
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
	}

	//stop and wait for every search
	for(std::unique_ptr<Search>& s : searches)
		s->gen->abortOptimization=true;
	gpuAbort=true;
	for(std::unique_ptr<Search>& s : searches)
		s->thread.join();
	if(gpuThread.joinable())
		gpuThread.join();
	updateProgress();

	double seconds=clock.elapsed()/1000.0;
	if(outcome==Success){
		if(winner>=0){
			const qint64* sd=searches[winner]->seeds;
			sol.copy(gt.rules, searches[winner]->gen->c);
			message=QString("found by CPU search %1 of %2 in %3 s (random seeds %4 %5 %6 %7 %8 %9)")
			 .arg(winner+1).arg(searches.size()).arg(seconds, 0, 'f', 1)
			 .arg(sd[0]).arg(sd[1]).arg(sd[2]).arg(sd[3]).arg(sd[4]).arg(sd[5]);
		}
		else{
			sol.copy(gt.rules, gpuSol);
			message=QString("found by the GPU engine in %1 s (%2)").arg(seconds, 0, 'f', 1).arg(gpuMessage);
		}
	}
	else if(outcome==TimeExceeded){
		message=QString("time exceeded: none of the %1 searches found a timetable").arg(searches.size()+(options.gpu ? 1 : 0));
		if(options.gpu && !gpuMessage.isEmpty())
			message+=QString("\nGPU engine: %1").arg(gpuMessage);
		//GPU only, and the GPU engine could not be used at all
		if(searches.empty() && (gpuResult==int(GpuSolver::NotAvailable) || gpuResult==int(GpuSolver::Unsupported)))
			outcome=Failed;
	}
	else if(outcome==Aborted)
		message=QString("stopped");
	return outcome;
}
