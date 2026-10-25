/*
File gpusolver.cpp - GPU (OpenCL) timetable engine. See gpusolver.h and gpusolver_kernel.cl.
*/

/***************************************************************************
 *                                                                         *
 *   This program is free software: you can redistribute it and/or modify  *
 *   it under the terms of the GNU Affero General Public License as        *
 *   published by the Free Software Foundation, either version 3 of the    *
 *   License, or (at your option) any later version.                       *
 *                                                                         *
 ***************************************************************************/

#include "gpusolver.h"

#include "timetable_defs.h"
#include "timetable.h"
#include "rules.h"
#include "solution.h"
#include "generate_pre.h"
#include "timeconstraint.h"
#include "spaceconstraint.h"
#include "room.h"

#include <QSet>
#include <mutex>

#include <QLibrary>
#include <QFile>
#include <QElapsedTimer>
#include <QVector>
#include <QSet>
#include <QHash>
#include <QMap>

#include <algorithm>
#include <utility>
#include <cstdio>

extern Timetable gt;

////////////////////////////////////////////////////////////////////////
//Minimal OpenCL 1.2 declarations; the library is loaded at run time.

#ifdef _WIN32
#define FET_CL_API __stdcall
#else
#define FET_CL_API
#endif

typedef int cl_int;
typedef unsigned int cl_uint;
typedef unsigned long long cl_ulong;
typedef cl_ulong cl_bitfield;
typedef struct _cl_platform_id* cl_platform_id;
typedef struct _cl_device_id* cl_device_id;
typedef struct _cl_context* cl_context;
typedef struct _cl_command_queue* cl_command_queue;
typedef struct _cl_mem* cl_mem;
typedef struct _cl_program* cl_program;
typedef struct _cl_kernel* cl_kernel;
typedef struct _cl_event* cl_event;

static const cl_int CL_SUCCESS_=0;
static const cl_bitfield CL_DEVICE_TYPE_GPU_=1<<2;
static const cl_bitfield CL_DEVICE_TYPE_ALL_=0xFFFFFFFF;
static const cl_uint CL_PLATFORM_NAME_=0x0902;
static const cl_uint CL_DEVICE_TYPE_INFO_=0x1000;
static const cl_uint CL_DEVICE_MAX_COMPUTE_UNITS_=0x1002;
static const cl_uint CL_DEVICE_MAX_WORK_GROUP_SIZE_=0x1004;
static const cl_uint CL_DEVICE_MAX_MEM_ALLOC_SIZE_=0x1010;
static const cl_uint CL_DEVICE_GLOBAL_MEM_SIZE_=0x101F;
static const cl_uint CL_DEVICE_LOCAL_MEM_SIZE_=0x1023;
static const cl_uint CL_DEVICE_NAME_=0x102B;
static const cl_uint CL_PROGRAM_BUILD_LOG_=0x1183;
static const cl_bitfield CL_MEM_READ_WRITE_=1<<0;
static const cl_bitfield CL_MEM_READ_ONLY_=1<<2;
static const cl_bitfield CL_MEM_COPY_HOST_PTR_=1<<5;

typedef cl_int (FET_CL_API *PclGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (FET_CL_API *PclGetPlatformInfo)(cl_platform_id, cl_uint, size_t, void*, size_t*);
typedef cl_int (FET_CL_API *PclGetDeviceIDs)(cl_platform_id, cl_bitfield, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (FET_CL_API *PclGetDeviceInfo)(cl_device_id, cl_uint, size_t, void*, size_t*);
typedef cl_context (FET_CL_API *PclCreateContext)(const intptr_t*, cl_uint, const cl_device_id*, void (FET_CL_API *)(const char*, const void*, size_t, void*), void*, cl_int*);
typedef cl_command_queue (FET_CL_API *PclCreateCommandQueue)(cl_context, cl_device_id, cl_bitfield, cl_int*);
typedef cl_program (FET_CL_API *PclCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int (FET_CL_API *PclBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void (FET_CL_API *)(cl_program, void*), void*);
typedef cl_int (FET_CL_API *PclGetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void*, size_t*);
typedef cl_kernel (FET_CL_API *PclCreateKernel)(cl_program, const char*, cl_int*);
typedef cl_mem (FET_CL_API *PclCreateBuffer)(cl_context, cl_bitfield, size_t, void*, cl_int*);
typedef cl_int (FET_CL_API *PclSetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_int (FET_CL_API *PclEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (FET_CL_API *PclEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (FET_CL_API *PclFinish)(cl_command_queue);
typedef cl_int (FET_CL_API *PclRelease)(void*);

namespace
{
struct OpenCL
{
	bool loaded=false;
	QString error;
	QLibrary lib;
	PclGetPlatformIDs getPlatformIDs=nullptr;
	PclGetPlatformInfo getPlatformInfo=nullptr;
	PclGetDeviceIDs getDeviceIDs=nullptr;
	PclGetDeviceInfo getDeviceInfo=nullptr;
	PclCreateContext createContext=nullptr;
	PclCreateCommandQueue createCommandQueue=nullptr;
	PclCreateProgramWithSource createProgramWithSource=nullptr;
	PclBuildProgram buildProgram=nullptr;
	PclGetProgramBuildInfo getProgramBuildInfo=nullptr;
	PclCreateKernel createKernel=nullptr;
	PclCreateBuffer createBuffer=nullptr;
	PclSetKernelArg setKernelArg=nullptr;
	PclEnqueueNDRangeKernel enqueueNDRangeKernel=nullptr;
	PclEnqueueReadBuffer enqueueReadBuffer=nullptr;
	PclFinish finish=nullptr;
	PclRelease releaseMemObject=nullptr;
	PclRelease releaseKernel=nullptr;
	PclRelease releaseProgram=nullptr;
	PclRelease releaseCommandQueue=nullptr;
	PclRelease releaseContext=nullptr;

	bool load()
	{
		if(loaded)
			return true;
#if defined(Q_OS_MAC)
		lib.setFileName("/System/Library/Frameworks/OpenCL.framework/OpenCL");
#else
		lib.setFileNameAndVersion("OpenCL", 1);
		if(!lib.load())
			lib.setFileName("OpenCL");
#endif
		if(!lib.isLoaded() && !lib.load()){
			error=QString("The OpenCL library is not installed (no graphics card driver with OpenCL support was found)");
			return false;
		}
#define FET_CL_RESOLVE(var, name) var=(decltype(var))lib.resolve(name); if(var==nullptr){ error=QString("OpenCL function %1 is missing").arg(name); return false; }
		FET_CL_RESOLVE(getPlatformIDs, "clGetPlatformIDs")
		FET_CL_RESOLVE(getPlatformInfo, "clGetPlatformInfo")
		FET_CL_RESOLVE(getDeviceIDs, "clGetDeviceIDs")
		FET_CL_RESOLVE(getDeviceInfo, "clGetDeviceInfo")
		FET_CL_RESOLVE(createContext, "clCreateContext")
		FET_CL_RESOLVE(createCommandQueue, "clCreateCommandQueue")
		FET_CL_RESOLVE(createProgramWithSource, "clCreateProgramWithSource")
		FET_CL_RESOLVE(buildProgram, "clBuildProgram")
		FET_CL_RESOLVE(getProgramBuildInfo, "clGetProgramBuildInfo")
		FET_CL_RESOLVE(createKernel, "clCreateKernel")
		FET_CL_RESOLVE(createBuffer, "clCreateBuffer")
		FET_CL_RESOLVE(setKernelArg, "clSetKernelArg")
		FET_CL_RESOLVE(enqueueNDRangeKernel, "clEnqueueNDRangeKernel")
		FET_CL_RESOLVE(enqueueReadBuffer, "clEnqueueReadBuffer")
		FET_CL_RESOLVE(finish, "clFinish")
		FET_CL_RESOLVE(releaseMemObject, "clReleaseMemObject")
		FET_CL_RESOLVE(releaseKernel, "clReleaseKernel")
		FET_CL_RESOLVE(releaseProgram, "clReleaseProgram")
		FET_CL_RESOLVE(releaseCommandQueue, "clReleaseCommandQueue")
		FET_CL_RESOLVE(releaseContext, "clReleaseContext")
#undef FET_CL_RESOLVE
		loaded=true;
		return true;
	}
};

OpenCL& cl()
{
	static OpenCL instance;
	return instance;
}

struct Device
{
	cl_device_id id;
	QString name;
	bool isGpu;
	cl_uint computeUnits;
};

//GPUs first, then other devices (e.g. CPU OpenCL drivers)
QList<Device> listDevices(QString* error)
{
	QList<Device> result;
	OpenCL& c=cl();
	if(!c.load()){
		if(error!=nullptr) *error=c.error;
		return result;
	}
	cl_uint np=0;
	if(c.getPlatformIDs(0, nullptr, &np)!=CL_SUCCESS_ || np==0){
		if(error!=nullptr) *error=QString("No OpenCL platform found");
		return result;
	}
	QVector<cl_platform_id> platforms(np);
	c.getPlatformIDs(np, platforms.data(), nullptr);
	QList<Device> others;
	for(cl_platform_id pl : std::as_const(platforms)){
		cl_uint nd=0;
		if(c.getDeviceIDs(pl, CL_DEVICE_TYPE_ALL_, 0, nullptr, &nd)!=CL_SUCCESS_ || nd==0)
			continue;
		QVector<cl_device_id> ids(nd);
		c.getDeviceIDs(pl, CL_DEVICE_TYPE_ALL_, nd, ids.data(), nullptr);
		for(cl_device_id id : std::as_const(ids)){
			Device d;
			d.id=id;
			char name[256]={0};
			c.getDeviceInfo(id, CL_DEVICE_NAME_, sizeof(name)-1, name, nullptr);
			d.name=QString::fromLatin1(name).trimmed();
			cl_bitfield type=0;
			c.getDeviceInfo(id, CL_DEVICE_TYPE_INFO_, sizeof(type), &type, nullptr);
			d.isGpu=(type & CL_DEVICE_TYPE_GPU_)!=0;
			d.computeUnits=1;
			c.getDeviceInfo(id, CL_DEVICE_MAX_COMPUTE_UNITS_, sizeof(d.computeUnits), &d.computeUnits, nullptr);
			if(d.isGpu)
				result.append(d);
			else
				others.append(d);
		}
	}
	result+=others;
	if(result.isEmpty() && error!=nullptr)
		*error=QString("No OpenCL device found");
	return result;
}

//must match gpusolver_kernel.cl
const int NP=24;
enum{
	P_FLAGS=0, P_MAXHD1=1, P_MAXHD2=2, P_MAXHC1=3, P_MAXHC2=4, P_MINHD=5, P_MINHD_ALLOWEMPTY=6, P_MAXGW=7, P_MAXGD=8,
	P_MAXDAYS=9, P_MINDAYS=10, P_EARLY=11, P_INT1=12, P_INT2=15, P_INT3=18, P_MAXSPAN=21, P_RESTC=22, P_RESTN=23
};
enum{
	T_MINDAYS=1, T_MINDAYS_CONSEC=2, T_MAXDAYS=3, T_MINGAPS=4, T_SAMETIME=5, T_SAMEHOUR=6, T_SAMEDAY=7, T_NOTOVERLAP=8,
	T_CONSEC_AB=9, T_CONSEC_BA=10, T_GROUPED=11, T_ORDERED_AB=12, T_ORDERED_BA=13, T_ENDS_A=14, T_ENDS_B=15
};
const int CI_N=8, CI_ITER=0, CI_SOLVED=2, CI_BESTHARD=3;
const int CF_N=4, CF_HARD=0, CF_SOFT=1, CF_BESTFEASSOFT=2;
const int WG=64;

//Problem data in the flat layout the kernel expects
struct Problem
{
	int nA, nRes, nRooms, nD, nH, nS;
	QVector<int> actDur, actResStart, actRes, actRoomStart, actRooms;
	QVector<float> actRoomCost, actNoRoomCost, timeCost;
	QVector<int> pairStart, pairOther, pairType, pairParam;
	QVector<float> pairW;
	QVector<int> tripStart, tripB, tripC;
	QVector<float> tripW;
	QVector<int> resP;
	QVector<float> resW;
	QVector<cl_ulong> resAvail;
	QVector<float> roomTimeCost;
	QVector<int> nextNB;
	QVector<int> movable;
	//groups of activities that must start together (100% same starting time); a lone activity is its own group
	QVector<int> leader, memStart, mem, gResStart, gRes, gResMemStart, gResMem, grpMovable;
	int maxGroup;
};

void setParam(Problem& p, int r, int idx, int value, double weight)
{
	if(weight<0 || value<0)
		return;
	p.resP[r*NP+idx]=value;
	p.resW[r*NP+idx]=float(weight);
	p.resP[r*NP+P_FLAGS]=1;
}

void setInterval(Problem& p, int r, int idx, int maxDays, int start, int end, double weight)
{
	if(weight<0 || maxDays<0)
		return;
	setParam(p, r, idx, maxDays, weight);
	p.resP[r*NP+idx+1]=start;
	p.resP[r*NP+idx+2]=end;
}

//returns an empty string on success, else why the data cannot be solved
QString buildProblem(Problem& p)
{
	Rules& r=gt.rules;
	p.nA=r.nInternalActivities;
	p.nD=r.nDaysPerWeek;
	p.nH=r.nHoursPerDay;
	p.nS=r.nHoursPerWeek;
	const int nT=r.nInternalTeachers;
	const int nG=r.nInternalSubgroups;
	p.nRes=nT+nG;
	p.nRooms=r.nInternalRooms;
	const int A=p.nA, S=p.nS, D=p.nD, H=p.nH;

	if(H>64)
		return QString("more than 64 hours per day");
	if(A==0)
		return QString("no activities");

	//activities: duration and the teachers+subgroups they use
	p.actDur.resize(A);
	p.actResStart.resize(A+1);
	p.actResStart[0]=0;
	for(int a=0; a<A; a++){
		const Activity& act=r.internalActivitiesList[a];
		p.actDur[a]=act.duration;
		for(int t : std::as_const(act.iTeachersList))
			p.actRes.append(t);
		for(int g : std::as_const(act.iSubgroupsList))
			p.actRes.append(nT+g);
		p.actResStart[a+1]=p.actRes.count();
	}

	//allowed starting slots (negative = forbidden, else soft cost)
	p.timeCost.resize(A*S);
	for(int a=0; a<A; a++){
		bool any=false;
		for(int s=0; s<S; s++){
			double w=notAllowedTimesPercentages(a, s);
			int h=s/D;
			float c;
			if(h+p.actDur[a]>H || w>=100.0)
				c=-1.0f;
			else
				c=(w<0) ? 0.0f : float(w/100.0);
			p.timeCost[a*S+s]=c;
			if(c>=0)
				any=true;
		}
		if(!any)
			return QString("impossible: activity with id %1 has no allowed time slot").arg(r.internalActivitiesList[a].id);
	}

	//rooms
	p.roomTimeCost.resize(qMax(1, p.nRooms*S));
	for(int rm=0; rm<p.nRooms; rm++)
		for(int s=0; s<S; s++){
			double w=notAllowedRoomTimePercentages(rm, s);
			p.roomTimeCost[rm*S+s]=(w>=100.0) ? -1.0f : ((w<0) ? 0.0f : float(w/100.0));
		}
	p.actRoomStart.resize(A+1);
	p.actRoomStart[0]=0;
	p.actNoRoomCost.resize(A);
	for(int a=0; a<A; a++){
		QSet<int> cand;
		float noRoom=0.0f; //negative: an activity must get a room
		QList<QPair<QSet<int>, float> > softItems;
		bool hardPreferred=false;
		if(!unspecifiedPreferredRoom[a]){
			QSet<int> inter, uni;
			bool first=true;
			for(const PreferredRoomsItem& it : std::as_const(activitiesPreferredRoomsList[a])){
				if(it.percentage>=100.0){
					inter=first ? it.preferredRooms : (inter & it.preferredRooms);
					first=false;
					hardPreferred=true;
				}
				else{
					softItems.append(qMakePair(it.preferredRooms, float(it.percentage/100.0)));
					uni|=it.preferredRooms;
					noRoom+=float(it.percentage/100.0);
				}
			}
			if(hardPreferred){
				cand=inter;
				noRoom=-1.0f;
			}
			else
				cand=uni;
		}
		if(!unspecifiedHomeRoom[a]){
			QSet<int> home;
			for(int rm : std::as_const(activitiesHomeRoomsHomeRooms[a]))
				home.insert(rm);
			double hw=activitiesHomeRoomsPercentage[a];
			if(unspecifiedPreferredRoom[a]){
				cand=home;
				noRoom=(hw>=100.0) ? -1.0f : float(hw/100.0);
			}
			else{
				if(!hardPreferred){
					cand|=home;
					if(hw>=100.0)
						noRoom=-1.0f;
					else if(noRoom>=0)
						noRoom+=float(hw/100.0);
				}
				if(hw>=100.0 && !(cand & home).isEmpty())
					cand&=home;
				else if(hw<100.0)
					softItems.append(qMakePair(home, float(hw/100.0)));
			}
		}
		if(cand.isEmpty() && noRoom<0)
			return QString("impossible: activity with id %1 has no allowed room").arg(r.internalActivitiesList[a].id);
		QList<int> sorted=cand.values();
		std::sort(sorted.begin(), sorted.end());
		for(int rm : std::as_const(sorted)){
			float c=0.0f;
			for(const QPair<QSet<int>, float>& it : std::as_const(softItems))
				if(!it.first.contains(rm))
					c+=it.second;
			p.actRooms.append(rm);
			p.actRoomCost.append(c);
		}
		p.actRoomStart[a+1]=p.actRooms.count();
		p.actNoRoomCost[a]=noRoom;
	}

	//pair constraints
	p.pairStart.resize(A+1);
	p.pairStart[0]=0;
	QHash<int, QSet<int> > endsDayPartners; //activity ending the students day -> activities sharing students
	if(haveActivityEndsStudentsDay){
		QVector<QList<int> > actsOfSubgroup(nG);
		for(int a=0; a<A; a++)
			for(int g : std::as_const(r.internalActivitiesList[a].iSubgroupsList))
				actsOfSubgroup[g].append(a);
		for(int a=0; a<A; a++)
			if(activityEndsStudentsDayPercentages[a]>=0)
				for(int g : std::as_const(r.internalActivitiesList[a].iSubgroupsList))
					for(int b : std::as_const(actsOfSubgroup[g]))
						if(b!=a){
							endsDayPartners[a].insert(b);
							endsDayPartners[b].insert(a);
						}
	}
	for(int a=0; a<A; a++){
		auto add=[&](int b, int type, int param, double w){
			p.pairOther.append(b);
			p.pairType.append(type);
			p.pairParam.append(param);
			p.pairW.append(float(w));
		};
		for(int i=0; i<minDaysListOfActivities[a].count(); i++)
			add(minDaysListOfActivities[a].at(i), minDaysListOfConsecutiveIfSameDay[a].at(i) ? T_MINDAYS_CONSEC : T_MINDAYS,
			 minDaysListOfMinDays[a].at(i), minDaysListOfWeightPercentages[a].at(i));
		for(int i=0; i<maxDaysListOfActivities[a].count(); i++)
			add(maxDaysListOfActivities[a].at(i), T_MAXDAYS, maxDaysListOfMaxDays[a].at(i), maxDaysListOfWeightPercentages[a].at(i));
		for(int i=0; i<minGapsBetweenActivitiesListOfActivities[a].count(); i++)
			add(minGapsBetweenActivitiesListOfActivities[a].at(i), T_MINGAPS, minGapsBetweenActivitiesListOfMinGaps[a].at(i),
			 minGapsBetweenActivitiesListOfWeightPercentages[a].at(i));
		for(int i=0; i<activitiesSameStartingTimeActivities[a].count(); i++)
			add(activitiesSameStartingTimeActivities[a].at(i), T_SAMETIME, 0, activitiesSameStartingTimePercentages[a].at(i));
		for(int i=0; i<activitiesSameStartingHourActivities[a].count(); i++)
			add(activitiesSameStartingHourActivities[a].at(i), T_SAMEHOUR, 0, activitiesSameStartingHourPercentages[a].at(i));
		for(int i=0; i<activitiesSameStartingDayActivities[a].count(); i++)
			add(activitiesSameStartingDayActivities[a].at(i), T_SAMEDAY, 0, activitiesSameStartingDayPercentages[a].at(i));
		for(int i=0; i<activitiesNotOverlappingActivities[a].count(); i++)
			add(activitiesNotOverlappingActivities[a].at(i), T_NOTOVERLAP, 0, activitiesNotOverlappingPercentages[a].at(i));
		for(int i=0; i<constrTwoActivitiesConsecutiveActivities[a].count(); i++)
			add(constrTwoActivitiesConsecutiveActivities[a].at(i), T_CONSEC_AB, 0, constrTwoActivitiesConsecutivePercentages[a].at(i));
		for(int i=0; i<inverseConstrTwoActivitiesConsecutiveActivities[a].count(); i++)
			add(inverseConstrTwoActivitiesConsecutiveActivities[a].at(i), T_CONSEC_BA, 0, inverseConstrTwoActivitiesConsecutivePercentages[a].at(i));
		for(int i=0; i<constrTwoActivitiesGroupedActivities[a].count(); i++)
			add(constrTwoActivitiesGroupedActivities[a].at(i), T_GROUPED, 0, constrTwoActivitiesGroupedPercentages[a].at(i));
		for(int i=0; i<constrTwoActivitiesOrderedActivities[a].count(); i++)
			add(constrTwoActivitiesOrderedActivities[a].at(i), T_ORDERED_AB, 0, constrTwoActivitiesOrderedPercentages[a].at(i));
		for(int i=0; i<inverseConstrTwoActivitiesOrderedActivities[a].count(); i++)
			add(inverseConstrTwoActivitiesOrderedActivities[a].at(i), T_ORDERED_BA, 0, inverseConstrTwoActivitiesOrderedPercentages[a].at(i));
		if(endsDayPartners.contains(a))
			for(int b : std::as_const(endsDayPartners[a])){
				if(activityEndsStudentsDayPercentages[a]>=0)
					add(b, T_ENDS_A, 0, activityEndsStudentsDayPercentages[a]);
				if(activityEndsStudentsDayPercentages[b]>=0)
					add(b, T_ENDS_B, 0, activityEndsStudentsDayPercentages[b]);
			}
		p.pairStart[a+1]=p.pairOther.count();
	}
	p.tripStart.resize(A+1);
	p.tripStart[0]=0;
	for(int a=0; a<A; a++){
		for(int i=0; i<constrThreeActivitiesGroupedActivities[a].count(); i++){
			p.tripB.append(constrThreeActivitiesGroupedActivities[a].at(i).first);
			p.tripC.append(constrThreeActivitiesGroupedActivities[a].at(i).second);
			p.tripW.append(float(constrThreeActivitiesGroupedPercentages[a].at(i)));
		}
		p.tripStart[a+1]=p.tripB.count();
	}

	//teachers and subgroups
	p.resP.fill(-1, p.nRes*NP);
	p.resW.fill(0.0f, p.nRes*NP);
	p.resAvail.resize(p.nRes*D);
	for(int t=0; t<nT; t++){
		p.resP[t*NP+P_FLAGS]=0;
		setParam(p, t, P_MAXHD1, teachersMaxHoursDailyMaxHours1[t], teachersMaxHoursDailyPercentages1[t]);
		setParam(p, t, P_MAXHD2, teachersMaxHoursDailyMaxHours2[t], teachersMaxHoursDailyPercentages2[t]);
		setParam(p, t, P_MAXHC1, teachersMaxHoursContinuouslyMaxHours1[t], teachersMaxHoursContinuouslyPercentages1[t]);
		setParam(p, t, P_MAXHC2, teachersMaxHoursContinuouslyMaxHours2[t], teachersMaxHoursContinuouslyPercentages2[t]);
		setParam(p, t, P_MINHD, teachersMinHoursDailyMinHours[t][1], teachersMinHoursDailyPercentages[t][1]); //official mode uses column 1
		p.resP[t*NP+P_MINHD_ALLOWEMPTY]=1; //the CPU engine always allows free days for teachers
		setParam(p, t, P_MAXGW, teachersMaxGapsPerWeekMaxGaps[t], teachersMaxGapsPerWeekPercentage[t]);
		setParam(p, t, P_MAXGD, teachersMaxGapsPerDayMaxGaps[t], teachersMaxGapsPerDayPercentage[t]);
		setParam(p, t, P_MAXDAYS, teachersMaxDaysPerWeekMaxDays[t], teachersMaxDaysPerWeekWeightPercentages[t]);
		setParam(p, t, P_MINDAYS, teachersMinDaysPerWeekMinDays[t], teachersMinDaysPerWeekPercentages[t]);
		for(int i=0; i<teachersIntervalMaxDaysPerWeekMaxDays[t].count() && i<3; i++)
			setInterval(p, t, P_INT1+3*i, teachersIntervalMaxDaysPerWeekMaxDays[t].at(i), teachersIntervalMaxDaysPerWeekIntervalStart[t].at(i),
			 teachersIntervalMaxDaysPerWeekIntervalEnd[t].at(i), teachersIntervalMaxDaysPerWeekPercentages[t].at(i));
		setParam(p, t, P_MAXSPAN, teachersMaxSpanPerDayMaxSpan[t], teachersMaxSpanPerDayPercentages[t]);
		setParam(p, t, P_RESTC, teachersMinRestingHoursCircularMinHours[t], teachersMinRestingHoursCircularPercentages[t]);
		setParam(p, t, P_RESTN, teachersMinRestingHoursNotCircularMinHours[t], teachersMinRestingHoursNotCircularPercentages[t]);
		for(int d=0; d<D; d++){
			cl_ulong m=0;
			for(int h=0; h<H; h++)
				if(!breakDayHour(d, h) && !teacherNotAvailableDayHour(t, d, h))
					m|=(cl_ulong(1)<<h);
			p.resAvail[t*D+d]=m;
		}
	}
	for(int g=0; g<nG; g++){
		int rr=nT+g;
		p.resP[rr*NP+P_FLAGS]=0;
		setParam(p, rr, P_MAXHD1, subgroupsMaxHoursDailyMaxHours1[g], subgroupsMaxHoursDailyPercentages1[g]);
		setParam(p, rr, P_MAXHD2, subgroupsMaxHoursDailyMaxHours2[g], subgroupsMaxHoursDailyPercentages2[g]);
		setParam(p, rr, P_MAXHC1, subgroupsMaxHoursContinuouslyMaxHours1[g], subgroupsMaxHoursContinuouslyPercentages1[g]);
		setParam(p, rr, P_MAXHC2, subgroupsMaxHoursContinuouslyMaxHours2[g], subgroupsMaxHoursContinuouslyPercentages2[g]);
		setParam(p, rr, P_MINHD, subgroupsMinHoursDailyMinHours[g][1], subgroupsMinHoursDailyPercentages[g][1]);
		p.resP[rr*NP+P_MINHD_ALLOWEMPTY]=subgroupsMinHoursDailyAllowEmptyDays[g] ? 1 : 0;
		setParam(p, rr, P_MAXGW, subgroupsMaxGapsPerWeekMaxGaps[g], subgroupsMaxGapsPerWeekPercentage[g]);
		if(haveStudentsMaxGapsPerDay)
			setParam(p, rr, P_MAXGD, subgroupsMaxGapsPerDayMaxGaps[g], subgroupsMaxGapsPerDayPercentage[g]);
		setParam(p, rr, P_EARLY, subgroupsEarlyMaxBeginningsAtSecondHourMaxBeginnings[g], subgroupsEarlyMaxBeginningsAtSecondHourPercentage[g]);
		setParam(p, rr, P_MAXDAYS, subgroupsMaxDaysPerWeekMaxDays[g], subgroupsMaxDaysPerWeekWeightPercentages[g]);
		for(int i=0; i<subgroupsIntervalMaxDaysPerWeekMaxDays[g].count() && i<3; i++)
			setInterval(p, rr, P_INT1+3*i, subgroupsIntervalMaxDaysPerWeekMaxDays[g].at(i), subgroupsIntervalMaxDaysPerWeekIntervalStart[g].at(i),
			 subgroupsIntervalMaxDaysPerWeekIntervalEnd[g].at(i), subgroupsIntervalMaxDaysPerWeekPercentages[g].at(i));
		setParam(p, rr, P_MAXSPAN, subgroupsMaxSpanPerDayMaxSpan[g], subgroupsMaxSpanPerDayPercentages[g]);
		setParam(p, rr, P_RESTC, subgroupsMinRestingHoursCircularMinHours[g], subgroupsMinRestingHoursCircularPercentages[g]);
		setParam(p, rr, P_RESTN, subgroupsMinRestingHoursNotCircularMinHours[g], subgroupsMinRestingHoursNotCircularPercentages[g]);
		for(int d=0; d<D; d++){
			cl_ulong m=0;
			for(int h=0; h<H; h++)
				if(!breakDayHour(d, h) && !subgroupNotAvailableDayHour(g, d, h))
					m|=(cl_ulong(1)<<h);
			p.resAvail[rr*D+d]=m;
		}
	}

	//first non-break hour at or after h, per day (NH if none)
	p.nextNB.resize(D*(H+1));
	for(int d=0; d<D; d++){
		p.nextNB[d*(H+1)+H]=H;
		for(int h=H-1; h>=0; h--)
			p.nextNB[d*(H+1)+h]=breakDayHour(d, h) ? p.nextNB[d*(H+1)+h+1] : h;
	}

	//groups: union-find over 100% "same starting time" pairs, the smallest index leads
	p.leader.resize(A);
	for(int a=0; a<A; a++)
		p.leader[a]=a;
	auto find=[&](int a){
		while(p.leader[a]!=a){
			p.leader[a]=p.leader[p.leader[a]];
			a=p.leader[a];
		}
		return a;
	};
	for(int a=0; a<A; a++)
		for(int i=0; i<activitiesSameStartingTimeActivities[a].count(); i++)
			if(activitiesSameStartingTimePercentages[a].at(i)>=100.0){
				int x=find(a), y=find(activitiesSameStartingTimeActivities[a].at(i));
				if(x!=y)
					p.leader[qMax(x, y)]=qMin(x, y);
			}
	for(int a=0; a<A; a++)
		p.leader[a]=find(a);
	QVector<QList<int> > members(A);
	for(int a=0; a<A; a++)
		members[p.leader[a]].append(a);
	p.maxGroup=1;
	p.memStart.resize(A+1);
	p.gResStart.resize(A+1);
	p.gResMemStart.append(0);
	for(int a=0; a<A; a++){
		p.memStart[a]=p.mem.count();
		p.gResStart[a]=p.gRes.count();
		if(p.leader[a]==a){
			for(int m : std::as_const(members[a]))
				p.mem.append(m);
			p.maxGroup=qMax(p.maxGroup, members[a].count());
			//distinct resources of the group, each with the members that use it
			QMap<int, QList<int> > users;
			for(int m : std::as_const(members[a]))
				for(int i=p.actResStart[m]; i<p.actResStart[m+1]; i++)
					users[p.actRes[i]].append(m);
			for(QMap<int, QList<int> >::const_iterator it=users.constBegin(); it!=users.constEnd(); ++it){
				p.gRes.append(it.key());
				for(int m : it.value())
					p.gResMem.append(m);
				p.gResMemStart.append(p.gResMem.count());
			}
		}
	}
	p.memStart[A]=p.mem.count();
	p.gResStart[A]=p.gRes.count();
	if(p.maxGroup>128)
		return QString("more than 128 activities are linked by 100% same starting time constraints");
	
	//groups that can move at all
	p.grpMovable.fill(0, A);
	for(int a=0; a<A; a++){
		if(p.leader[a]!=a)
			continue;
		int nAllowed=0;
		for(int s=0; s<S; s++){
			bool ok=true;
			for(int m : std::as_const(members[a]))
				if(p.timeCost[m*S+s]<0)
					ok=false;
			if(ok)
				nAllowed++;
		}
		if(nAllowed==0)
			return QString("impossible: the activities starting together with the activity with id %1 have no common allowed time slot")
			 .arg(r.internalActivitiesList[a].id);
		bool roomChoice=false;
		for(int m : std::as_const(members[a]))
			if(p.actRoomStart[m+1]-p.actRoomStart[m]+(p.actNoRoomCost[m]>=0 ? 1 : 0)>1)
				roomChoice=true;
		if(nAllowed>1 || roomChoice){
			p.movable.append(a);
			p.grpMovable[a]=1;
		}
	}
	if(p.movable.isEmpty())
		p.movable.append(0);

	//OpenCL does not allow empty buffers
	for(QVector<int>* v : {&p.actRes, &p.actRooms, &p.pairOther, &p.pairType, &p.pairParam, &p.tripB, &p.tripC, &p.gRes, &p.gResMem})
		if(v->isEmpty()) v->append(0);
	for(QVector<float>* v : {&p.actRoomCost, &p.pairW, &p.tripW})
		if(v->isEmpty()) v->append(0.0f);
	if(p.resAvail.isEmpty()) p.resAvail.append(0);
	if(p.resP.isEmpty()){ p.resP.fill(0, NP); p.resW.fill(0.0f, NP); }

	return QString();
}

//Owns the OpenCL objects of one solve() call and frees them on exit
struct Session
{
	cl_context context=nullptr;
	cl_command_queue queue=nullptr;
	cl_program program=nullptr;
	cl_kernel kInit=nullptr, kSearch=nullptr;
	QList<cl_mem> buffers;

	~Session()
	{
		OpenCL& c=cl();
		for(cl_mem m : std::as_const(buffers))
			c.releaseMemObject(m);
		if(kInit) c.releaseKernel(kInit);
		if(kSearch) c.releaseKernel(kSearch);
		if(program) c.releaseProgram(program);
		if(queue) c.releaseCommandQueue(queue);
		if(context) c.releaseContext(context);
	}

	cl_mem buffer(size_t bytes, const void* data, bool readOnly, cl_int* err)
	{
		OpenCL& c=cl();
		cl_bitfield flags=readOnly ? CL_MEM_READ_ONLY_ : CL_MEM_READ_WRITE_;
		if(data!=nullptr)
			flags|=CL_MEM_COPY_HOST_PTR_;
		cl_mem m=c.createBuffer(context, flags, qMax<size_t>(bytes, 4), const_cast<void*>(data), err);
		if(m!=nullptr)
			buffers.append(m);
		return m;
	}
};

template <typename T> cl_mem upload(Session& s, const QVector<T>& v, cl_int* err)
{
	return s.buffer(v.size()*sizeof(T), v.constData(), true, err);
}

//FET_GPU_DEBUG: list which 100% constraints FET's own code finds broken in the best unfinished GPU timetable
void debugReport(Session& ses, cl_mem bTimes, cl_mem bRooms, const QVector<int>& info, const QVector<float>& infoF, int nc, int nA)
{
	Q_UNUSED(info);
	int best=0;
	for(int k=1; k<nc; k++)
		if(infoF[k*CF_N+CF_HARD]<infoF[best*CF_N+CF_HARD])
			best=k;
	QVector<int> times(nA), rooms(nA);
	cl().enqueueReadBuffer(ses.queue, bTimes, 1, size_t(best)*nA*4, size_t(nA)*4, times.data(), 0, nullptr, nullptr);
	cl().enqueueReadBuffer(ses.queue, bRooms, 1, size_t(best)*nA*4, size_t(nA)*4, rooms.data(), 0, nullptr, nullptr);
	static Solution s;
	s.makeUnallocated(gt.rules);
	for(int a=0; a<nA; a++){
		s.times[a]=times[a];
		s.rooms[a]=(rooms[a]<0) ? UNSPECIFIED_ROOM : rooms[a];
	}
	Rules& r=gt.rules;
	std::lock_guard<std::mutex> lock(constraintCheckMutex);
	s.changedForMatrixCalculationTeachers=s.changedForMatrixCalculationStudents=s.changedForMatrixCalculationRooms=true;
	s.teachersMatrixReady=s.subgroupsMatrixReady=s.roomsMatrixReady=false;
	s.nPlacedActivities=-1;
	FakeString report;
	QList<double> cl2;
	QList<QString> dl;
	fprintf(stderr, "GPU debug: chain %d, model hard cost %g, soft %g\n", best, infoF[best*CF_N+CF_HARD], infoF[best*CF_N+CF_SOFT]);
	for(int i=0; i<r.nInternalTimeConstraints; i++){
		TimeConstraint* ctr=r.internalTimeConstraintsList[i];
		if(ctr->weightPercentage<100.0) continue;
		cl2.clear(); dl.clear();
		double f=ctr->fitness(s, r, cl2, dl, &report);
		if(f>0) fprintf(stderr, "  broken x%g: %s\n", f, qPrintable(ctr->getDescription(r)));
	}
	for(int i=0; i<r.nInternalSpaceConstraints; i++){
		SpaceConstraint* ctr=r.internalSpaceConstraintsList[i];
		if(ctr->weightPercentage<100.0) continue;
		cl2.clear(); dl.clear();
		double f=ctr->fitness(s, r, cl2, dl, &report);
		if(f>0) fprintf(stderr, "  broken x%g: %s\n", f, qPrintable(ctr->getDescription(r)));
	}
}

}

QStringList GpuSolver::devices(QString* error)
{
	QStringList names;
	for(const Device& d : listDevices(error))
		names.append(d.isGpu ? d.name : d.name+QString(" (CPU OpenCL)"));
	return names;
}

QString GpuSolver::unsupportedReason()
{
	Rules& r=gt.rules;
	if(r.mode!=OFFICIAL)
		return QString("the GPU engine supports only the official mode");
	//the constraint types the GPU engine models; any other type makes it refuse the data
	static const QSet<int> timeTypes{
		CONSTRAINT_BASIC_COMPULSORY_TIME,
		CONSTRAINT_BREAK_TIMES,
		CONSTRAINT_TEACHER_NOT_AVAILABLE_TIMES,
		CONSTRAINT_TEACHERS_MAX_HOURS_DAILY,
		CONSTRAINT_TEACHER_MAX_DAYS_PER_WEEK,
		CONSTRAINT_TEACHERS_MAX_GAPS_PER_WEEK,
		CONSTRAINT_TEACHER_MAX_GAPS_PER_WEEK,
		CONSTRAINT_TEACHER_MAX_HOURS_DAILY,
		CONSTRAINT_TEACHERS_MAX_HOURS_CONTINUOUSLY,
		CONSTRAINT_TEACHER_MAX_HOURS_CONTINUOUSLY,
		CONSTRAINT_TEACHERS_MIN_HOURS_DAILY,
		CONSTRAINT_TEACHER_MIN_HOURS_DAILY,
		CONSTRAINT_TEACHERS_MAX_GAPS_PER_DAY,
		CONSTRAINT_TEACHER_MAX_GAPS_PER_DAY,
		CONSTRAINT_STUDENTS_EARLY_MAX_BEGINNINGS_AT_SECOND_HOUR,
		CONSTRAINT_STUDENTS_SET_EARLY_MAX_BEGINNINGS_AT_SECOND_HOUR,
		CONSTRAINT_STUDENTS_SET_NOT_AVAILABLE_TIMES,
		CONSTRAINT_STUDENTS_MAX_GAPS_PER_WEEK,
		CONSTRAINT_STUDENTS_SET_MAX_GAPS_PER_WEEK,
		CONSTRAINT_STUDENTS_MAX_HOURS_DAILY,
		CONSTRAINT_STUDENTS_SET_MAX_HOURS_DAILY,
		CONSTRAINT_STUDENTS_MAX_HOURS_CONTINUOUSLY,
		CONSTRAINT_STUDENTS_SET_MAX_HOURS_CONTINUOUSLY,
		CONSTRAINT_STUDENTS_MIN_HOURS_DAILY,
		CONSTRAINT_STUDENTS_SET_MIN_HOURS_DAILY,
		CONSTRAINT_ACTIVITY_ENDS_STUDENTS_DAY,
		CONSTRAINT_ACTIVITY_PREFERRED_STARTING_TIME,
		CONSTRAINT_ACTIVITIES_SAME_STARTING_TIME,
		CONSTRAINT_ACTIVITIES_NOT_OVERLAPPING,
		CONSTRAINT_MIN_DAYS_BETWEEN_ACTIVITIES,
		CONSTRAINT_ACTIVITY_PREFERRED_TIME_SLOTS,
		CONSTRAINT_ACTIVITIES_PREFERRED_TIME_SLOTS,
		CONSTRAINT_ACTIVITY_PREFERRED_STARTING_TIMES,
		CONSTRAINT_ACTIVITIES_PREFERRED_STARTING_TIMES,
		CONSTRAINT_ACTIVITIES_SAME_STARTING_HOUR,
		CONSTRAINT_ACTIVITIES_SAME_STARTING_DAY,
		CONSTRAINT_TWO_ACTIVITIES_CONSECUTIVE,
		CONSTRAINT_TWO_ACTIVITIES_ORDERED,
		CONSTRAINT_MIN_GAPS_BETWEEN_ACTIVITIES,
		CONSTRAINT_SUBACTIVITIES_PREFERRED_TIME_SLOTS,
		CONSTRAINT_SUBACTIVITIES_PREFERRED_STARTING_TIMES,
		CONSTRAINT_TEACHER_INTERVAL_MAX_DAYS_PER_WEEK,
		CONSTRAINT_TEACHERS_INTERVAL_MAX_DAYS_PER_WEEK,
		CONSTRAINT_STUDENTS_SET_INTERVAL_MAX_DAYS_PER_WEEK,
		CONSTRAINT_STUDENTS_INTERVAL_MAX_DAYS_PER_WEEK,
		CONSTRAINT_ACTIVITIES_END_STUDENTS_DAY,
		CONSTRAINT_TWO_ACTIVITIES_GROUPED,
		CONSTRAINT_TEACHERS_MAX_DAYS_PER_WEEK,
		CONSTRAINT_THREE_ACTIVITIES_GROUPED,
		CONSTRAINT_MAX_DAYS_BETWEEN_ACTIVITIES,
		CONSTRAINT_TEACHERS_MIN_DAYS_PER_WEEK,
		CONSTRAINT_TEACHER_MIN_DAYS_PER_WEEK,
		CONSTRAINT_STUDENTS_MAX_GAPS_PER_DAY,
		CONSTRAINT_STUDENTS_SET_MAX_GAPS_PER_DAY,
		CONSTRAINT_STUDENTS_SET_MAX_DAYS_PER_WEEK,
		CONSTRAINT_STUDENTS_MAX_DAYS_PER_WEEK,
		CONSTRAINT_TEACHER_MAX_SPAN_PER_DAY,
		CONSTRAINT_TEACHERS_MAX_SPAN_PER_DAY,
		CONSTRAINT_TEACHER_MIN_RESTING_HOURS,
		CONSTRAINT_TEACHERS_MIN_RESTING_HOURS,
		CONSTRAINT_STUDENTS_SET_MAX_SPAN_PER_DAY,
		CONSTRAINT_STUDENTS_MAX_SPAN_PER_DAY,
		CONSTRAINT_STUDENTS_SET_MIN_RESTING_HOURS,
		CONSTRAINT_STUDENTS_MIN_RESTING_HOURS
	};
	static const QSet<int> spaceTypes{
		CONSTRAINT_BASIC_COMPULSORY_SPACE,
		CONSTRAINT_ROOM_NOT_AVAILABLE_TIMES,
		CONSTRAINT_ACTIVITY_PREFERRED_ROOM,
		CONSTRAINT_ACTIVITY_PREFERRED_ROOMS,
		CONSTRAINT_STUDENTS_SET_HOME_ROOM,
		CONSTRAINT_STUDENTS_SET_HOME_ROOMS,
		CONSTRAINT_TEACHER_HOME_ROOM,
		CONSTRAINT_TEACHER_HOME_ROOMS,
		CONSTRAINT_SUBJECT_PREFERRED_ROOM,
		CONSTRAINT_SUBJECT_PREFERRED_ROOMS,
		CONSTRAINT_SUBJECT_ACTIVITY_TAG_PREFERRED_ROOM,
		CONSTRAINT_SUBJECT_ACTIVITY_TAG_PREFERRED_ROOMS,
		CONSTRAINT_ACTIVITY_TAG_PREFERRED_ROOM,
		CONSTRAINT_ACTIVITY_TAG_PREFERRED_ROOMS
	};
	for(int i=0; i<r.nInternalTimeConstraints; i++)
		if(!timeTypes.contains(r.internalTimeConstraintsList[i]->type))
			return QString("the GPU engine does not support the time constraint: %1").arg(r.internalTimeConstraintsList[i]->getDescription(r));
	for(int i=0; i<r.nInternalSpaceConstraints; i++)
		if(!spaceTypes.contains(r.internalSpaceConstraintsList[i]->type))
			return QString("the GPU engine does not support the space constraint: %1").arg(r.internalSpaceConstraintsList[i]->getDescription(r));
	for(int i=0; i<r.nInternalRooms; i++)
		if(r.internalRoomsList[i]->isVirtual)
			return QString("the GPU engine does not support virtual rooms (room %1)").arg(r.internalRoomsList[i]->name);
	for(int t=0; t<r.nInternalTeachers; t++)
		if(teachersIntervalMaxDaysPerWeekMaxDays[t].count()>3)
			return QString("the GPU engine supports at most 3 interval max days per week constraints per teacher");
	for(int g=0; g<r.nInternalSubgroups; g++)
		if(subgroupsIntervalMaxDaysPerWeekMaxDays[g].count()>3)
			return QString("the GPU engine supports at most 3 interval max days per week constraints per students set");
	if(r.nHoursPerDay>64)
		return QString("the GPU engine supports at most 64 hours per day");
	return QString();
}

GpuSolver::Result GpuSolver::solve(int maxSeconds, int device, int chains, unsigned int seed, Solution& sol, QString& message, std::atomic<bool>* abort,
 const QVector<int>* start)
{
	QElapsedTimer clock;
	clock.start();

	message=unsupportedReason();
	if(!message.isEmpty())
		return Unsupported;

	QString err;
	QList<Device> devs=listDevices(&err);
	if(devs.isEmpty()){
		message=err;
		return NotAvailable;
	}
	if(device<0 || device>=devs.count())
		device=0;
	const Device dev=devs.at(device);
	OpenCL& c=cl();

	Problem p;
	message=buildProblem(p);
	if(!message.isEmpty())
		return message.startsWith("impossible") ? Impossible : Unsupported;

	QFile kf(":/engine/gpusolver_kernel.cl");
	if(!kf.open(QIODevice::ReadOnly)){
		message=QString("the GPU kernel source is missing from the program");
		return NotAvailable;
	}
	QByteArray source=kf.readAll();

	Session ses;
	cl_int e=0;
	ses.context=c.createContext(nullptr, 1, &dev.id, nullptr, nullptr, &e);
	if(ses.context==nullptr){
		message=QString("cannot use the device %1 (OpenCL error %2)").arg(dev.name).arg(e);
		return NotAvailable;
	}
	ses.queue=c.createCommandQueue(ses.context, dev.id, 0, &e);
	if(ses.queue==nullptr){
		message=QString("cannot create an OpenCL queue (error %1)").arg(e);
		return NotAvailable;
	}

	//how many independent searches fit on the device
	cl_ulong globalMem=0, maxAlloc=0, localMem=0;
	size_t maxWG=0;
	c.getDeviceInfo(dev.id, CL_DEVICE_GLOBAL_MEM_SIZE_, sizeof(globalMem), &globalMem, nullptr);
	c.getDeviceInfo(dev.id, CL_DEVICE_MAX_MEM_ALLOC_SIZE_, sizeof(maxAlloc), &maxAlloc, nullptr);
	c.getDeviceInfo(dev.id, CL_DEVICE_LOCAL_MEM_SIZE_, sizeof(localMem), &localMem, nullptr);
	c.getDeviceInfo(dev.id, CL_DEVICE_MAX_WORK_GROUP_SIZE_, sizeof(maxWG), &maxWG, nullptr);
	if(maxWG<size_t(WG)){
		message=QString("the device %1 is too small (work-group size %2)").arg(dev.name).arg(maxWG);
		return NotAvailable;
	}
	size_t localBytes=size_t(p.nS)*(sizeof(float)*2+sizeof(int)*3)+WG*(sizeof(float)*3+sizeof(int))+(64*3+512+256*3+16)*sizeof(int)+size_t((p.nRes+31)/32)*4;
	if(localBytes>localMem){
		message=QString("too many time slots per week for the device %1").arg(dev.name);
		return NotAvailable;
	}
	const size_t occBytes=size_t(p.nRes)*p.nS*2, roomOccBytes=size_t(qMax(1, p.nRooms))*p.nS*2;
	const size_t perChain=size_t(p.nA)*4*8+occBytes+size_t(p.nS)*4+size_t(p.nRes)*p.nD*32+roomOccBytes+size_t(p.nRes)*8+WG*4+(CI_N+CF_N)*4;
	int nc=chains;
	if(nc<=0)
		nc=int(dev.computeUnits)*8;
	nc=qMax(1, int(qMin<cl_ulong>(cl_ulong(nc), (globalMem/2)/qMax<size_t>(perChain, 1))));
	nc=qMax(1, int(qMin<cl_ulong>(cl_ulong(nc), maxAlloc/qMax<size_t>(qMax(occBytes, qMax(roomOccBytes, size_t(p.nRes)*p.nD*32)), 1))));

	QString options=QString("-cl-std=CL1.2 -DNA=%1 -DNRES=%2 -DNROOMS=%3 -DND=%4 -DNH=%5 -DNS=%6 -DNP=%7 -DWG=%8 -DNMOV=%9")
	 .arg(p.nA).arg(p.nRes).arg(qMax(1, p.nRooms)).arg(p.nD).arg(p.nH).arg(p.nS).arg(NP).arg(WG).arg(p.movable.count())
	 +QString(" -DMAXG=%1 -DNRESW=%2").arg(p.maxGroup).arg((p.nRes+31)/32);
	if(qEnvironmentVariableIsSet("FET_GPU_FLAGS")) //for experiments
		options+=" "+QString::fromLocal8Bit(qgetenv("FET_GPU_FLAGS"));
	if(qEnvironmentVariableIsSet("FET_GPU_DEBUG")){
		QVector<int> need(p.nRes, 0);
		for(int a=0; a<p.nA; a++)
			for(int i=p.actResStart[a]; i<p.actResStart[a+1]; i++)
				need[p.actRes[i]]+=p.actDur[a];
		int full=0, ninety=0;
		for(int r=0; r<p.nRes; r++){
			int avail=0;
			for(int d=0; d<p.nD; d++)
				for(int h=0; h<p.nH; h++)
					if(p.resAvail[r*p.nD+d] & (cl_ulong(1)<<h)) avail++;
			if(need[r]>=avail) full++;
			else if(need[r]*10>=avail*9) ninety++;
		}
		fprintf(stderr, "GPU debug: %d teachers/subgroups are fully booked, %d are 90%%+ booked\n", full, ninety);
		fprintf(stderr, "GPU debug: %d activities, %d groups movable (largest %d), %d teachers+subgroups, %d rooms, %d slots, "
		 "%d resource uses, %d room options, %d pair entries, %d chains\n",
		 p.nA, int(p.movable.count()), p.maxGroup, p.nRes, p.nRooms, p.nS, int(p.actRes.count()), int(p.actRooms.count()), int(p.pairOther.count()), nc);
	}
	const char* src=source.constData();
	size_t srcLen=size_t(source.size());
	ses.program=c.createProgramWithSource(ses.context, 1, &src, &srcLen, &e);
	QByteArray opt=options.toLatin1();
	if(ses.program==nullptr || c.buildProgram(ses.program, 1, &dev.id, opt.constData(), nullptr, nullptr)!=CL_SUCCESS_){
		size_t len=0;
		if(ses.program!=nullptr)
			c.getProgramBuildInfo(ses.program, dev.id, CL_PROGRAM_BUILD_LOG_, 0, nullptr, &len);
		QByteArray log(int(len)+1, 0);
		if(len>0)
			c.getProgramBuildInfo(ses.program, dev.id, CL_PROGRAM_BUILD_LOG_, len, log.data(), nullptr);
		message=QString("the GPU kernel could not be compiled for %1: %2").arg(dev.name).arg(QString::fromLatin1(log).left(2000));
		return NotAvailable;
	}
	ses.kInit=c.createKernel(ses.program, "k_init", &e);
	ses.kSearch=c.createKernel(ses.program, "k_search", &e);
	if(ses.kInit==nullptr || ses.kSearch==nullptr){
		message=QString("cannot create the GPU kernels (error %1)").arg(e);
		return NotAvailable;
	}

	//problem buffers, in kernel argument order
	QList<cl_mem> prob;
	cl_int be=0;
	prob<<upload(ses, p.actDur, &be)<<upload(ses, p.actResStart, &be)<<upload(ses, p.actRes, &be)
		<<upload(ses, p.actRoomStart, &be)<<upload(ses, p.actRooms, &be)<<upload(ses, p.actRoomCost, &be)
		<<upload(ses, p.actNoRoomCost, &be)<<upload(ses, p.timeCost, &be)
		<<upload(ses, p.pairStart, &be)<<upload(ses, p.pairOther, &be)<<upload(ses, p.pairType, &be)
		<<upload(ses, p.pairParam, &be)<<upload(ses, p.pairW, &be)
		<<upload(ses, p.tripStart, &be)<<upload(ses, p.tripB, &be)<<upload(ses, p.tripC, &be)<<upload(ses, p.tripW, &be)
		<<upload(ses, p.resP, &be)<<upload(ses, p.resW, &be)<<upload(ses, p.resAvail, &be)
		<<upload(ses, p.roomTimeCost, &be)<<upload(ses, p.nextNB, &be)<<upload(ses, p.movable, &be)
		<<upload(ses, p.leader, &be)<<upload(ses, p.memStart, &be)<<upload(ses, p.mem, &be)
		<<upload(ses, p.gResStart, &be)<<upload(ses, p.gRes, &be)<<upload(ses, p.gResMemStart, &be)<<upload(ses, p.gResMem, &be)
		<<upload(ses, p.grpMovable, &be);
	//chain state buffers
	cl_mem bTimes=ses.buffer(size_t(nc)*p.nA*4, nullptr, false, &be);
	cl_mem bRooms=ses.buffer(size_t(nc)*p.nA*4, nullptr, false, &be);
	cl_mem bOcc=ses.buffer(size_t(nc)*occBytes, nullptr, false, &be);
	cl_mem bRoomOcc=ses.buffer(size_t(nc)*roomOccBytes, nullptr, false, &be);
	cl_mem bResCur=ses.buffer(size_t(nc)*p.nRes*8, nullptr, false, &be);
	cl_mem bTabuSlot=ses.buffer(size_t(nc)*p.nA*4, nullptr, false, &be);
	cl_mem bTabuUntil=ses.buffer(size_t(nc)*p.nA*4, nullptr, false, &be);
	cl_mem bRng=ses.buffer(size_t(nc)*WG*4, nullptr, false, &be);
	cl_mem bInfo=ses.buffer(size_t(nc)*CI_N*4, nullptr, false, &be);
	cl_mem bInfoF=ses.buffer(size_t(nc)*CF_N*4, nullptr, false, &be);
	cl_mem bBestTimes=ses.buffer(size_t(nc)*p.nA*4, nullptr, false, &be);
	cl_mem bBestRooms=ses.buffer(size_t(nc)*p.nA*4, nullptr, false, &be);
	cl_mem bScratch=ses.buffer(size_t(nc)*p.nA*4, nullptr, false, &be);
	cl_mem bSlotHead=ses.buffer(size_t(nc)*p.nS*4, nullptr, false, &be);
	cl_mem bNext=ses.buffer(size_t(nc)*p.nA*4, nullptr, false, &be);
	cl_mem bDay=ses.buffer(size_t(nc)*p.nRes*p.nD*32, nullptr, false, &be); //DayStat in the kernel is 32 bytes
	QList<cl_mem> chainBufs;
	chainBufs<<bTimes<<bRooms<<bOcc<<bRoomOcc<<bResCur<<bTabuSlot<<bTabuUntil<<bRng<<bInfo<<bInfoF<<bBestTimes<<bBestRooms<<bSlotHead<<bNext<<bDay;
	if(bScratch==nullptr)
		be=-1;
	for(cl_mem m : std::as_const(prob)+chainBufs+(QList<cl_mem>()<<bScratch))
		if(m==nullptr){
			message=QString("not enough memory on %1 (OpenCL error %2)").arg(dev.name).arg(be);
			return NotAvailable;
		}

	auto setCommon=[&](cl_kernel k)->cl_uint{
		cl_uint i=0;
		for(cl_mem m : std::as_const(prob)+chainBufs)
			c.setKernelArg(k, i++, sizeof(cl_mem), &m);
		return i;
	};
	const size_t global=size_t(nc)*WG, local=WG;

	//a quarter of the searches always start at random, for variety
	const int nStartChains=(start!=nullptr && start->size()==2*p.nA) ? qMax(1, nc*3/4) : 0;
	QVector<int> startData=(nStartChains>0) ? *start : QVector<int>(2*p.nA, -1);
	cl_mem bStart=upload(ses, startData, &be);
	if(bStart==nullptr){
		message=QString("not enough memory on %1 (OpenCL error %2)").arg(dev.name).arg(be);
		return NotAvailable;
	}
	cl_uint ai=setCommon(ses.kInit);
	c.setKernelArg(ses.kInit, ai++, sizeof(cl_uint), &seed);
	c.setKernelArg(ses.kInit, ai++, WG*sizeof(float)*2, nullptr);
	c.setKernelArg(ses.kInit, ai++, sizeof(cl_mem), &bStart);
	c.setKernelArg(ses.kInit, ai++, sizeof(int), &nStartChains);
	e=c.enqueueNDRangeKernel(ses.queue, ses.kInit, 1, nullptr, &global, &local, 0, nullptr, nullptr);
	if(e!=CL_SUCCESS_ || c.finish(ses.queue)!=CL_SUCCESS_){
		message=QString("the GPU could not start the search (OpenCL error %1)").arg(e);
		return NotAvailable;
	}

	cl_uint argIters=setCommon(ses.kSearch);
	c.setKernelArg(ses.kSearch, argIters++, sizeof(cl_mem), &bScratch);
	int stagnation=qMax(2000, 4*p.movable.count());
	c.setKernelArg(ses.kSearch, argIters+2, sizeof(int), &stagnation);
	c.setKernelArg(ses.kSearch, argIters+3, size_t(p.nS)*sizeof(float)*2, nullptr);
	c.setKernelArg(ses.kSearch, argIters+4, size_t(p.nS)*sizeof(int), nullptr);
	c.setKernelArg(ses.kSearch, argIters+5, WG*sizeof(float), nullptr);
	c.setKernelArg(ses.kSearch, argIters+6, WG*sizeof(int), nullptr);
	c.setKernelArg(ses.kSearch, argIters+7, WG*sizeof(float)*2, nullptr);
	c.setKernelArg(ses.kSearch, argIters+8, size_t(p.nS)*sizeof(int), nullptr);
	c.setKernelArg(ses.kSearch, argIters+9, size_t(p.nS)*sizeof(int), nullptr);

	QVector<int> info(nc*CI_N);
	QVector<float> infoF(nc*CF_N);
	int iters=16; //per launch; adapted so one launch takes about 100 ms (long launches can trip the display driver watchdog)
	qint64 solvedAt=-1;
	int polish=0;
	Result result=TimeExceeded;
	for(;;){
		if(abort!=nullptr && *abort){
			result=Aborted;
			break;
		}
		qint64 now=clock.elapsed();
		if(now>=qint64(maxSeconds)*1000){
			result=(solvedAt>=0) ? Solved : TimeExceeded;
			break;
		}
		//after the first valid timetable, spend a little extra time lowering the soft cost
		if(solvedAt>=0 && now-solvedAt>=qMin<qint64>(qMax<qint64>(solvedAt/4, 500), 10000)){
			result=Solved;
			break;
		}
		c.setKernelArg(ses.kSearch, argIters, sizeof(int), &iters);
		c.setKernelArg(ses.kSearch, argIters+1, sizeof(int), &polish);
		QElapsedTimer launch;
		launch.start();
		e=c.enqueueNDRangeKernel(ses.queue, ses.kSearch, 1, nullptr, &global, &local, 0, nullptr, nullptr);
		if(e==CL_SUCCESS_)
			e=c.enqueueReadBuffer(ses.queue, bInfo, 1, 0, size_t(nc)*CI_N*4, info.data(), 0, nullptr, nullptr);
		if(e==CL_SUCCESS_)
			e=c.enqueueReadBuffer(ses.queue, bInfoF, 1, 0, size_t(nc)*CF_N*4, infoF.data(), 0, nullptr, nullptr);
		if(e!=CL_SUCCESS_){
			message=QString("the GPU search failed (OpenCL error %1)").arg(e);
			return NotAvailable;
		}
		qint64 ms=launch.elapsed();
		if(qEnvironmentVariableIsSet("FET_GPU_DEBUG")){
			static qint64 lastPrint=0;
			if(clock.elapsed()-lastPrint>2000){
				lastPrint=clock.elapsed();
				int bh=0x7fffffff;
				for(int k=0; k<nc; k++) bh=qMin(bh, info[k*CI_N+CI_BESTHARD]);
				fprintf(stderr, "GPU debug: t=%.1fs launch %lld ms for %d steps per chain, best hard %d\n", clock.elapsed()/1000.0, ms, iters, bh);
			}
		}
		if(ms<60 && iters<1000000)
			iters*=2;
		else if(ms>200 && iters>1)
			iters=qMax(1, iters/2);

		if(solvedAt<0)
			for(int k=0; k<nc; k++)
				if(info[k*CI_N+CI_SOLVED]!=0){
					solvedAt=clock.elapsed();
					polish=1;
					break;
				}
	}
	if(result!=Solved){
		int best=0x7fffffff;
		qint64 totalIters=0;
		for(int k=0; k<nc; k++){
			best=qMin(best, info[k*CI_N+CI_BESTHARD]);
			totalIters+=info[k*CI_N+CI_ITER];
		}
		message=QString("%1 GPU searches on %2 did %3 steps; the best had %4 broken compulsory constraint(s) or clashes left")
		 .arg(nc).arg(dev.name).arg(totalIters).arg(best);
		if(qEnvironmentVariableIsSet("FET_GPU_DEBUG"))
			debugReport(ses, bTimes, bRooms, info, infoF, nc, p.nA);
		return result;
	}

	//pick the valid timetable with the lowest soft cost
	int bestChain=-1;
	for(int k=0; k<nc; k++)
		if(info[k*CI_N+CI_SOLVED]!=0 && (bestChain<0 || infoF[k*CF_N+CF_BESTFEASSOFT]<infoF[bestChain*CF_N+CF_BESTFEASSOFT]))
			bestChain=k;
	QVector<int> times(p.nA), rooms(p.nA);
	e=c.enqueueReadBuffer(ses.queue, bBestTimes, 1, size_t(bestChain)*p.nA*4, size_t(p.nA)*4, times.data(), 0, nullptr, nullptr);
	if(e==CL_SUCCESS_)
		e=c.enqueueReadBuffer(ses.queue, bBestRooms, 1, size_t(bestChain)*p.nA*4, size_t(p.nA)*4, rooms.data(), 0, nullptr, nullptr);
	if(e!=CL_SUCCESS_){
		message=QString("cannot read the timetable from the GPU (OpenCL error %1)").arg(e);
		return NotAvailable;
	}

	sol.makeUnallocated(gt.rules);
	for(int a=0; a<p.nA; a++){
		sol.times[a]=times[a];
		sol.rooms[a]=(rooms[a]<0) ? UNSPECIFIED_ROOM : rooms[a];
	}
	sol._fitness=-1;
	sol.changedForMatrixCalculationTeachers=true;
	sol.changedForMatrixCalculationStudents=true;
	sol.changedForMatrixCalculationRooms=true;

	//the GPU model is checked against FET's own constraint code before the result is accepted
	QString first;
	int broken=sol.brokenHardConstraints(gt.rules, &first);
	qint64 totalIters=0;
	for(int k=0; k<nc; k++)
		totalIters+=info[k*CI_N+CI_ITER];
	if(broken>0){
		message=QString("the GPU timetable breaks %1 compulsory constraint(s) when checked by FET (first: %2). "
		 "Please report this, and use the CPU engine for this file.").arg(broken).arg(first);
		return Unsupported;
	}
	message=QString("found by %1 parallel GPU searches on %2 in %3 s (%4 steps)")
	 .arg(nc).arg(dev.name).arg(solvedAt/1000.0, 0, 'f', 1).arg(totalIters);
	return Solved;
}
