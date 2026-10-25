/*
File gpusolver_kernel.cl - OpenCL search kernels for the FET GPU engine.

Each work-group runs one independent search ("chain") over a complete timetable.
Activities joined by 100% "same starting time" constraints form a group that always moves
together; a single activity is a group of one. A step picks a badly placed group, evaluates
every starting slot for it in parallel (one work-item per slot, each also choosing rooms),
both as a plain move and as a swap with the group that currently occupies that slot, and
applies the best non-tabu choice. Cost is (hard, soft): hard counts broken 100%
constraints and clashes, soft is the weighted sum of broken constraints below 100%.

Compile-time constants (passed with -D by the host): NA activities, NRES teachers+subgroups,
NROOMS rooms, ND days, NH hours per day, NS = ND*NH slots (slot = hour*ND + day, as in FET),
NP resource parameters, WG work-group size, NMOV movable groups (by their leader activity),
MAXG the size of the largest group.

This file is part of FET, licensed under the GNU Affero General Public License version 3 or later.
*/

#define HARDW 1000.0f

/* search settings */
#ifndef TABU_MIN
#define TABU_MIN 10       /* a group may not return to the slot it left for TABU_MIN..TABU_MIN+TABU_RAND-1 steps */
#endif
#ifndef TABU_RAND
#define TABU_RAND 10
#endif
#ifndef NOISE
#define NOISE 0.02f       /* chance of a random move when no move improves */
#endif
#ifndef RANDPICK
#define RANDPICK 0.1f     /* chance of choosing a random group instead of the worst sampled one */
#endif
#define INF 1.0e30f

/* resource parameter indices (int value in resP, weight in resW; value -1 = constraint absent) */
#define P_FLAGS 0          /* 1 if the resource has any constraint besides "no clashes" */
#define P_MAXHD1 1
#define P_MAXHD2 2
#define P_MAXHC1 3
#define P_MAXHC2 4
#define P_MINHD 5
#define P_MINHD_ALLOWEMPTY 6
#define P_MAXGW 7
#define P_MAXGD 8
#define P_MAXDAYS 9
#define P_MINDAYS 10
#define P_EARLY 11
#define P_INT1 12          /* max days; start hour in P_INT1+1, end hour in P_INT1+2 */
#define P_INT2 15
#define P_INT3 18
#define P_MAXSPAN 21
#define P_RESTC 22         /* min resting hours, circular */
#define P_RESTN 23         /* min resting hours, not circular */

/* pair constraint types; "A" is the activity whose list holds the entry, "B" the other one */
#define T_MINDAYS 1
#define T_MINDAYS_CONSEC 2
#define T_MAXDAYS 3
#define T_MINGAPS 4
#define T_SAMETIME 5
#define T_SAMEHOUR 6
#define T_SAMEDAY 7
#define T_NOTOVERLAP 8
#define T_CONSEC_AB 9      /* A immediately followed by B */
#define T_CONSEC_BA 10
#define T_GROUPED 11
#define T_ORDERED_AB 12    /* A before B */
#define T_ORDERED_BA 13
#define T_ENDS_A 14        /* A ends the students day: B (same students) must not end later that day */
#define T_ENDS_B 15

/* per chain info, ints */
#define CI_ITER 0
#define CI_LASTIMPROVE 1
#define CI_SOLVED 2
#define CI_BESTHARD 3
#define CI_N 8
/* per chain info, floats */
#define CF_HARD 0
#define CF_SOFT 1
#define CF_BESTFEASSOFT 2
#define CF_N 4

/* statistics of one teacher or subgroup on one day (cached per chain) */
typedef struct {
	float2 cost;      /* clashes and day-level constraints (hard, soft) */
	int hours, gaps, firstGaps, first, last;
	int ints;         /* bit i: busy in the interval of "interval max days per week" number i+1 */
} DayStat;

typedef struct {
	__global const int* actDur;
	__global const int* actResStart;
	__global const int* actRes;
	__global const int* actRoomStart;
	__global const int* actRooms;
	__global const float* actRoomCost;
	__global const float* actNoRoomCost;
	__global const float* timeCost;
	__global const int* pairStart;
	__global const int* pairOther;
	__global const int* pairType;
	__global const int* pairParam;
	__global const float* pairW;
	__global const int* tripStart;
	__global const int* tripB;
	__global const int* tripC;
	__global const float* tripW;
	__global const int* resP;
	__global const float* resW;
	__global const ulong* resAvail;
	__global const float* roomTimeCost;
	__global const int* nextNB;
	__global const int* movable;
	__global const int* leader;      /* leader activity of each activity's group */
	__global const int* memStart;    /* members of the group led by a: mem[memStart[a]..memStart[a+1]) */
	__global const int* mem;
	__global const int* gResStart;   /* distinct resources of the group led by a: gRes[gResStart[a]..gResStart[a+1]) */
	__global const int* gRes;
	__global const int* gResMemStart;/* members of the group using resource entry k: gResMem[gResMemStart[k]..gResMemStart[k+1]) */
	__global const int* gResMem;
	__global const int* grpMovable;  /* 1 for the leader of a group that can move */
} Problem;

#define PROBLEM_ARGS \
	__global const int* actDur, __global const int* actResStart, __global const int* actRes, \
	__global const int* actRoomStart, __global const int* actRooms, __global const float* actRoomCost, \
	__global const float* actNoRoomCost, __global const float* timeCost, \
	__global const int* pairStart, __global const int* pairOther, __global const int* pairType, \
	__global const int* pairParam, __global const float* pairW, \
	__global const int* tripStart, __global const int* tripB, __global const int* tripC, __global const float* tripW, \
	__global const int* resP, __global const float* resW, __global const ulong* resAvail, \
	__global const float* roomTimeCost, __global const int* nextNB, __global const int* movable, \
	__global const int* leader, __global const int* memStart, __global const int* mem, \
	__global const int* gResStart, __global const int* gRes, __global const int* gResMemStart, __global const int* gResMem, 	__global const int* grpMovable

#define CHAIN_ARGS \
	__global int* gTimes, __global int* gRooms, __global ushort* gOcc, __global ushort* gRoomOcc, \
	__global float2* gResCur, __global int* gTabuSlot, __global int* gTabuUntil, __global uint* gRng, \
	__global int* gInfo, __global float* gInfoF, __global int* gBestTimes, __global int* gBestRooms, __global int* gSlotHead, __global int* gNext, __global DayStat* gDay

#define FILL_PROBLEM(P) \
	P.actDur=actDur; P.actResStart=actResStart; P.actRes=actRes; P.actRoomStart=actRoomStart; \
	P.actRooms=actRooms; P.actRoomCost=actRoomCost; P.actNoRoomCost=actNoRoomCost; P.timeCost=timeCost; \
	P.pairStart=pairStart; P.pairOther=pairOther; P.pairType=pairType; P.pairParam=pairParam; P.pairW=pairW; \
	P.tripStart=tripStart; P.tripB=tripB; P.tripC=tripC; P.tripW=tripW; P.resP=resP; P.resW=resW; \
	P.resAvail=resAvail; P.roomTimeCost=roomTimeCost; P.nextNB=nextNB; P.movable=movable; \
	P.leader=leader; P.memStart=memStart; P.mem=mem; P.gResStart=gResStart; P.gRes=gRes; \
	P.gResMemStart=gResMemStart; P.gResMem=gResMem; P.grpMovable=grpMovable;

inline uint rnd(uint* s)
{
	/* xorshift32 */
	uint x=*s;
	x^=x<<13; x^=x>>17; x^=x<<5;
	*s=x;
	return x;
}

inline float rndf(uint* s)
{
	return (rnd(s)>>8)*(1.0f/16777216.0f);
}

/* weight >= 100 counts as hard, otherwise as soft weight/100 per unit */
inline void addCost(float2* c, float w, int amount)
{
	if(amount<=0)
		return;
	if(w>=100.0f)
		c->x+=amount;
	else
		c->y+=amount*w*0.01f;
}

inline bool covers(int t, int dur, int s)
{
	return t>=0 && s>=t && s<t+dur*ND && (s-t)%ND==0;
}

inline int lowBit(ulong x)
{
	return popcount((x & (~x+1UL))-1UL);
}

inline int highBit(ulong x)
{
	return 63-clz(x);
}

/* longest run of consecutive set bits */
inline int longestRun(ulong x)
{
	int n=0;
	while(x!=0){
		x&=x>>1;
		n++;
	}
	return n;
}

inline ulong hourMask(int from, int to)
{
	/* bits from..to-1 */
	ulong hi=(to>=64) ? ~0UL : ((1UL<<to)-1UL);
	return hi & ~((1UL<<from)-1UL);
}

/* occupancy count n of a resource at cell s after its users gu[0..ng) move from tg0 to tg1 */
inline int movedCount(const Problem* P, int n, int s, __global const int* gu, int ng, int tg0, int tg1)
{
	for(int k=0; k<ng; k++){
		int dur=P->actDur[gu[k]];
		if(covers(tg0, dur, s)) n--;
		if(covers(tg1, dur, s)) n++;
	}
	return n;
}

/* the statistics of one resource on one day, from its busy hours and clash count */
DayStat dayStats(const Problem* P, int r, int d, ulong busy, int clash)
{
	__global const int* p=P->resP+r*NP;
	__global const float* w=P->resW+r*NP;
	DayStat st;
	st.cost=(float2)(clash, 0.0f);
	st.hours=popcount(busy);
	st.gaps=0; st.firstGaps=0; st.first=-1; st.last=-1; st.ints=0;
	if(p[P_FLAGS]==0)
		return st;
	if(st.hours>0){
		ulong avail=P->resAvail[r*ND+d];
		st.first=lowBit(busy);
		st.last=highBit(busy);
		st.gaps=popcount(avail & ~busy & hourMask(st.first, st.last+1));
		st.firstGaps=popcount(avail & ~busy & hourMask(0, st.first));
		int run=longestRun(busy);
		if(p[P_MAXHD1]>=0) addCost(&st.cost, w[P_MAXHD1], st.hours-p[P_MAXHD1]);
		if(p[P_MAXHD2]>=0) addCost(&st.cost, w[P_MAXHD2], st.hours-p[P_MAXHD2]);
		if(p[P_MAXHC1]>=0) addCost(&st.cost, w[P_MAXHC1], run-p[P_MAXHC1]);
		if(p[P_MAXHC2]>=0) addCost(&st.cost, w[P_MAXHC2], run-p[P_MAXHC2]);
		if(p[P_MINHD]>=0) addCost(&st.cost, w[P_MINHD], p[P_MINHD]-st.hours);
		if(p[P_MAXGD]>=0) addCost(&st.cost, w[P_MAXGD], st.gaps-p[P_MAXGD]);
		if(p[P_MAXSPAN]>=0) addCost(&st.cost, w[P_MAXSPAN], st.last-st.first+1-p[P_MAXSPAN]);
		if(p[P_EARLY]>=0 && st.firstGaps>=2) addCost(&st.cost, w[P_EARLY], 1);
		if(p[P_INT1]>=0 && (busy & hourMask(p[P_INT1+1], p[P_INT1+2]))!=0) st.ints|=1;
		if(p[P_INT2]>=0 && (busy & hourMask(p[P_INT2+1], p[P_INT2+2]))!=0) st.ints|=2;
		if(p[P_INT3]>=0 && (busy & hourMask(p[P_INT3+1], p[P_INT3+2]))!=0) st.ints|=4;
	}
	else if(p[P_MINHD]>=0 && p[P_MINHD_ALLOWEMPTY]==0)
		addCost(&st.cost, w[P_MINHD], p[P_MINHD]);
	return st;
}

/* week-level constraints of resource r from its day statistics (all days given in st) */
float2 weekCost(const Problem* P, int r, const DayStat* st)
{
	__global const int* p=P->resP+r*NP;
	__global const float* w=P->resW+r*NP;
	float2 c=(float2)(0.0f, 0.0f);
	if(p[P_FLAGS]==0)
		return c;
	int weekGaps=0, busyDays=0, early1=0, int1=0, int2=0, int3=0;
	for(int d=0; d<ND; d++){
		if(st[d].hours==0)
			continue;
		weekGaps+=st[d].gaps;
		busyDays++;
		if(st[d].firstGaps>=1) early1++;
		if(st[d].ints&1) int1++;
		if(st[d].ints&2) int2++;
		if(st[d].ints&4) int3++;
		if(d>0 && st[d-1].hours>0){
			int rest=(NH-1-st[d-1].last)+st[d].first;
			if(p[P_RESTN]>=0) addCost(&c, w[P_RESTN], p[P_RESTN]-rest);
			if(p[P_RESTC]>=0) addCost(&c, w[P_RESTC], p[P_RESTC]-rest);
		}
	}
	if(p[P_RESTC]>=0 && ND>1 && st[0].hours>0 && st[ND-1].hours>0)
		addCost(&c, w[P_RESTC], p[P_RESTC]-((NH-1-st[ND-1].last)+st[0].first));
	if(p[P_MAXGW]>=0) addCost(&c, w[P_MAXGW], weekGaps-p[P_MAXGW]);
	if(p[P_MAXDAYS]>=0) addCost(&c, w[P_MAXDAYS], busyDays-p[P_MAXDAYS]);
	if(p[P_MINDAYS]>=0) addCost(&c, w[P_MINDAYS], p[P_MINDAYS]-busyDays);
	if(p[P_EARLY]>=0) addCost(&c, w[P_EARLY], early1-p[P_EARLY]);
	if(p[P_INT1]>=0) addCost(&c, w[P_INT1], int1-p[P_INT1]);
	if(p[P_INT2]>=0) addCost(&c, w[P_INT2], int2-p[P_INT2]);
	if(p[P_INT3]>=0) addCost(&c, w[P_INT3], int3-p[P_INT3]);
	return c;
}

/* statistics of resource r on day d with the moves applied (see movedCount) */
DayStat movedDay(const Problem* P, __global const ushort* occ, int r, int d, __global const int* gu, int ng, int tg0, int tg1)
{
	ulong busy=0;
	int clash=0;
	__global const ushort* o=occ+r*NS;
	for(int h=0; h<NH; h++){
		int s=h*ND+d;
		int n=movedCount(P, o[s], s, gu, ng, tg0, tg1);
		if(n>0) busy|=(1UL<<h);
		if(n>1) clash+=n-1;
	}
	return dayStats(P, r, d, busy, clash);
}


#define SOFTFIX 4096.0f /* soft costs are summed as fixed point integers (local float atomics need OpenCL 2) */
#ifndef NSWAP
#define NSWAP 8 /* chain move candidates evaluated per step */
#endif

/*
Change in the cost of resource r for every allowed slot s when the members gu[0..ng) of the
group at t0 move to s; added to lHard[s]/lSoft[s]. Slots with ldelta[s].x>=INF are skipped.
The resource's data is read once and reused for all slots.
*/
void resAllSlots(const Problem* P, __global const ushort* occ, __global const DayStat* dayCache, int r,
	__global const int* gu, int ng, int t0, __local const float2* ldelta, __local int* lHard, __local int* lSoft)
{
	DayStat st[ND];
	for(int d=0; d<ND; d++)
		st[d]=dayCache[r*ND+d];
	const bool constrained=(P->resP[r*NP+P_FLAGS]!=0);
	float2 weekOld=constrained ? weekCost(P, r, st) : (float2)(0.0f, 0.0f);
	int d0=t0%ND;
	DayStat removed=movedDay(P, occ, r, d0, gu, ng, t0, -1);
	for(int s=0; s<NS; s++){
		if(ldelta[s].x>=INF*0.5f || s==t0)
			continue;
		int ds=s%ND;
		float2 c=-st[d0].cost;
		DayStat a=(ds==d0) ? movedDay(P, occ, r, d0, gu, ng, t0, s) : removed;
		c+=a.cost;
		DayStat saved0=st[d0], saved1;
		st[d0]=a;
		if(ds!=d0){
			DayStat b=movedDay(P, occ, r, ds, gu, ng, t0, s);
			c+=b.cost-st[ds].cost;
			saved1=st[ds];
			st[ds]=b;
		}
		if(constrained)
			c+=weekCost(P, r, st)-weekOld;
		st[d0]=saved0;
		if(ds!=d0)
			st[ds]=saved1;
		atomic_add(&lHard[s], (int)round(c.x));
		atomic_add(&lSoft[s], (int)round(c.y*SOFTFIX));
	}
}

/* week cost of resource r in the current timetable; refreshes its day cache */
float2 resCostFull(const Problem* P, __global const ushort* occ, __global DayStat* dayCache, int r)
{
	DayStat st[ND];
	float2 c=(float2)(0.0f, 0.0f);
	for(int d=0; d<ND; d++){
		st[d]=movedDay(P, occ, r, d, P->mem, 0, -1, -1);
		dayCache[r*ND+d]=st[d];
		c+=st[d].cost;
	}
	return c+weekCost(P, r, st);
}

/* A (start ta, duration da) is immediately followed by B, with only break hours between them */
inline bool adjacent(const Problem* P, int ta, int da, int tb)
{
	int dA=ta%ND, hA=ta/ND, dB=tb%ND, hB=tb/ND;
	if(dA!=dB || hA+da>hB)
		return false;
	return P->nextNB[dA*(NH+1)+hA+da]==hB;
}

/* violation amount of a pair constraint (0 if respected) */
int pairViolation(const Problem* P, int type, int param, int ta, int da, int tb, int db)
{
	int dA=ta%ND, hA=ta/ND, dB=tb%ND, hB=tb/ND;
	int dist=abs(dA-dB);
	switch(type){
		case T_MINDAYS:
		case T_MINDAYS_CONSEC:
			return dist<param ? param-dist : 0;
		case T_MAXDAYS:
			return dist>param ? dist-param : 0;
		case T_MINGAPS:
			if(dA!=dB) return 0;
			if(hB>=hA) return (hA+da+param>hB) ? 1 : 0;
			return (hB+db+param>hA) ? 1 : 0;
		case T_SAMETIME:
			return ta!=tb ? 1 : 0;
		case T_SAMEHOUR:
			return hA!=hB ? 1 : 0;
		case T_SAMEDAY:
			return dA!=dB ? 1 : 0;
		case T_NOTOVERLAP:
			return (dA==dB && hA<hB+db && hB<hA+da) ? 1 : 0;
		case T_CONSEC_AB:
			return adjacent(P, ta, da, tb) ? 0 : 1;
		case T_CONSEC_BA:
			return adjacent(P, tb, db, ta) ? 0 : 1;
		case T_GROUPED:
			return (adjacent(P, ta, da, tb) || adjacent(P, tb, db, ta)) ? 0 : 1;
		case T_ORDERED_AB:
			return (dA<dB || (dA==dB && hA+da-1<hB)) ? 0 : 1;
		case T_ORDERED_BA:
			return (dB<dA || (dA==dB && hB+db-1<hA)) ? 0 : 1;
		case T_ENDS_A:
			return (dA==dB && hB+db>hA+da) ? 1 : 0;
		case T_ENDS_B:
			return (dA==dB && hA+da>hB+db) ? 1 : 0;
	}
	return 0;
}


/*
Cost of the pair constraints of activity a at slot t. Pass g=-1 when nothing moves; then every
pair counts in full and the chain total halves it (each pair is listed for both activities).
When a belongs to the moving group g, partners in g are taken at slot t too and count half,
because their own list holds the same pair.
*/
float2 pairOnlyCost(const Problem* P, __global const int* times, int a, int t, int g)
{
	float2 c=(float2)(0.0f, 0.0f);
	int da=P->actDur[a];
	for(int i=P->pairStart[a]; i<P->pairStart[a+1]; i++){
		int b=P->pairOther[i];
		int ms=(g>=0 && P->leader[b]==g) ? t : -1;
		int tb=(ms>=0) ? ms : times[b];
		int type=P->pairType[i];
		int db=P->actDur[b];
		int v=pairViolation(P, type, P->pairParam[i], t, da, tb, db);
		float2 pc=(float2)(0.0f, 0.0f);
		addCost(&pc, P->pairW[i], v);
		/* min days with "consecutive if same day": same day but not adjacent is never allowed */
		if(type==T_MINDAYS_CONSEC && v>0 && t%ND==tb%ND && !adjacent(P, t, da, tb) && !adjacent(P, tb, db, t))
			pc.x+=1.0f;
		c+=(ms>=0) ? pc*0.5f : pc;
	}
	return c;
}

/* "three activities grouped" cost of a at t; a triple is listed for all three, so with k members in the moving group g it counts 1/k */
float2 tripCost(const Problem* P, __global const int* times, int a, int t, int g)
{
	float2 c=(float2)(0.0f, 0.0f);
	for(int i=P->tripStart[a]; i<P->tripStart[a+1]; i++){
		int b=P->tripB[i], cc=P->tripC[i];
		int mb=(g>=0 && P->leader[b]==g) ? t : -1;
		int mc=(g>=0 && P->leader[cc]==g) ? t : -1;
		int t1=t, t2=(mb>=0) ? mb : times[b], t3=(mc>=0) ? mc : times[cc];
		int d1=P->actDur[a], d2=P->actDur[b], d3=P->actDur[cc];
		/* sort by start; on the same day slot order equals hour order */
		if(t2<t1){ int x=t1; t1=t2; t2=x; x=d1; d1=d2; d2=x; }
		if(t3<t2){ int x=t2; t2=t3; t3=x; x=d2; d2=d3; d3=x; }
		if(t2<t1){ int x=t1; t1=t2; t2=x; x=d1; d1=d2; d2=x; }
		bool ok=(t1%ND==t2%ND) && (t2%ND==t3%ND) && adjacent(P, t1, d1, t2) && adjacent(P, t2, d2, t3);
		float2 pc=(float2)(0.0f, 0.0f);
		addCost(&pc, P->tripW[i], ok ? 0 : 1);
		int k=(g>=0) ? 1+(mb>=0 ? 1 : 0)+(mc>=0 ? 1 : 0) : 1;
		c+=pc*(1.0f/k);
	}
	return c;
}

/* cost of activity a in room r at slot t (r=-1 for no room), not counting room clashes */
float2 roomOwnCost(const Problem* P, int a, int t, int r)
{
	int dur=P->actDur[a];
	if(r<0){
		float nr=P->actNoRoomCost[a];
		return (float2)(nr>=0.0f ? 0.0f : 1.0f, nr>=0.0f ? nr : 0.0f);
	}
	float2 c=(float2)(1.0f, 0.0f); /* a room that is not allowed is a hard violation */
	for(int i=P->actRoomStart[a]; i<P->actRoomStart[a+1]; i++)
		if(P->actRooms[i]==r){ c=(float2)(0.0f, P->actRoomCost[i]); break; }
	for(int k=0; k<dur; k++){
		float rt=P->roomTimeCost[r*NS+t+k*ND];
		if(rt<0.0f) c.x+=1.0f; else c.y+=rt;
	}
	return c;
}

/* number of cells the two placements have in common */
inline int overlapCells(int t1, int d1, int t2, int d2)
{
	if(t1%ND!=t2%ND)
		return 0;
	int h1=t1/ND, h2=t2/ND;
	return max(0, min(h1+d1, h2+d2)-max(h1, h2));
}

/* activities outside group g that use room r during the cells of an activity of duration dur at t */
int roomOthers(const Problem* P, __global const ushort* roomOcc, __global const int* times, __global const int* rooms,
	int g, int r, int t, int dur)
{
	int n=0;
	for(int k=0; k<dur; k++)
		n+=roomOcc[r*NS+t+k*ND];
	for(int i=P->memStart[g]; i<P->memStart[g+1]; i++){
		int m=P->mem[i];
		if(rooms[m]==r)
			n-=overlapCells(times[m], P->actDur[m], t, dur);
	}
	return n;
}

/* members of group h that use resource r (looked up in h's resource list) */
inline int usersInGroup(const Problem* P, int h, int r, __global const int** users)
{
	if(h<0) return 0;
	for(int k=P->gResStart[h]; k<P->gResStart[h+1]; k++)
		if(P->gRes[k]==r){
			*users=P->gResMem+P->gResMemStart[k];
			return P->gResMemStart[k+1]-P->gResMemStart[k];
		}
	return 0;
}

/*
Cost of every term that changes when group g goes to slot tg, except its teachers and subgroups
(withRes=1 adds their current cost from resCur; resAllSlots evaluates moves). choose=1 gives every
member its best room in turn (members placed earlier count as occupying their room); choose=0
keeps the current rooms. store, if not NULL, receives the rooms by activity. *changed tells
whether any member's room would change.
*/
float2 moveCost(const Problem* P, __global const int* times, __global const int* rooms, __global const ushort* roomOcc,
	__global const float2* resCur, int g, int tg, int withRes, int choose, __global int* store, bool* changed)
{
	float2 c=(float2)(0.0f, 0.0f);
	for(int i=P->memStart[g]; i<P->memStart[g+1]; i++){
		int m=P->mem[i];
		float tc=P->timeCost[m*NS+tg];
		if(tc<0.0f) c.x+=1.0f; else c.y+=tc;
		c+=pairOnlyCost(P, times, m, tg, g);
		c+=tripCost(P, times, m, tg, g);
	}
	if(withRes)
		for(int k=P->gResStart[g]; k<P->gResStart[g+1]; k++)
			c+=resCur[P->gRes[k]];

	/* rooms */
	int picks[MAXG], pdur[MAXG];
	int np=0;
	*changed=false;
	int t=tg;
	{
		for(int i=P->memStart[g]; i<P->memStart[g+1]; i++){
			int m=P->mem[i];
			int dur=P->actDur[m];
			int pick=rooms[m];
			float2 best;
			if(choose){
				best=(float2)(INF, INF);
				pick=-1;
				float nr=P->actNoRoomCost[m];
				if(nr>=0.0f)
					best=(float2)(0.0f, nr);
				for(int qq=P->actRoomStart[m]; qq<P->actRoomStart[m+1]; qq++){
					int r=P->actRooms[qq];
					float2 rc=(float2)(0.0f, P->actRoomCost[qq]);
					bool bad=false;
					for(int k=0; k<dur; k++){
						float rt=P->roomTimeCost[r*NS+t+k*ND];
						if(rt<0.0f){ bad=true; break; }
						rc.y+=rt;
					}
					if(bad) continue;
					rc.x+=roomOthers(P, roomOcc, times, rooms, g, r, t, dur);
					for(int j=0; j<np; j++)
						if(picks[j]==r)
							rc.x+=min(pdur[j], dur); /* all members start at t */
					if(rc.x*HARDW+rc.y < best.x*HARDW+best.y){
						best=rc;
						pick=r;
					}
				}
				if(best.x>=INF*0.5f)
					return (float2)(INF, INF);
			}
			else{
				best=roomOwnCost(P, m, t, pick);
				if(pick>=0){
					best.x+=roomOthers(P, roomOcc, times, rooms, g, pick, t, dur);
					for(int j=0; j<np; j++)
						if(picks[j]==pick)
							best.x+=min(pdur[j], dur);
				}
			}
			picks[np]=pick; pdur[np]=dur; np++;
			if(pick!=rooms[m])
				*changed=true;
			if(store!=NULL)
				store[m]=pick;
			c+=best;
		}
	}
	return c;
}

inline float score(float2 c)
{
	return c.x*HARDW+c.y;
}

/* every member of group g may start at slot s */
inline bool groupAllowed(const Problem* P, int g, int s)
{
	for(int i=P->memStart[g]; i<P->memStart[g+1]; i++)
		if(P->timeCost[P->mem[i]*NS+s]<0.0f)
			return false;
	return true;
}


/*
Kempe chain moves. In a tightly packed timetable a single activity cannot move without a
clash, so a chain move exchanges two slots for a whole set of groups: starting from group g
(slot t0 -> s), every group starting at the destination slot that shares a teacher or
subgroup with a chain member joins the chain in the opposite direction, until it closes.
*/
#define KMAX 64    /* largest chain */
#define RMAX 512   /* most teachers+subgroups a chain may touch */

/* groups x and y share a teacher or subgroup (both resource lists are sorted) */
bool groupsConflict(const Problem* P, int x, int y)
{
	int i=P->gResStart[x], ie=P->gResStart[x+1], j=P->gResStart[y], je=P->gResStart[y+1];
	while(i<ie && j<je){
		int a=P->gRes[i], b=P->gRes[j];
		if(a==b) return true;
		if(a<b) i++; else j++;
	}
	return false;
}

/* chain of groups for moving g from t0 to s; returns its length, or -1 if it cannot be built */
int buildChain(const Problem* P, __global const int* slotHead, __global const int* next, int g, int t0, int s,
	__local int* cg, __local int* cfrom, __local int* cto)
{
	int n=0;
	cg[n]=g; cfrom[n]=t0; cto[n]=s; n++;
	for(int i=0; i<n; i++){
		int x=cg[i], dest=cto[i], back=cfrom[i];
		for(int y=slotHead[dest]; y>=0; y=next[y]){
			bool in=false;
			for(int j=0; j<n && !in; j++)
				if(cg[j]==y) in=true;
			if(in || !groupsConflict(P, x, y))
				continue;
			if(P->grpMovable[y]==0 || !groupAllowed(P, y, back) || n==KMAX)
				return -1;
			cg[n]=y; cfrom[n]=dest; cto[n]=back; n++;
		}
	}
	return n;
}

/* index of the chain group containing activity b, or -1 */
inline int chainIndex(const Problem* P, int b, __local const int* cg, int n)
{
	int l=P->leader[b];
	for(int i=0; i<n; i++)
		if(cg[i]==l) return i;
	return -1;
}


/* change of the cost of resource r when the chain moves (only the days of t0 and s change) */
float2 resChainDelta(const Problem* P, __global const ushort* occ, __global const DayStat* dayCache, __global const float2* resCur,
	int r, __local const int* cg, __local const int* cfrom, __local const int* cto, int n, int t0, int s)
{
	DayStat st[ND];
	for(int d=0; d<ND; d++)
		st[d]=dayCache[r*ND+d];
	/* occupancy of r on the two changed days (index 0: day of t0, 1: day of s), then the chain's changes */
	int days[2];
	days[0]=t0%ND;
	days[1]=(s%ND!=t0%ND) ? s%ND : -1;
	int cnt[2][NH];
	for(int q=0; q<2; q++)
		if(days[q]>=0)
			for(int hh=0; hh<NH; hh++)
				cnt[q][hh]=occ[r*NS+hh*ND+days[q]];
	for(int i=0; i<n; i++){
		__global const int* users=P->mem;
		int nu=usersInGroup(P, cg[i], r, &users);
		for(int u=0; u<nu; u++){
			int dur=P->actDur[users[u]];
			for(int side=0; side<2; side++){
				int t=(side==0) ? cfrom[i] : cto[i];
				int q=(t%ND==days[0]) ? 0 : 1;
				for(int k=0; k<dur; k++)
					cnt[q][t/ND+k]+=(side==0) ? -1 : 1;
			}
		}
	}
	for(int q=0; q<2; q++){
		if(days[q]<0) continue;
		ulong busy=0;
		int clash=0;
		for(int hh=0; hh<NH; hh++){
			if(cnt[q][hh]>0) busy|=(1UL<<hh);
			if(cnt[q][hh]>1) clash+=cnt[q][hh]-1;
		}
		st[days[q]]=dayStats(P, r, days[q], busy, clash);
	}
	float2 c=(float2)(0.0f, 0.0f);
	for(int d=0; d<ND; d++)
		c+=st[d].cost;
	return c+weekCost(P, r, st)-resCur[r];
}




#define MMAX 256   /* most activities a chain may move */

/*
Time, room, pair and triple cost of chain member m at slot t, before (after=false) or after
the move. Pairs and triples inside the chain are shared out as in pairOnlyCost/tripCost.
Room clashes are not included (see roomMemberDelta).
*/
float2 memberChainCost(const Problem* P, __global const int* times, __global const int* rooms,
	int m, int t, __local const int* cg, __local const int* cto, int n, bool after)
{
	float2 c=(float2)(0.0f, 0.0f);
	int dm=P->actDur[m];
	float tc=P->timeCost[m*NS+t];
	if(tc<0.0f) c.x+=1.0f; else c.y+=tc;
	c+=roomOwnCost(P, m, t, rooms[m]);
	for(int pi=P->pairStart[m]; pi<P->pairStart[m+1]; pi++){
		int b=P->pairOther[pi];
		int ci=chainIndex(P, b, cg, n);
		int tb=(ci>=0 && after) ? cto[ci] : times[b];
		int type=P->pairType[pi];
		int db=P->actDur[b];
		int v=pairViolation(P, type, P->pairParam[pi], t, dm, tb, db);
		float2 pc=(float2)(0.0f, 0.0f);
		addCost(&pc, P->pairW[pi], v);
		if(type==T_MINDAYS_CONSEC && v>0 && t%ND==tb%ND && !adjacent(P, t, dm, tb) && !adjacent(P, tb, db, t))
			pc.x+=1.0f;
		c+=(ci>=0) ? pc*0.5f : pc;
	}
	for(int ti=P->tripStart[m]; ti<P->tripStart[m+1]; ti++){
		int b=P->tripB[ti], cc=P->tripC[ti];
		int cb=chainIndex(P, b, cg, n), ccc=chainIndex(P, cc, cg, n);
		int t1=t, t2=(cb>=0 && after) ? cto[cb] : times[b], t3=(ccc>=0 && after) ? cto[ccc] : times[cc];
		int d1=dm, d2=P->actDur[b], d3=P->actDur[cc];
		if(t2<t1){ int x=t1; t1=t2; t2=x; x=d1; d1=d2; d2=x; }
		if(t3<t2){ int x=t2; t2=t3; t3=x; x=d2; d2=d3; d3=x; }
		if(t2<t1){ int x=t1; t1=t2; t2=x; x=d1; d1=d2; d2=x; }
		bool ok=(t1%ND==t2%ND) && (t2%ND==t3%ND) && adjacent(P, t1, d1, t2) && adjacent(P, t2, d2, t3);
		float2 pc=(float2)(0.0f, 0.0f);
		addCost(&pc, P->tripW[ti], ok ? 0 : 1);
		c+=pc*(1.0f/(1+(cb>=0 ? 1 : 0)+(ccc>=0 ? 1 : 0)));
	}
	return c;
}

/*
Change in room clash pairs caused by chain member j (activities keep their rooms). Every
(room, cell) touched by the move is counted once, by the first member in chain order that
touches it.
*/
float roomMemberDelta(const Problem* P, __global const int* rooms, __global const ushort* roomOcc,
	__local const int* lcm, __local const int* lcmFrom, __local const int* lcmTo, int nm, int j)
{
	int m=lcm[j];
	int r=rooms[m];
	if(r<0)
		return 0.0f;
	float delta=0.0f;
	int dur=P->actDur[m];
	for(int side=0; side<2; side++)
		for(int k=0; k<dur; k++){
			int cell=(side==0 ? lcmFrom[j] : lcmTo[j])+k*ND;
			bool seen=false;
			for(int j2=0; j2<=j && !seen; j2++){
				if(rooms[lcm[j2]]!=r) continue;
				int d2=P->actDur[lcm[j2]];
				for(int side2=0; side2<2 && !seen; side2++)
					for(int k2=0; k2<d2 && !seen; k2++){
						bool earlier=(j2<j) || (side2<side) || (side2==side && k2<k);
						if(earlier && (side2==0 ? lcmFrom[j2] : lcmTo[j2])+k2*ND==cell)
							seen=true;
					}
			}
			if(seen) continue;
			int cnt=roomOcc[r*NS+cell], cnt2=cnt;
			for(int j2=0; j2<nm; j2++){
				if(rooms[lcm[j2]]!=r) continue;
				int d2=P->actDur[lcm[j2]];
				if(covers(lcmFrom[j2], d2, cell)) cnt2--;
				if(covers(lcmTo[j2], d2, cell)) cnt2++;
			}
			delta+=(cnt2*(cnt2-1)-cnt*(cnt-1))/2;
		}
	return delta;
}

/* list the activities of the chain (one work-item); returns their number or -1 if too many */
int chainMembers(const Problem* P, __local const int* cg, __local const int* cfrom, __local const int* cto, int n,
	__local int* lcm, __local int* lcmFrom, __local int* lcmTo)
{
	int nm=0;
	for(int i=0; i<n; i++)
		for(int q=P->memStart[cg[i]]; q<P->memStart[cg[i]+1]; q++){
			if(nm==MMAX) return -1;
			lcm[nm]=P->mem[q];
			lcmFrom[nm]=cfrom[i];
			lcmTo[nm]=cto[i];
			nm++;
		}
	return nm;
}

/* all work-items: the distinct teachers+subgroups of the chain activities into lres; *lResN>RMAX means too many */
void chainResourcesParallel(const Problem* P, __local const int* lcm, int nm, __local int* lres, volatile __local int* lResN,
	volatile __local uint* bits, int lid)
{
	for(int j=lid; j<nm; j+=WG){
		int m=lcm[j];
		for(int q=P->actResStart[m]; q<P->actResStart[m+1]; q++){
			int r=P->actRes[q];
			uint bit=1u<<(r&31);
			if(!(atomic_or(&bits[r>>5], bit) & bit)){
				int idx=atomic_inc(lResN);
				if(idx<RMAX) lres[idx]=r;
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	int nr=*lResN;
	if(nr<=RMAX)
		for(int k=lid; k<nr; k+=WG)
			bits[lres[k]>>5]=0;
	else
		for(int k=lid; k<NRESW; k+=WG)
			bits[k]=0;
	barrier(CLK_LOCAL_MEM_FENCE);
}

void slotListRemove(__global int* slotHead, __global int* next, int g, int s)
{
	if(slotHead[s]==g){
		slotHead[s]=next[g];
		return;
	}
	for(int x=slotHead[s]; x>=0; x=next[x])
		if(next[x]==g){
			next[x]=next[g];
			return;
		}
}

void slotListAdd(__global int* slotHead, __global int* next, int g, int s)
{
	next[g]=slotHead[s];
	slotHead[s]=g;
}

/* move group g from slot t0 to t with the rooms in newRooms[member]; one work-item only */
void applyGroup(const Problem* P, __global int* times, __global int* rooms, __global ushort* occ, __global ushort* roomOcc,
	__global int* slotHead, __global int* next, int g, int t0, int t, __global const int* newRooms)
{
	if(t!=t0){
		slotListRemove(slotHead, next, g, t0);
		slotListAdd(slotHead, next, g, t);
	}
	for(int i=P->memStart[g]; i<P->memStart[g+1]; i++){
		int m=P->mem[i];
		int dur=P->actDur[m];
		for(int q=P->actResStart[m]; q<P->actResStart[m+1]; q++){
			int r=P->actRes[q];
			for(int k=0; k<dur; k++){
				occ[r*NS+t0+k*ND]--;
				occ[r*NS+t+k*ND]++;
			}
		}
		int r0=rooms[m], rn=newRooms[m];
		for(int k=0; k<dur; k++){
			if(r0>=0) roomOcc[r0*NS+t0+k*ND]--;
			if(rn>=0) roomOcc[rn*NS+t+k*ND]++;
		}
		times[m]=t;
		rooms[m]=rn;
	}
}

/* recompute every resource cost and the chain totals; all work-items of the group must call it */
void resync(const Problem* P, __global const int* times, __global const int* rooms, __global const ushort* occ,
	__global DayStat* dayCache, __global const ushort* roomOcc, __global float2* resCur, __global float* infoF, __local float2* scratch, int lid)
{
	float2 sum=(float2)(0.0f, 0.0f);
	for(int r=lid; r<NRES; r+=WG){
		float2 c=resCostFull(P, occ, dayCache, r);
		resCur[r]=c;
		sum+=c;
	}
	for(int a=lid; a<NA; a+=WG){
		int t=times[a];
		float tc=P->timeCost[a*NS+t];
		if(tc<0.0f) sum.x+=1.0f; else sum.y+=tc;
		sum+=roomOwnCost(P, a, t, rooms[a]);
		sum+=pairOnlyCost(P, times, a, t, -1)*0.5f;
		sum+=tripCost(P, times, a, t, -1)*(1.0f/3.0f);
	}
	/* room clashes: n activities in a room at one time make n*(n-1)/2 clashing pairs */
	for(int i=lid; i<NROOMS*NS; i+=WG){
		int n=roomOcc[i];
		if(n>1) sum.x+=n*(n-1)/2;
	}
	scratch[lid]=sum;
	barrier(CLK_LOCAL_MEM_FENCE);
	for(int k=WG/2; k>0; k>>=1){
		if(lid<k)
			scratch[lid]+=scratch[lid+k];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if(lid==0){
		infoF[CF_HARD]=round(scratch[0].x);
		infoF[CF_SOFT]=scratch[0].y;
	}
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

/* first slot allowed for every member of group g, at or after a random position */
int randomAllowedSlot(const Problem* P, int g, uint* rng)
{
	int start=rnd(rng)%NS;
	for(int k=0; k<NS; k++){
		int s=(start+k)%NS;
		if(groupAllowed(P, g, s))
			return s;
	}
	return start;
}

int randomRoom(const Problem* P, int a, uint* rng)
{
	int n=P->actRoomStart[a+1]-P->actRoomStart[a];
	if(n==0 || (P->actNoRoomCost[a]>=0.0f && rnd(rng)%(n+1)==0))
		return -1;
	return P->actRooms[P->actRoomStart[a]+rnd(rng)%n];
}

//the room `r` (-1 = no room) is a valid choice for activity a
bool roomAllowed(const Problem* P, int a, int r)
{
	int n=P->actRoomStart[a+1]-P->actRoomStart[a];
	if(r==-1)
		return n==0 || P->actNoRoomCost[a]>=0.0f;
	for(int q=P->actRoomStart[a]; q<P->actRoomStart[a+1]; q++)
		if(P->actRooms[q]==r)
			return true;
	return false;
}

//gStart: NA start times (-1 = none) then NA start rooms (-2 = none); used by the first nStartChains chains
__kernel void k_init(PROBLEM_ARGS, CHAIN_ARGS, uint seed, __local float2* scratch, __global const int* gStart, int nStartChains)
{
	Problem P; FILL_PROBLEM(P)
	int c=get_group_id(0), lid=get_local_id(0);
	__global int* times=gTimes+c*NA;
	__global int* rooms=gRooms+c*NA;
	__global ushort* occ=gOcc+(size_t)c*NRES*NS;
	__global int* slotHead=gSlotHead+c*NS;
	__global int* next=gNext+c*NA;
	__global DayStat* dayCache=gDay+(size_t)c*NRES*ND;
	__global ushort* roomOcc=gRoomOcc+(size_t)c*NROOMS*NS;
	__global float2* resCur=gResCur+c*NRES;

	uint rng=seed ^ (uint)(c*7919+lid*104729+1);
	if(rng==0) rng=1;
	for(int k=0; k<8; k++) rnd(&rng);

	for(int i=lid; i<NRES*NS; i+=WG) occ[i]=0;
	for(int i=lid; i<NS; i+=WG) slotHead[i]=-1;
	for(int i=lid; i<NROOMS*NS; i+=WG) roomOcc[i]=0;
	for(int a=lid; a<NA; a+=WG){
		gTabuSlot[c*NA+a]=-1;
		gTabuUntil[c*NA+a]=0;
		int st=(c<nStartChains) ? gStart[a] : -1;
		int sr=(c<nStartChains) ? gStart[NA+a] : -2;
		if(P.leader[a]==a)
			times[a]=(st>=0 && st<NS && groupAllowed(&P, a, st)) ? st : randomAllowedSlot(&P, a, &rng);
		rooms[a]=(sr>=-1 && roomAllowed(&P, a, sr)) ? sr : randomRoom(&P, a, &rng);
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
	for(int a=lid; a<NA; a+=WG)
		if(P.leader[a]!=a)
			times[a]=times[P.leader[a]];
	gRng[c*WG+lid]=rng;
	barrier(CLK_GLOBAL_MEM_FENCE);
	if(lid==0){
		for(int a=0; a<NA; a++){
			int t=times[a], dur=P.actDur[a];
			for(int q=P.actResStart[a]; q<P.actResStart[a+1]; q++)
				for(int k=0; k<dur; k++)
					occ[P.actRes[q]*NS+t+k*ND]++;
			if(P.leader[a]==a)
				slotListAdd(slotHead, next, a, t);
			if(rooms[a]>=0)
				for(int k=0; k<dur; k++)
					roomOcc[rooms[a]*NS+t+k*ND]++;
		}
		gInfo[c*CI_N+CI_ITER]=0;
		gInfo[c*CI_N+CI_LASTIMPROVE]=0;
		gInfo[c*CI_N+CI_SOLVED]=0;
		gInfo[c*CI_N+CI_BESTHARD]=0x7fffffff;
		gInfoF[c*CF_N+CF_BESTFEASSOFT]=INF;
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
	resync(&P, times, rooms, occ, dayCache, roomOcc, resCur, gInfoF+c*CF_N, scratch, lid);
}

/*
Runs `iters` search steps on every chain. A chain that reached hard cost 0 stores its
timetable in gBestTimes/gBestRooms; with polish=0 it then stops, with polish=1 it keeps
improving the soft cost while staying at hard cost 0.
gScratch (NA ints per chain) holds room choices while a move is applied.
*/
__kernel void k_search(PROBLEM_ARGS, CHAIN_ARGS, __global int* gScratch, int iters, int polish, int stagnation,
	__local float2* ldelta, __local int* lswap, __local float* lkey, __local int* lidx, __local float2* scratch,
	__local int* lHard, __local int* lSoft)
{
	Problem P; FILL_PROBLEM(P)
	int c=get_group_id(0), lid=get_local_id(0);
	__global int* times=gTimes+c*NA;
	__global int* rooms=gRooms+c*NA;
	__global ushort* occ=gOcc+(size_t)c*NRES*NS;
	__global int* slotHead=gSlotHead+c*NS;
	__global int* next=gNext+c*NA;
	__global DayStat* dayCache=gDay+(size_t)c*NRES*ND;
	__global ushort* roomOcc=gRoomOcc+(size_t)c*NROOMS*NS;
	__global float2* resCur=gResCur+c*NRES;
	__global int* info=gInfo+c*CI_N;
	__global float* infoF=gInfoF+c*CF_N;
	__global int* tabuSlot=gTabuSlot+c*NA;
	__global int* tabuUntil=gTabuUntil+c*NA;
	__global int* scr=gScratch+c*NA;
	uint rng=gRng[c*WG+lid];

	__local int selT, selH, doPerturb;
	__local int lSwapSlot[NSWAP], lNSwap, lSwapAcc[4];
	__local int lcg[KMAX], lcfrom[KMAX], lcto[KMAX], lres[RMAX], lChainN, lMemN;
	__local int lcm[MMAX], lcmFrom[MMAX], lcmTo[MMAX];
	volatile __local int lResN;
	volatile __local uint lbits[NRESW];
	for(int k=lid; k<NRESW; k+=WG)
		lbits[k]=0;
	__local float2 selDelta;

	/* exact totals at the start of every launch, so rounding errors cannot accumulate */
	resync(&P, times, rooms, occ, dayCache, roomOcc, resCur, infoF, scratch, lid);

	for(int step=0; step<iters; step++){
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		if(info[CI_SOLVED]!=0 && polish==0)
			break;
		int iter=info[CI_ITER];
		bool polishing=(info[CI_SOLVED]!=0);
		bool ch;

		/* 1. choose a group: the worst of WG random samples, or a random one (10%, or always when polishing) */
		{
			int g=P.movable[rnd(&rng)%NMOV];
			float bad;
			if(polishing || rndf(&rng)<RANDPICK)
				bad=rndf(&rng);
			else
				bad=score(moveCost(&P, times, rooms, roomOcc, resCur, g, times[g], 1, 0, NULL, &ch))+rndf(&rng);
			lkey[lid]=bad;
			lidx[lid]=g;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		for(int k=WG/2; k>0; k>>=1){
			if(lid<k && lkey[lid+k]>lkey[lid]){
				lkey[lid]=lkey[lid+k];
				lidx[lid]=lidx[lid+k];
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		int g=lidx[0];
		int t0=times[g];
		barrier(CLK_LOCAL_MEM_FENCE);

		/* 2. evaluate every slot for g */
		float2 base=moveCost(&P, times, rooms, roomOcc, resCur, g, t0, 0, 0, NULL, &ch);
		float curHard=infoF[CF_HARD];
		int bestHard=info[CI_BESTHARD];
		/* 2a. plain move: time, pairs and rooms per slot (one work-item per slot) */
		for(int s=lid; s<NS; s+=WG){
			float2 d=(float2)(INF, INF);
			if(groupAllowed(&P, g, s)){
				bool changed;
				float2 cst=moveCost(&P, times, rooms, roomOcc, resCur, g, s, 0, 1, NULL, &changed);
				if(cst.x<INF*0.5f && !(s==t0 && !changed))
					d=cst-base;
			}
			ldelta[s]=d;
			lHard[s]=0;
			lSoft[s]=0;
			lswap[s]=-1;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		/* 2b. plain move: teachers and subgroups (one work-item per resource, all slots at once) */
		for(int k=P.gResStart[g]+lid; k<P.gResStart[g+1]; k+=WG)
			resAllSlots(&P, occ, dayCache, P.gRes[k], P.gResMem+P.gResMemStart[k], P.gResMemStart[k+1]-P.gResMemStart[k],
			 t0, ldelta, lHard, lSoft);
		barrier(CLK_LOCAL_MEM_FENCE);
		for(int s=lid; s<NS; s+=WG)
			if(ldelta[s].x<INF*0.5f)
				ldelta[s]+=(float2)(lHard[s], lSoft[s]/SOFTFIX);
		barrier(CLK_LOCAL_MEM_FENCE);
		
		/* 2c. chain moves: for a few slots where the plain move adds hard cost, exchange the
		   slots for g and every group linked to it by shared teachers or subgroups */
		if(lid==0){
			int n=0;
			int start=rnd(&rng)%NS;
			for(int k=0; k<NS && n<NSWAP; k++){
				int s=(start+k)%NS;
				if(s!=t0 && ldelta[s].x<INF*0.5f && ldelta[s].x>=0.5f)
					lSwapSlot[n++]=s;
			}
			lNSwap=n;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		for(int j=0; j<lNSwap; j++){
			int s=lSwapSlot[j];
			if(lid==0){
				lChainN=buildChain(&P, slotHead, next, g, t0, s, lcg, lcfrom, lcto);
				lMemN=(lChainN>=2) ? chainMembers(&P, lcg, lcfrom, lcto, lChainN, lcm, lcmFrom, lcmTo) : -1;
				lResN=0;
				lSwapAcc[0]=0; lSwapAcc[1]=0; lSwapAcc[2]=0; lSwapAcc[3]=0;
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			if(lMemN<0)
				continue; /* uniform: every work-item reads the same value */
			chainResourcesParallel(&P, lcm, lMemN, lres, &lResN, lbits, lid);
			if(lResN>RMAX)
				continue;
			for(int k=lid; k<lResN; k+=WG){
				float2 dr=resChainDelta(&P, occ, dayCache, resCur, lres[k], lcg, lcfrom, lcto, lChainN, t0, s);
				atomic_add(&lSwapAcc[0], (int)round(dr.x));
				atomic_add(&lSwapAcc[1], (int)round(dr.y*SOFTFIX));
			}
			for(int k=lid; k<lMemN; k+=WG){
				float2 dm=memberChainCost(&P, times, rooms, lcm[k], lcmTo[k], lcg, lcto, lChainN, true)
					-memberChainCost(&P, times, rooms, lcm[k], lcmFrom[k], lcg, lcto, lChainN, false);
				dm.x+=roomMemberDelta(&P, rooms, roomOcc, lcm, lcmFrom, lcmTo, lMemN, k);
				/* pair shares are halves and triple shares thirds: sum hard costs in sixths */
				atomic_add(&lSwapAcc[2], (int)round(dm.x*6.0f));
				atomic_add(&lSwapAcc[3], (int)round(dm.y*SOFTFIX));
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			if(lid==0){
				float2 d=(float2)(lSwapAcc[0]+round(lSwapAcc[2]/6.0f), (lSwapAcc[1]+lSwapAcc[3])/SOFTFIX);
				if(score(d)<score(ldelta[s])){
					ldelta[s]=d;
					lswap[s]=1;
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}

		/* 2d. tabu and polishing rules */
		for(int s=lid; s<NS; s+=WG){
			float2 d=ldelta[s];
			if(d.x<INF*0.5f){
				bool tabu=(s==tabuSlot[g] && iter<tabuUntil[g]);
				if(tabu && !(curHard+d.x<bestHard))
					d=(float2)(INF, INF);
				if(polishing && curHard+d.x>0.5f)
					d=(float2)(INF, INF); /* polishing must keep the timetable valid */
			}
			ldelta[s]=d;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		
		/* 3. best slot, ties broken at random */
		{
			float bestScore=INF;
			int bestS=-1;
			float bestKey=0.0f;
			for(int s=lid; s<NS; s+=WG){
				float2 d=ldelta[s];
				if(d.x>=INF*0.5f) continue;
				float sc=score(d);
				float key=rndf(&rng);
				if(sc<bestScore || (sc==bestScore && key>bestKey)){
					bestScore=sc; bestS=s; bestKey=key;
				}
			}
			lkey[lid]=bestScore;
			lidx[lid]=bestS;
			barrier(CLK_LOCAL_MEM_FENCE);
			for(int k=WG/2; k>0; k>>=1){
				if(lid<k){
					if(lkey[lid+k]<lkey[lid] || (lkey[lid+k]==lkey[lid] && lidx[lid+k]>=0 && (rnd(&rng)&1))){
						lkey[lid]=lkey[lid+k];
						lidx[lid]=lidx[lid+k];
					}
				}
				barrier(CLK_LOCAL_MEM_FENCE);
			}
			if(lid==0){
				int s=lidx[0];
				/* random walk: sometimes take a random allowed move instead of a non-improving one */
				if(!polishing && (s<0 || lkey[0]>=0.0f) && rndf(&rng)<NOISE){
					int tries=0;
					int s2;
					do{
						s2=rnd(&rng)%NS;
						tries++;
					} while(ldelta[s2].x>=INF*0.5f && tries<NS);
					if(ldelta[s2].x<INF*0.5f) s=s2;
				}
				selT=s;
				selH=(s>=0) ? lswap[s] : -1; /* 1: chain move */
				selDelta=(s>=0) ? ldelta[s] : (float2)(0.0f, 0.0f);
				doPerturb=(!polishing && iter-info[CI_LASTIMPROVE]>stagnation) ? 1 : 0;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		/* 4. apply the move */
		int s=selT;
		if(s>=0){
			bool chain=(selH>0);
			if(lid==0){
				if(chain){
					lChainN=buildChain(&P, slotHead, next, g, t0, s, lcg, lcfrom, lcto);
					lMemN=chainMembers(&P, lcg, lcfrom, lcto, lChainN, lcm, lcmFrom, lcmTo);
					lResN=0;
					for(int i=0; i<lChainN; i++)
						for(int q=P.memStart[lcg[i]]; q<P.memStart[lcg[i]+1]; q++)
							scr[P.mem[q]]=rooms[P.mem[q]];
					for(int i=0; i<lChainN; i++){
						applyGroup(&P, times, rooms, occ, roomOcc, slotHead, next, lcg[i], lcfrom[i], lcto[i], scr);
						tabuSlot[lcg[i]]=lcfrom[i];
						tabuUntil[lcg[i]]=iter+TABU_MIN+rnd(&rng)%TABU_RAND;
					}
				}
				else{
					bool changed;
					moveCost(&P, times, rooms, roomOcc, resCur, g, s, 0, 1, scr, &changed);
					applyGroup(&P, times, rooms, occ, roomOcc, slotHead, next, g, t0, s, scr);
					if(s!=t0){
						tabuSlot[g]=t0;
						tabuUntil[g]=iter+TABU_MIN+rnd(&rng)%TABU_RAND;
					}
				}
				infoF[CF_HARD]+=selDelta.x;
				infoF[CF_SOFT]+=selDelta.y;
			}
			barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
			if(chain){
				/* the chain fitted when it was evaluated, so its resource list fits now */
				chainResourcesParallel(&P, lcm, lMemN, lres, &lResN, lbits, lid);
				for(int k=lid; k<lResN; k+=WG)
					resCur[lres[k]]=resCostFull(&P, occ, dayCache, lres[k]);
			}
			else{
				for(int k=P.gResStart[g]+lid; k<P.gResStart[g+1]; k+=WG){
					int r=P.gRes[k];
					resCur[r]=resCostFull(&P, occ, dayCache, r);
				}
			}
		}
		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
		
		/* 5. stagnation: shake the timetable with random moves, then recompute everything */
		if(doPerturb){
			if(lid==0){
				int n=NMOV/20+2;
				for(int k=0; k<n; k++){
					int b=P.movable[rnd(&rng)%NMOV];
					int sb=randomAllowedSlot(&P, b, &rng);
					for(int i=P.memStart[b]; i<P.memStart[b+1]; i++)
						scr[P.mem[i]]=randomRoom(&P, P.mem[i], &rng);
					applyGroup(&P, times, rooms, occ, roomOcc, slotHead, next, b, times[b], sb, scr);
				}
				info[CI_LASTIMPROVE]=iter;
			}
			barrier(CLK_GLOBAL_MEM_FENCE);
			resync(&P, times, rooms, occ, dayCache, roomOcc, resCur, infoF, scratch, lid);
		}

		/* 6. bookkeeping */
		if(lid==0){
			int hc=(int)round(infoF[CF_HARD]);
			if(hc<info[CI_BESTHARD]){
				info[CI_BESTHARD]=hc;
				info[CI_LASTIMPROVE]=iter;
			}
			info[CI_ITER]=iter+1;
		}
		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
		if((int)round(infoF[CF_HARD])==0 && infoF[CF_SOFT]<infoF[CF_BESTFEASSOFT]-1e-4f){
			/* confirm with an exact recount before trusting the running totals */
			resync(&P, times, rooms, occ, dayCache, roomOcc, resCur, infoF, scratch, lid);
			if((int)round(infoF[CF_HARD])==0 && infoF[CF_SOFT]<infoF[CF_BESTFEASSOFT]-1e-4f){
				for(int b=lid; b<NA; b+=WG){
					gBestTimes[c*NA+b]=times[b];
					gBestRooms[c*NA+b]=rooms[b];
				}
				barrier(CLK_GLOBAL_MEM_FENCE);
				if(lid==0){
					infoF[CF_BESTFEASSOFT]=infoF[CF_SOFT];
					info[CI_SOLVED]=1;
				}
			}
		}
	}
	gRng[c*WG+lid]=rng;
}
