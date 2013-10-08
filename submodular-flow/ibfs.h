
#ifndef _IBFS_H__
#define _IBFS_H__


//#define STATS


#pragma warning(disable:4786)
#include <time.h>
#include <sys/timeb.h>
#include <algorithm>
#include <vector>

template <typename captype, typename tcaptype, typename flowtype> class IBFSGraph
{
public:
	typedef enum
	{
		SOURCE	= 0,
		SINK	= 1
	} termtype;
	typedef int node_id;

	IBFSGraph(int numNodes, int numEdges, void (*errorFunction)(char*) = NULL);
	~IBFSGraph();
	int add_node(int numNodes);
	void add_edge(int nodeIndexFrom, int nodeIndexTo, captype capacity, captype reverseCapacity);
	void add_tweights(int nodeIndex, tcaptype capacityFromSource, tcaptype CapacityToSink);

	// to separate the graph creation and maximum flow for measurements,
	// call prepareGraph and then call maxflowClean
	// prepareGraph is only required because of the limited API for building the graph
	// (specifically - the degree of nodes is not given)
	void prepareGraph();
	flowtype maxflowClean();

	flowtype maxflow()
	{
		prepareGraph();
		return maxflowClean();
	}

	typedef std::pair<int, int> edgeInfo;
	typedef std::pair<edgeInfo, flowtype> flowInfo;
	typedef std::vector<flowInfo> flowList;
	typedef std::pair<tcaptype, tcaptype> STFlowInfo;
	typedef std::vector<STFlowInfo> STFlowList;

	flowList getFlow()
	{
		flowList ret;
		ret.clear();
		for (arc* edge = arcs; edge < arcLast; ++edge){
			ret.push_back(std::make_pair(std::make_pair(edge->u, edge->v), edge->cap - edge->rCap));
		}
		return ret;
	}
	STFlowList getSTFlow()
	{
		STFlowList ret;
		ret.clear();
		for (node* vtx = nodes; vtx < nodeLast; ++vtx){
			if (vtx->srcSinkCap >= 0)
				ret.push_back(std::make_pair(vtx->srcCap - vtx->srcSinkCap, vtx->sinkCap));
			else
				ret.push_back(std::make_pair(vtx->srcCap, vtx->sinkCap + vtx->srcSinkCap));
		}
		return ret;
	}

	termtype what_segment(int nodeIndex, termtype default_segm = SOURCE);


private:

	struct node;
	struct arc;

	struct arc
	{
		node*		head;
		arc*		sister;
		int			sister_rCap :1;
		captype		rCap;
		captype		cap;
		int		u,v;
	};

	struct node
	{
		arc			*firstArc;
		arc			*parent;
		node		*nextActive;
		node		*firstSon;
		int			nextSibling;
		int			label;		// distance to the source or the sink
								// label > 0: distance from src
								// label < 0: -distance from sink
		tcaptype		srcCap;
		tcaptype		sinkCap;		
		union
		{
			tcaptype	srcSinkCap;		// srcSinkCap > 0: capacity from the source
										// srcSinkCap < 0: -capacity to the sink
			node		*nextOrphan;
		};
	};

	struct AugmentationInfo
	{
		captype remainingDeficit;
		captype remainingExcess;
		captype flowDeficit;
		captype flowExcess;
	};

	node		*nodes, *nodeLast;
	arc			*arcs, *arcLast;
	flowtype	flow;

	void augment(arc *bridge, AugmentationInfo* augInfo);
	void adoptionSrc();
	void adoptionSink();

	node* orphanFirst;
	node* orphanLast;

	int activeLevel;
	node* activeFirst0;
	node* activeFirst1;
	node* activeLast1;

	void (*errorFunction)(char *);
	int nNodes;

#ifdef STATS
	double numAugs;
	double grownSinkTree;
	double grownSourceTree;
	double numOrphans;

	double growthArcs;
	double numPushes;
	double orphanArcs1;
	double orphanArcs2;
	double orphanArcs3;
	
	double numOrphans0;
	double numOrphans1;
	double numOrphans2;
	double augLenMin;
	double augLenMax;
#endif

};





template <typename captype, typename tcaptype, typename flowtype> inline void IBFSGraph<captype, tcaptype, flowtype>::add_tweights(int nodeIndex, tcaptype capacitySource, tcaptype capacitySink)
{
	nodes[nodeIndex].srcCap += capacitySource;
	nodes[nodeIndex].sinkCap += capacitySink;
	flowtype f = nodes[nodeIndex].srcSinkCap;
	if (f > 0)
	{
		capacitySource += f;
	}
	else
	{
		capacitySink -= f;
	}
	if (capacitySource < capacitySink)
	{
		flow += capacitySource;
	}
	else
	{
		flow += capacitySink;
	}
	nodes[nodeIndex].srcSinkCap = capacitySource - capacitySink;
}

template <typename captype, typename tcaptype, typename flowtype> inline void IBFSGraph<captype, tcaptype, flowtype>::add_edge(int nodeIndexFrom, int nodeIndexTo, captype capacity, captype reverseCapacity)
{
	arc *aFwd = arcLast;
	arcLast++;
	arc *aRev = arcLast;
	arcLast++;

	node* x = nodes + nodeIndexFrom;
	x->label++;
	node* y = nodes + nodeIndexTo;
	y->label++;

	aRev->sister = aFwd;
	aFwd->sister = aRev;
	aFwd->rCap = capacity;
	aFwd->cap = capacity;
	aRev->rCap = reverseCapacity;
	aRev->cap = reverseCapacity;
	aFwd->head = y;
	aRev->head = x;
	aFwd->u = nodeIndexFrom;
	aFwd->v = nodeIndexTo;
	aRev->u = nodeIndexTo;
	aRev->v = nodeIndexFrom;
}



template <typename captype, typename tcaptype, typename flowtype> inline typename IBFSGraph<captype, tcaptype, flowtype>::termtype IBFSGraph<captype, tcaptype, flowtype>::what_segment(int nodeIndex, termtype default_segm)
{
	if (nodes[nodeIndex].parent != NULL)
	{
		if (nodes[nodeIndex].label > 0)
		{
			return SOURCE;
		}
		else
		{
			return SINK;
		}
	}
	return default_segm;
}

template <typename captype, typename tcaptype, typename flowtype> inline int IBFSGraph<captype, tcaptype, flowtype>::add_node(int numNodes)
{
	int n = nNodes;
	nNodes += numNodes;
	return n;
}

#endif


