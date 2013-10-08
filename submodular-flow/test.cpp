#include <cstdio>
#include "ibfs.h"

typedef double flowtype;
typedef std::pair<int, int> edgeInfo;
typedef std::pair<edgeInfo, flowtype> flowInfo;
typedef std::vector<flowInfo> flowList;
typedef std::pair<flowtype, flowtype> STFlowInfo;
typedef std::vector<STFlowInfo> STFlowList;

int main(){
	IBFSGraph<flowtype, flowtype, flowtype> g(4,4);
	g.add_edge(0, 1, 2, 0);
	g.add_edge(1, 2, 2, 2);
	g.add_edge(1, 3, 2, 2);
	g.add_edge(2, 3, 1.5, 0);
	g.add_tweights(0, 4.5, 0);
	g.add_tweights(1, 0, 1);
	g.add_tweights(2, 1.6, 0);
	g.add_tweights(3, 0, 5);
	printf("Max Flow: %.2lf\n", g.maxflow());
	printf("S-T Cut:\n");
	printf("%d\n", g.what_segment(0));
	printf("%d\n", g.what_segment(1));
	printf("%d\n", g.what_segment(2));
	printf("%d\n", g.what_segment(3));
	printf("Internal Flow Value:\n");
	flowList flow = g.getFlow();
	for (int i = 0; i < flow.size(); ++i){
		printf("%d %d %.2lf\n", flow[i].first.first, flow[i].first.second, flow[i].second);	
	}
	printf("External Flow Value:\n");
	STFlowList STFlow = g.getSTFlow();
	for (int i = 0; i < STFlow.size(); ++i){
		printf("%d %.2lf %.2lf\n", i, STFlow[i].first, STFlow[i].second);	
	}
	return 0;
}
