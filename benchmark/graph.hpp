#ifndef __GCGRAPH_HPP__
#define __GCGRAPH_HPP__

#include <vector>
#include <queue>
#include <algorithm>
#include <limits>
#include <iostream>

struct GraphNode;
struct GraphArc;

typedef int FlowType;

struct GraphArc {
    GraphNode* head; //destination node
    GraphArc* sister; //reverse edge
    FlowType r_cap; //residual capacity
    GraphArc* next; //next arc in the adjacency list
};

struct GraphNode {
    GraphArc* first; //first arc in the adjacency list
    int height;
    FlowType excess;
    bool reachable; // used for inSourceSegment
};

template <typename TWeight>
class GCGraph {
public:
    GCGraph() : flow(0) {}


    int addVtx() {
        nodes.push_back(GraphNode{nullptr, 0, 0, false});
        return static_cast<int>(nodes.size()) - 1;
    }

    void addEdges(int from, int to, TWeight cap, TWeight rev_cap) {
        GraphArc a = { &nodes[to], nullptr, cap , nullptr };
        GraphArc a_rev = { &nodes[from], nullptr, rev_cap , nullptr };
        arcs.push_back(a);
        arcs.push_back(a_rev);

        GraphArc* forwardArc = &arcs[arcs.size() - 2];
        GraphArc* backwardArc = &arcs[arcs.size() - 1];
        forwardArc->sister = backwardArc;
        backwardArc->sister = forwardArc;

        forwardArc->next = nodes[from].first; 
        nodes[from].first = forwardArc;

        backwardArc->next = nodes[to].first;
        nodes[to].first = backwardArc;
    }

    void addTermWeights(int i, TWeight source_cap, TWeight sink_cap) {
        addEdges(SOURCE, i, source_cap, 0);
        addEdges(i, SINK, sink_cap, 0);
    }

    void create(int node_count, int edge_count) {
        nodes.clear();
        arcs.clear();
        nodes.reserve(node_count + 2); // +2 for SOURCE and SINK
        arcs.reserve(2 * edge_count + 4 * node_count); // estimate with safety margin

        // Add SOURCE and SINK first
        nodes.push_back(GraphNode{nullptr, 0, 0, false}); // SOURCE (index 0)
        nodes.push_back(GraphNode{nullptr, 0, 0, false}); // SINK (index 1)

        // for (int i = 0; i < node_count; ++i) {
        //     addVtx();
        // }

        flow = 0;
    }

    TWeight maxFlow() {
        std::cout << "max flow called\n";
        initialize_preflow();

        std::cout << "initialize preflow called\n";

        std::queue<int> active;
        for (int i = 0; i < nodes.size(); ++i) {
            if (i != SOURCE && i != SINK && nodes[i].excess > 0) {
                active.push(i);
            }
        }

        std::cout << "active queue initialized\n";

        while (!active.empty()) {
            int u = active.front();
            active.pop();
            if (discharge(u)) {
                active.push(u);
            }
        }

        markSourceSegment();
        return flow;
    }

    bool inSourceSegment(int i) const {
        return nodes[i].reachable;
    }

private:
    std::vector<GraphNode> nodes;
    std::vector<GraphArc> arcs;
    FlowType flow;

    const int SOURCE = 0;
    const int SINK = 1;

    void initialize_preflow() {
        for (auto& node : nodes) {
            node.height = 0;
            node.excess = 0;
            node.reachable = false;
        }
        nodes[SOURCE].height = nodes.size();
        std::cout << "source height set\n";

        GraphArc* arc = nodes[SOURCE].first;
        while (arc != nullptr) {
            FlowType cap = arc->r_cap;

            if (cap > 0) {
                arc->r_cap = 0;
                arc->sister->r_cap += cap;
                arc->head->excess += cap;
                nodes[SOURCE].excess -= cap;
            }
            arc = arc->next;
        }

        std::cout << "source excess set\n";
    }

    bool discharge(int u) {
        auto& node = nodes[u];
        while (node.excess > 0) {
            GraphArc* arc = node.first;
            bool pushed = false;
            while (arc) {
                if (arc->r_cap > 0 && node.height == arc->head->height + 1) {
                    push(node, *arc);
                    pushed = true;
                    if (node.excess == 0) {
                        return false;
                    }
                }
                arc = arc->next;
            }

            if (!pushed) {
                relabel(u);
            }
        }
        return node.excess > 0;
    }

    void push(GraphNode& u, GraphArc& arc) {
        FlowType send = std::min(u.excess, arc.r_cap);
        arc.r_cap -= send;
        arc.sister->r_cap += send;
        arc.head->excess += send;
        u.excess -= send;
        if (arc.head == &nodes[SINK] && send > 0) {
            flow += send;
        }
    }

    void relabel(int u) {
        int min_height = std::numeric_limits<int>::max();
        for (GraphArc* arc = nodes[u].first; arc; arc = arc->next) {
            if (arc->r_cap > 0) {
                min_height = std::min(min_height, arc->head->height);
            }
        }
        if (min_height < std::numeric_limits<int>::max()) {
            nodes[u].height = min_height + 1;
        }
    }

    void markSourceSegment() {
        std::cout << "mark source segment called\n";
        std::queue<int> q;
        q.push(SOURCE);
        nodes[SOURCE].reachable = true;
        
        std::cout << "SOURCE reachable: " << nodes[SOURCE].reachable << "\n";

        while (!q.empty()) {
            int u = q.front(); 
            q.pop();
            GraphArc* arc = nodes[u].first;

            while (arc != nullptr) {
                int v = static_cast<int>(arc->head - &nodes[0]);
            
                if (arc->r_cap > 0 && !nodes[v].reachable) {
                    nodes[v].reachable = true;
                    std::cout << "node " << v << " reachable\n";
                    q.push(v);
                }
                arc = arc->next;
            }
        }
    }
};

#endif

