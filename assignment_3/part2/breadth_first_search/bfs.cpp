#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(Graph g, vertex_set *frontier, vertex_set *new_frontier, int *distances)
{   
    #pragma omp parallel
    {
        // local count
        int new_frontier_count = 0;
        // local frontier [] = Vertex array, array size = g->num_nodes;
        Vertex *new_frontier_temp = new Vertex[g->num_nodes];

        #pragma omp for schedule(dynamic, 1024)
        for (int i = 0; i < frontier->count; i++)
        {
            
            // 從 frontier 中的每個 node search 其所有 neighbor
            int node = frontier->vertices[i];

            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1) ? g->num_edges : g->outgoing_starts[node + 1];

            // attempt to add all neighbors to the new frontier
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                int outgoing = g->outgoing_edges[neighbor];

                if (distances[outgoing] == NOT_VISITED_MARKER)
                {
                    if(__sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, distances[node] + 1)){
                        // distances[outgoing] = distances[node] + 1;
                        // int index = new_frontier->count++;
                        // new_frontier->vertices[new_frontier_count++] = outgoing;
                        new_frontier_temp[new_frontier_count++] = outgoing;
                    }
                }
            }
        }

        #pragma omp critical
        {
            // 將 local frontier copy 至起始位置在 new_frontier->vertices 的 end，要 copy new_frontier_count 的大小
            memcpy(new_frontier->vertices + new_frontier->count, new_frontier_temp, sizeof(int) * new_frontier_count);
            // 更新 new_frontier->count，已 copy 的數量
            new_frontier->count += new_frontier_count;
        }

        free(new_frontier_temp);
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    #pragma omp parallel for
    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

// Execute one step of the "bottom-up" BFS. For each unvisited node,
// check if it connects back to any nodes on the frontier. If so, add
// it to the new frontier.
void bottom_up_step(Graph g, vertex_set *frontier, vertex_set *new_frontier, int *distances, int current_level)
{
    #pragma omp parallel
    {
        int new_frontier_count = 0;
        Vertex *new_frontier_temp = new Vertex[g->num_nodes];

        #pragma omp for schedule(dynamic, 1024)
        for (int node = 0; node < g->num_nodes; node++)
        {
            if (distances[node] == NOT_VISITED_MARKER)
            {
                int start_edge = g->incoming_starts[node];
                int end_edge = (node == g->num_nodes - 1) ? g->num_edges : g->incoming_starts[node + 1];

                // 檢查每個 node 是否有 neighbor 在 frontier 中
                for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
                {
                    int incoming = g->incoming_edges[neighbor];
                    
                    // if neighbor node 在 current frontier 中，則
                    if (distances[incoming] == current_level)
                    {
                        // 將 node 加到 new_frontier，update distance = current level + 1
                        new_frontier_temp[new_frontier_count++] = node;
                        distances[node] = current_level + 1;
                        break;
                    }
                }
            }
        }

        #pragma omp critical
        {
            memcpy(new_frontier->vertices + new_frontier->count, new_frontier_temp, sizeof(int) * new_frontier_count);
            new_frontier->count += new_frontier_count;
        }

        free(new_frontier_temp);
    }
}



void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // Initialize the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    int current_level = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);
        bottom_up_step(graph, frontier, new_frontier, sol->distances, current_level);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        current_level++;

        // Swap the frontiers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // Initialize the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    int current_level = 0;
    const int THRESHOLD = graph->num_nodes / 10;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        // 如果 frontier 比較小，需要 search 的 node 數不多 --> top_down
        if(frontier->count < THRESHOLD)
        {
            top_down_step(graph, frontier, new_frontier, sol->distances);
        }
        // 如果 frontier 比較大，反之
        else
        {
            bottom_up_step(graph, frontier, new_frontier, sol->distances, current_level);
        }

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        current_level++;

        // Swap the frontiers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}
