#include "CommonDataStructure.cuh"

#pragma region GraphStructure_Part

GraphStructure::GraphStructure()
{
	vertexNum = 0;
	adjList.clear();
}

GraphStructure::GraphStructure(int num)
{
	vertexNum = 0;
	adjList.clear();
	ReserveSpace(num);
}

GraphStructure::~GraphStructure()
{

}

void GraphStructure::ReserveSpace(int num)
{
	adjList.reserve(num);
}

void GraphStructure::SetVertexNum(int vNum)
{
	vertexNum = vNum;
	adjList.resize(vertexNum);
}

void GraphStructure::AddEdge(int s, int e)
{
	adjList[s].insert(e);
}

void GraphStructure::DeleteEdge(int s, int e)
{
	adjList[s].erase(adjList[s].find(e));
}

void GraphStructure::DeleteAllEdge(int s)
{
	adjList[s].clear();
}

#pragma endregion