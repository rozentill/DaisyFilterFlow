#include <set>
#include <vector>
#include <cstdlib>
using std::set;

class GraphStructure
{
public:
	std::vector<set<int>> adjList;
	int vertexNum;
	GraphStructure();
	GraphStructure(int num);
	~GraphStructure();
	void ReserveSpace(int num);
	void SetVertexNum(int vNum);
	void AddEdge(int s, int e);

	void DeleteEdge(int s, int e);
	void DeleteAllEdge(int s);
};
