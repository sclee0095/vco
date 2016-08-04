#pragma once

#include "std_include.h"
//#include "win_util.h"
//#include <cstdint>
#include <queue>
//#include <list>


class cGeodesic
{
public:
	cGeodesic();
	~cGeodesic();

	
};

//class OverSegmentation;
//struct Node {
//	explicit Node(int to = 0, float w = 0);
//	int to;
//	float w;
//	bool operator<(const Node & o) const;
//	bool operator>(const Node & o) const;
//};
//struct Node2 {
//	explicit Node2(int to = 0, int from = 0, float w = 0);
//	int to, from;
//	float w;
//	bool operator<(const Node2 & o) const;
//	bool operator>(const Node2 & o) const;
//};
//
//struct GeodesicDistance {
//protected:
//	typedef std::priority_queue< Node, std::vector<Node >, std::greater<Node > > PQ;
//	typedef std::priority_queue< Node2, std::vector<Node2>, std::greater<Node2> > PQ2;
//	virtual void updatePQ(VectorXf & d, PQ & q) const;
//	virtual VectorXf backPropGradientPQ(PQ2 & q, const VectorXf & d) const;
//
//	const int N_;
//	VectorXf d_;
//	std::vector< std::vector< Node > > adj_list_;
//public:
//	// Geodesic distance functions
//	GeodesicDistance(const OverSegmentation & os);
//	GeodesicDistance(const Edges & edges, const VectorXf & edge_weights);
//
//	virtual GeodesicDistance& reset(float v = 1e10);
//	virtual GeodesicDistance& update(int nd, float v = 0);
//	virtual GeodesicDistance& update(const VectorXf & new_min);
//	virtual VectorXf compute(int nd) const;
//	virtual VectorXf compute(const VectorXf & start) const;
//	virtual RMatrixXf compute(const RMatrixXf & start) const;
//	virtual VectorXf backPropGradient(int nd, const VectorXf & g) const;
//	virtual VectorXf backPropGradient(const VectorXf & start, const VectorXf & g) const;
//	virtual int N() const;
//	virtual const VectorXf & d() const;
//};
//
//int geodesicCenter(const Edges & edges, const VectorXf & edge_weights);
//
