/*
 * GPisMap - Online Continuous Mapping using Gaussian Process Implicit Surfaces
 * https://github.com/leebhoram/GPisMap
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License v3 as published by
 * the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of any FITNESS FOR A
 * PARTICULAR PURPOSE. See the GNU General Public License v3 for more details.
 *
 * You should have received a copy of the GNU General Public License v3
 * along with this program; if not, you can access it online at
 * http://www.gnu.org/licenses/gpl-3.0.html.
 *
 * Authors: Bhoram Lee <bhoram.lee@gmail.com>
 */

#include "quadtree.h"
#include <cmath>

constexpr double EPS = 1e-12;

double
sqdist(const Point<double> &pt1, const Point<double> &pt2) {
    double dx = (pt1.x - pt2.x);
    double dy = (pt1.y - pt2.y);
    return Eigen::Vector2d(dx, dy).squaredNorm();
    // double s = dx * dx + dy * dy;  // numerical instability
    // return s;
}

QuadTree::QuadTree(Point<double> c)
    : m_node_(nullptr),
      m_gp_(nullptr),
      m_leaf_(true),
      m_max_depth_reached_(false),
      m_root_limit_reached_(false),
      m_num_nodes_(0),
      m_north_west_(nullptr),
      m_north_east_(nullptr),
      m_south_west_(nullptr),
      m_south_east_(nullptr),
      m_tree_(nullptr) {
    m_boundary_ = Aabb(c, QuadTree::m_param_.initroot_halfleng);
}

QuadTree::QuadTree(Aabb boundary, QuadTree *const p)
    : m_node_(nullptr),
      m_gp_(nullptr),
      m_leaf_(true),
      m_max_depth_reached_(false),
      m_root_limit_reached_(false),
      m_num_nodes_(0),
      m_north_west_(nullptr),
      m_north_east_(nullptr),
      m_south_west_(nullptr),
      m_south_east_(nullptr),
      m_tree_(nullptr) {
    m_boundary_ = boundary;
    if (m_boundary_.GetHalfLength() < QuadTree::m_param_.min_halfleng) m_max_depth_reached_ = true;
    if (m_boundary_.GetHalfLength() > QuadTree::m_param_.max_halfleng) m_root_limit_reached_ = true;
    if (p != nullptr) m_tree_ = p;
}

QuadTree::QuadTree(Aabb boundary, QuadTree *const ch, int child_type)
    : m_node_(nullptr),
      m_gp_(nullptr),
      m_max_depth_reached_(false),
      m_root_limit_reached_(false),
      m_num_nodes_(0),
      m_north_west_(nullptr),
      m_north_east_(nullptr),
      m_south_west_(nullptr),
      m_south_east_(nullptr),
      m_tree_(nullptr) {
    m_boundary_ = boundary;
    if (m_boundary_.GetHalfLength() < QuadTree::m_param_.min_halfleng) m_max_depth_reached_ = true;
    if (m_boundary_.GetHalfLength() > QuadTree::m_param_.max_halfleng) m_root_limit_reached_ = true;
    if (child_type == 0) {
        m_leaf_ = true;
    } else {
        m_leaf_ = false;
        SubdivideExcept(child_type);
        if (child_type == m_child_type_nw_) m_north_west_ = ch;
        if (child_type == m_child_type_ne_) m_north_east_ = ch;
        if (child_type == m_child_type_sw_) m_south_west_ = ch;
        if (child_type == m_child_type_se_) m_south_east_ = ch;
    }
}

void
QuadTree::DeleteChildren() {
    if (m_north_west_) {
        delete m_north_west_;
        m_north_west_ = nullptr;
    }
    if (m_north_east_) {
        delete m_north_east_;
        m_north_east_ = nullptr;
    }
    if (m_south_west_) {
        delete m_south_west_;
        m_south_west_ = nullptr;
    }
    if (m_south_east_) {
        delete m_south_east_;
        m_south_east_ = nullptr;
    }
    m_leaf_ = true;
}

QuadTree *
QuadTree::GetRoot() {
    QuadTree *p = this;
    QuadTree *p1 = p->GetParent();
    while (p1 != nullptr) {
        p = p1;
        p1 = p->GetParent();
    }
    return p;
}

bool
QuadTree::InsertToParent(std::shared_ptr<Node> n) {
    double l = GetHalfLength();
    Point<double> c = GetCenter();

    // Find out what type the current node is
    const Point<double> np = n->GetPos();

    Point<double> par_c;
    int childType = 0;
    if (np.x < c.x && np.y > c.y) {
        childType = m_child_type_se_;
        par_c.x = c.x - l;
        par_c.y = c.y + l;
    }
    if (np.x > c.x && np.y > c.y) {
        childType = m_child_type_sw_;
        par_c.x = c.x + l;
        par_c.y = c.y + l;
    }
    if (np.x < c.x && np.y < c.y) {
        childType = m_child_type_ne_;
        par_c.x = c.x - l;
        par_c.y = c.y - l;
    }
    if (np.x > c.x && np.y < c.y) {
        childType = m_child_type_nw_;
        par_c.x = c.x + l;
        par_c.y = c.y - l;
    }

    Aabb parbb(par_c, double(2.0) * l);
    m_tree_ = new QuadTree(parbb, this, childType);
    return m_tree_->Insert(n);
}

bool
QuadTree::InsertToParent(std::shared_ptr<Node> n, std::unordered_set<QuadTree *> &quads) {
    double l = GetHalfLength();
    Point<double> c = GetCenter();

    // Find out what type the current node is
    const Point<double> np = n->GetPos();

    Point<double> par_c;
    int childType = 0;
    if (np.x < c.x && np.y > c.y) {
        childType = m_child_type_se_;
        par_c.x = c.x - l;
        par_c.y = c.y + l;
    }
    if (np.x > c.x && np.y > c.y) {
        childType = m_child_type_sw_;
        par_c.x = c.x + l;
        par_c.y = c.y + l;
    }
    if (np.x < c.x && np.y < c.y) {
        childType = m_child_type_ne_;
        par_c.x = c.x - l;
        par_c.y = c.y - l;
    }
    if (np.x > c.x && np.y < c.y) {
        childType = m_child_type_nw_;
        par_c.x = c.x + l;
        par_c.y = c.y - l;
    }

    Aabb parbb(par_c, double(2.0) * l);
    m_tree_ = new QuadTree(parbb, this, childType);
    return m_tree_->Insert(n, quads);
}

bool
QuadTree::Insert(std::shared_ptr<Node> n) {

    // Ignore objects that do not belong in this quad tree
    if (!m_boundary_.ContainsPoint(n->GetPos())) {
        if (GetParent() == nullptr) {
            if (m_root_limit_reached_) {
                return false;
            } else
                return InsertToParent(n);
        }
        return false;  // object cannot be added
    }

    if (m_max_depth_reached_) {
        if (m_node_ == nullptr) {  // If this is the first point in this quad tree, add the object here
            m_node_ = n;
            m_num_nodes_ = 1;
            return true;
        } else  // no more points accepted at this resolution
            return false;
    }

    if (IsLeaf()) {

        if (m_boundary_.GetHalfLength() > QuadTree::m_param_.cluster_halfleng) {
            Subdivide();  // always Insert to the possibly smaller tree
        } else {
            if (m_node_ == nullptr)
            // If this is the first point in this quad tree, add the object here
            {
                m_node_ = n;
                m_num_nodes_ = 1;
                return true;
            }

            // Otherwise, subdivide and then add the point to whichever node will accept it
            //numNodes = 0;
            if (sqdist(m_node_->GetPos(), n->GetPos()) < QuadTree::m_param_.min_halfleng_sqr) { return false; }

            Subdivide();
            if (m_north_west_->Insert(m_node_)) {
                ;
            } else if (m_north_east_->Insert(m_node_)) {
                ;
            } else if (m_south_west_->Insert(m_node_)) {
                ;
            } else if (m_south_east_->Insert(m_node_)) {
                ;
            }
            m_node_ = nullptr;
        }
    }

    if (m_north_west_->Insert(n)) {
        UpdateCount();
        return true;
    }
    if (m_north_east_->Insert(n)) {
        UpdateCount();
        return true;
    }
    if (m_south_west_->Insert(n)) {
        UpdateCount();
        return true;
    }
    if (m_south_east_->Insert(n)) {
        UpdateCount();
        return true;
    }

    return false;
}

bool
QuadTree::Insert(std::shared_ptr<Node> n, std::unordered_set<QuadTree *> &quads) {
    // Ignore objects that do not belong in this quad tree
    if (!m_boundary_.ContainsPoint(n->GetPos())) {
        if (GetParent() == nullptr) {
            // FIXME: when InsertToParent is called, if inserting the node successfully, quads is not updated
            if (m_root_limit_reached_) {
                return false;
            } else
                return InsertToParent(n, quads);  // FIXED
        }
        return false;  // object cannot be added
    }

    if (m_max_depth_reached_) {
        if (m_node_ == nullptr) {  // If this is the first point in this quad tree, add the object here
            m_node_ = n;
            m_num_nodes_ = 1;
            if (std::fabs(GetHalfLength() - QuadTree::m_param_.cluster_halfleng) < 1e-3) quads.insert(this);
            return true;
        } else  // no more points accepted at this resolution
            return false;
    }

    if (IsLeaf()) {

        if (m_boundary_.GetHalfLength() > QuadTree::m_param_.cluster_halfleng) {
            Subdivide();
        } else {
            if (m_node_ == nullptr)
            // If this is the first point in this quad tree, add the object here
            {
                m_node_ = n;
                m_num_nodes_ = 1;
                if (fabs(GetHalfLength() - QuadTree::m_param_.cluster_halfleng) < 1e-3) quads.insert(this);
                return true;
            }

            // Otherwise, subdivide and then add the point to whichever node will accept it
            if (sqdist(m_node_->GetPos(), n->GetPos()) < QuadTree::m_param_.min_halfleng_sqr) { return false; }

            Subdivide();
            if (m_north_west_->Insert(m_node_, quads)) {
                ;
            } else if (m_north_east_->Insert(m_node_, quads)) {
                ;
            } else if (m_south_west_->Insert(m_node_, quads)) {
                ;
            } else if (m_south_east_->Insert(m_node_, quads)) {
                ;
            }
            m_node_ = nullptr;
        }
    }

    if (m_north_west_->Insert(n, quads)) {
        if (fabs(GetHalfLength() - QuadTree::m_param_.cluster_halfleng) < 1e-3) quads.insert(this);
        UpdateCount();
        return true;
    }

    if (m_north_east_->Insert(n, quads)) {
        if (fabs(GetHalfLength() - QuadTree::m_param_.cluster_halfleng) < 1e-3) quads.insert(this);
        UpdateCount();
        return true;
    }

    if (m_south_west_->Insert(n, quads)) {
        if (fabs(GetHalfLength() - QuadTree::m_param_.cluster_halfleng) < 1e-3) quads.insert(this);
        UpdateCount();
        return true;
    }

    if (m_south_east_->Insert(n, quads)) {
        if (fabs(GetHalfLength() - QuadTree::m_param_.cluster_halfleng) < 1e-3) quads.insert(this);
        UpdateCount();
        return true;
    }

    return false;
}

void
QuadTree::UpdateCount() {
    if (m_leaf_ == false) {
        m_num_nodes_ = 0;
        m_num_nodes_ += m_north_west_->GetNodeCount();
        m_num_nodes_ += m_north_east_->GetNodeCount();
        m_num_nodes_ += m_south_west_->GetNodeCount();
        m_num_nodes_ += m_south_east_->GetNodeCount();
    }
}

bool
QuadTree::IsNotNew(std::shared_ptr<Node> n) {
    if (!m_boundary_.ContainsPoint(n->GetPos())) {
        return false;  // object cannot be added
    }

    if (IsEmptyLeaf()) return false;

    if (!IsEmpty() && (sqdist(m_node_->GetPos(), n->GetPos()) < QuadTree::m_param_.min_halfleng_sqr)) { return true; }

    if (IsLeaf()) return false;

    if (m_north_west_->IsNotNew(n)) return true;
    if (m_north_east_->IsNotNew(n)) return true;
    if (m_south_west_->IsNotNew(n)) return true;
    if (m_south_east_->IsNotNew(n)) return true;

    return false;
}

bool
QuadTree::Remove(std::shared_ptr<Node> n) {
    // Ignore objects that do not belong in this quad tree
    if (!m_boundary_.ContainsPoint(n->GetPos())) {
        return false;  // object cannot be added
    }

    if (IsEmptyLeaf()) return false;

    if (!IsEmpty() && (sqdist(m_node_->GetPos(), n->GetPos()) < EPS)) {
        m_node_ = nullptr;
        m_num_nodes_ = 0;
        return true;
    }

    if (IsLeaf()) return false;

    bool res = m_north_west_->Remove(n);
    res |= m_north_east_->Remove(n);
    res |= m_south_west_->Remove(n);
    res |= m_south_east_->Remove(n);

    if (res) {
        bool res2 = m_north_west_->IsEmptyLeaf();
        res2 &= m_north_east_->IsEmptyLeaf();
        res2 &= m_south_west_->IsEmptyLeaf();
        res2 &= m_south_east_->IsEmptyLeaf();
        if (res2) {
            DeleteChildren();
            m_leaf_ = true;
            m_num_nodes_ = 0;
        }
    }
    UpdateCount();

    return res;
}

bool
QuadTree::Remove(std::shared_ptr<Node> n, std::unordered_set<QuadTree *> &quads) {
    // Ignore objects that do not belong in this quad tree
    if (!m_boundary_.ContainsPoint(n->GetPos())) {
        return false;  // object cannot be added
    }

    if (IsEmptyLeaf()) return false;

    if (!IsEmpty() && (sqdist(m_node_->GetPos(), n->GetPos()) < EPS)) {
        m_node_ = nullptr;
        m_num_nodes_ = 0;
        return true;
    }

    if (IsLeaf()) return false;

    bool res = m_north_west_->Remove(n, quads);
    if (!res) res |= m_north_east_->Remove(n, quads);
    if (!res) res |= m_south_west_->Remove(n, quads);
    if (!res) res |= m_south_east_->Remove(n, quads);

    if (res) {
        bool res2 = m_north_west_->IsEmptyLeaf();
        res2 &= m_north_east_->IsEmptyLeaf();
        res2 &= m_south_west_->IsEmptyLeaf();
        res2 &= m_south_east_->IsEmptyLeaf();
        if (res2) {
            quads.erase(m_north_west_);
            quads.erase(m_north_east_);
            quads.erase(m_south_west_);
            quads.erase(m_south_east_);
            DeleteChildren();
            m_leaf_ = true;
            m_num_nodes_ = 0;
        }
    }
    UpdateCount();

    return res;
}

void
QuadTree::CollectTrees(bool (*qualify)(const QuadTree *tree), std::vector<QuadTree *> &trees) {  // NOLINT(misc-no-recursion)
    if (qualify(this)) { trees.push_back(this); }

    if (IsLeaf()) { return; }

    m_north_west_->CollectTrees(qualify, trees);
    m_north_east_->CollectTrees(qualify, trees);
    m_south_west_->CollectTrees(qualify, trees);
    m_south_east_->CollectTrees(qualify, trees);
}

void
QuadTree::Update(std::shared_ptr<OnGpis> gp) {
    m_gp_ = gp;
}

bool
QuadTree::Update(std::shared_ptr<Node> n) {
    // Ignore objects that do not belong in this quad tree
    if (!m_boundary_.ContainsPoint(n->GetPos())) {
        return false;  // object cannot be added
    }

    if (IsEmptyLeaf()) return false;

    if (!IsEmpty() && (sqdist(m_node_->GetPos(), n->GetPos()) < EPS)) {
        m_node_ = n;
        return true;
    }

    if (IsLeaf()) return false;

    if (m_north_west_->Update(n)) return true;
    if (m_north_east_->Update(n)) return true;
    if (m_south_west_->Update(n)) return true;
    if (m_south_east_->Update(n)) return true;

    return false;
}

bool
QuadTree::Update(std::shared_ptr<Node> n, std::unordered_set<QuadTree *> &quads) {
    // Ignore objects that do not belong in this quad tree
    if (!m_boundary_.ContainsPoint(n->GetPos())) {
        return false;  // object cannot be added
    }

    if (IsEmptyLeaf()) return false;

    if (!IsEmpty() && (sqdist(m_node_->GetPos(), n->GetPos()) < EPS)) {
        m_node_ = n;
        if (fabs(GetHalfLength() - QuadTree::m_param_.cluster_halfleng) < 1e-3) quads.insert(this);
        return true;
    }

    if (IsLeaf()) return false;

    if (m_north_west_->Update(n, quads)) {
        if (fabs(GetHalfLength() - QuadTree::m_param_.cluster_halfleng) < 1e-3) quads.insert(this);
        return true;
    }
    if (m_north_east_->Update(n, quads)) {
        if (fabs(GetHalfLength() - QuadTree::m_param_.cluster_halfleng) < 1e-3) quads.insert(this);
        return true;
    }
    if (m_south_west_->Update(n, quads)) {
        if (fabs(GetHalfLength() - QuadTree::m_param_.cluster_halfleng) < 1e-3) quads.insert(this);
        return true;
    }
    if (m_south_east_->Update(n, quads)) {
        if (fabs(GetHalfLength() - QuadTree::m_param_.cluster_halfleng) < 1e-3) quads.insert(this);
        return true;
    }

    return false;
}

void
QuadTree::Subdivide() {
    double l = m_boundary_.GetHalfLength() * 0.5;
    Point<double> c = m_boundary_.GetCenter();
    Point<double> nw_c = Point<double>(c.x - l, c.y + l);
    Aabb nw(nw_c, l);
    m_north_west_ = new QuadTree(nw, this);

    Point<double> ne_c = Point<double>(c.x + l, c.y + l);
    Aabb ne(ne_c, l);
    m_north_east_ = new QuadTree(ne, this);

    Point<double> sw_c = Point<double>(c.x - l, c.y - l);
    Aabb sw(sw_c, l);
    m_south_west_ = new QuadTree(sw, this);

    Point<double> se_c = Point<double>(c.x + l, c.y - l);
    Aabb se(se_c, l);
    m_south_east_ = new QuadTree(se, this);

    m_leaf_ = false;
}

void
QuadTree::SubdivideExcept(int child_type) {
    double l = m_boundary_.GetHalfLength() * 0.5;
    Point<double> c = m_boundary_.GetCenter();

    if (child_type != m_child_type_nw_) {
        Point<double> nw_c = Point<double>(c.x - l, c.y + l);
        Aabb nw(nw_c, l);
        m_north_west_ = new QuadTree(nw, this);
    }

    if (child_type != m_child_type_ne_) {
        Point<double> ne_c = Point<double>(c.x + l, c.y + l);
        Aabb ne(ne_c, l);
        m_north_east_ = new QuadTree(ne, this);
    }

    if (child_type != m_child_type_sw_) {
        Point<double> sw_c = Point<double>(c.x - l, c.y - l);
        Aabb sw(sw_c, l);
        m_south_west_ = new QuadTree(sw, this);
    }

    if (child_type != m_child_type_se_) {
        Point<double> se_c = Point<double>(c.x + l, c.y - l);
        Aabb se(se_c, l);
        m_south_east_ = new QuadTree(se, this);
    }

    m_leaf_ = false;
}

// Find all points that appear within a range
void
QuadTree::QueryRange(Aabb range, std::vector<std::shared_ptr<Node>> &nodes) {
    // Automatically abort if the range does not intersect this quad
    if (!m_boundary_.IntersectsAabb(range) || IsEmptyLeaf()) {
        return;  // empty list
    }

    // Check objects at this quad level
    if (IsLeaf()) {
        // FIXME: why within the inscribed circle instead of the bbox?
        if (range.ContainsPoint(m_node_->GetPos())) {
            // if (sqdist(node->GetPos(), range.GetCenter()) <  range.GetHalfLengthSq()){
            nodes.push_back(m_node_);
        }
        return;
    }

    // Otherwise, add the points from the children
    m_north_west_->QueryRange(range, nodes);
    m_north_east_->QueryRange(range, nodes);
    m_south_west_->QueryRange(range, nodes);
    m_south_east_->QueryRange(range, nodes);
}

void
QuadTree::GetAllChildrenNonEmptyNodes(std::vector<std::shared_ptr<Node>> &nodes) {
    if (IsEmptyLeaf()) return;

    if (IsLeaf()) {
        nodes.push_back(m_node_);
        return;
    }

    m_north_west_->GetAllChildrenNonEmptyNodes(nodes);
    m_north_east_->GetAllChildrenNonEmptyNodes(nodes);
    m_south_west_->GetAllChildrenNonEmptyNodes(nodes);
    m_south_east_->GetAllChildrenNonEmptyNodes(nodes);
    return;
}

void
QuadTree::QueryNonEmptyLevelC(Aabb range, std::vector<QuadTree *> &quads) {
    // Automatically abort if the range does not intersect this quad
    if (!m_boundary_.IntersectsAabb(range) || IsEmptyLeaf()) {
        return;  // empty list
    }

    // FIXED
    if (IsLeaf()) {  // no children, but non-empty
        quads.push_back(this);
    } else {
        if (m_boundary_.GetHalfLength() > QuadTree::m_param_.cluster_halfleng) {
            // visit children to find the cluster resolution level
            for (auto child: {m_north_west_, m_north_east_, m_south_west_, m_south_east_}) {
                if (child != nullptr) { child->QueryNonEmptyLevelC(range, quads); }
            }
        } else {
            // reach the cluster resolution level, stop here
            // there are sub-resolution levels, they will be traversed by
            quads.push_back(this);
        }
    }
}

void
QuadTree::QueryNonEmptyLevelC(Aabb range, std::vector<QuadTree *> &quads, std::vector<double> &sqdst) {

    // Automatically abort if the range does not intersect this quad
    if (!m_boundary_.IntersectsAabb(range) || IsEmptyLeaf()) {
        return;  // empty list
    }

    // if (IsLeaf()){ // no children
    //     if (boundary.GetHalfLength() > QuadTree::m_param_.cluster_halfleng + 0.001){
    //         // FIXME: never reached, if it is a non-empty leaf, this if-statement is never true, check
    //         // the Insert method
    //         return; // although this is not empty, it is not representative enough
    //     }
    // }

    // // FIXME: why add 0.001 ?
    // if (boundary.GetHalfLength() > QuadTree::m_param_.cluster_halfleng+0.001){
    //     // Otherwise, add the points from the children
    //     northWest->QueryNonEmptyLevelC(range,quads,sqdst);
    //     northEast->QueryNonEmptyLevelC(range,quads,sqdst);
    //     southWest->QueryNonEmptyLevelC(range,quads,sqdst);
    //     southEast->QueryNonEmptyLevelC(range,quads,sqdst);
    // }
    // else
    // {   //  a non-empty leaf
    //     // or a non-leaf whose size if smaller than cluster_halfleng
    //     sqdst.push_back(sqdist(GetCenter(),range.GetCenter()));
    //     quads.push_back(this);
    // }

    // FIXED
    if (IsLeaf()) {  // no children, but non-empty
        sqdst.push_back(sqdist(GetCenter(), range.GetCenter()));
        quads.push_back(this);
    } else {
        if (m_boundary_.GetHalfLength() > QuadTree::m_param_.cluster_halfleng) {
            // visit children to find the cluster resolution level
            for (auto child: {m_north_west_, m_north_east_, m_south_west_, m_south_east_}) {
                if (child != nullptr) { child->QueryNonEmptyLevelC(range, quads, sqdst); }
            }
        } else {
            // reach the cluster resolution level, stop here
            // there are sub-resolution levels, they will be traversed by
            // GetAllChildrenNonEmptyNodes
            sqdst.push_back(sqdist(GetCenter(), range.GetCenter()));
            quads.push_back(this);
        }
    }
}

std::ostream &
operator<<(std::ostream &os, QuadTree *a_tree) {
    static std::vector<std::string> indents;
    static auto printIndent = [&]() -> std::string {
        std::stringstream ss;
        for (auto &indent: indents) { ss << indent; }
        return ss.str();
    };
    static std::vector<QuadTree *> treeStack;
    static std::vector<int> levelStack;

    int NW = 0;
    int NE = 1;
    int SW = 2;
    int SE = 3;
    int ROOT = 4;
    static std::vector<int> childTypeStack;
    const char *ChildTypeNames[5] = {"kNorthWest", "kNorthEast", "kSouthWest", "kSouthEast", "kRoot"};

    if (a_tree == nullptr) {
        std::cout << "null";
        return os;
    }

    int level = 0;
    int childType = ROOT;

    indents.clear();
    treeStack.push_back(a_tree);
    levelStack.push_back(level);
    childTypeStack.push_back(childType);

    while (!treeStack.empty()) {
        auto tree = treeStack.back();
        treeStack.pop_back();
        level = levelStack.back();
        levelStack.pop_back();
        childType = childTypeStack.back();
        childTypeStack.pop_back();

        std::string name = ChildTypeNames[childType];
        std::vector<std::shared_ptr<Node>> nodes;
        tree->GetAllChildrenNonEmptyNodes(nodes);

        auto c = tree->GetCenter();
        os << printIndent() << name << ": center = [" << c.x << ", " << c.y << "], half size = " << tree->GetHalfLength() << ", is leaf: " << tree->IsLeaf()
           << ", has SurfaceData: " << (tree->m_gp_ != nullptr) << ", number of nodes = " << nodes.size() << std::endl;

        // print m_nodes_
        if (!tree->IsEmpty()) {
            indents.pop_back();
            if (childType == SE) {
                indents.emplace_back("    ");
            } else {
                indents.emplace_back("│   ");
            }
            auto n = tree->m_node_;
            auto np = n->GetPos();
            os << printIndent() << "└──* kSurface Node" << std::setw(2) << 0 << ": position = [" << np.x << ", " << np.y << ']' << std::endl;
            os << printIndent() << "    └──* distance = " << n->m_val_ << ", gradient = [" << n->m_grad_.x << ", " << n->m_grad_.y
               << "], var_position = " << n->m_pose_sig_ << ", var_gradient = " << n->m_grad_sig_ << std::endl;
            indents.pop_back();
            indents.emplace_back("├── ");
        }

        // add child m_nodes_
        if (!tree->IsLeaf()) {
            level++;

            treeStack.push_back(tree->m_south_east_);
            levelStack.push_back(level);
            childTypeStack.push_back(SE);

            treeStack.push_back(tree->m_south_west_);
            levelStack.push_back(level);
            childTypeStack.push_back(SW);

            treeStack.push_back(tree->m_north_east_);
            levelStack.push_back(level);
            childTypeStack.push_back(NE);

            treeStack.push_back(tree->m_north_west_);
            levelStack.push_back(level);
            childTypeStack.push_back(NW);

            if (!indents.empty()) { indents.pop_back(); }
            if (childType == SE) {
                indents.emplace_back("    ");
            } else if (level > 1) {
                indents.emplace_back("│   ");
            }
            indents.emplace_back("├── ");
        }

        if (!levelStack.empty() && level > levelStack.back()) {
            auto n = level - levelStack.back();
            for (int i = 0; i <= n; ++i) { indents.pop_back(); }
            indents.emplace_back("├── ");
        }

        if ((!treeStack.empty()) && (childTypeStack.back() == SE)) {
            indents.pop_back();
            indents.emplace_back("└── ");
        }
    }

    return os;
}
