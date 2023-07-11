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

#pragma once

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <unordered_set>
#include <vector>

#include "OnGpis.h"
#include "strct.h"

class Aabb {
    Point<double> m_center_;
    double m_half_length_;
    double m_half_length_sq_;
    double m_xmin_;
    double m_xmax_;
    double m_ymin_;
    double m_ymax_;

    Point<double> m_pt_nw_;
    Point<double> m_pt_ne_;
    Point<double> m_pt_sw_;
    Point<double> m_pt_se_;

public:
    Aabb() {
        m_half_length_ = double(0.0);
        m_half_length_sq_ = double(0.0);
        m_xmin_ = double(0.0);
        m_xmax_ = double(0.0);
        m_ymin_ = double(0.0);
        m_ymax_ = double(0.0);
    }

    Aabb(Point<double> center, double half_length) {
        m_center_ = center;
        m_half_length_ = half_length;
        m_half_length_sq_ = m_half_length_ * m_half_length_;
        m_xmin_ = m_center_.x - m_half_length_;
        m_xmax_ = m_center_.x + m_half_length_;
        m_ymin_ = m_center_.y - m_half_length_;
        m_ymax_ = m_center_.y + m_half_length_;
        m_pt_nw_ = Point<double>(m_xmin_, m_ymax_);
        m_pt_ne_ = Point<double>(m_xmax_, m_ymax_);
        m_pt_sw_ = Point<double>(m_xmin_, m_ymin_);
        m_pt_se_ = Point<double>(m_xmax_, m_ymin_);
    }

    Aabb(double x, double y, double half_length) {
        m_center_ = Point<double>(x, y);
        m_half_length_ = half_length;
        m_half_length_sq_ = m_half_length_ * m_half_length_;
        m_xmin_ = m_center_.x - m_half_length_;
        m_xmax_ = m_center_.x + m_half_length_;
        m_ymin_ = m_center_.y - m_half_length_;
        m_ymax_ = m_center_.y + m_half_length_;
        m_pt_nw_ = Point<double>(m_xmin_, m_ymax_);
        m_pt_ne_ = Point<double>(m_xmax_, m_ymax_);
        m_pt_sw_ = Point<double>(m_xmin_, m_ymin_);
        m_pt_se_ = Point<double>(m_xmax_, m_ymin_);
    }

    Point<double>
    GetCenter() {
        return m_center_;
    }

    [[nodiscard]] double
    GetHalfLength() const {
        return m_half_length_;
    }

    [[nodiscard]] double
    GetHalfLengthSq() const {
        return m_half_length_sq_;
    }

    [[nodiscard]] double
    GetXMinBound() const {
        return m_xmin_;
    }

    [[nodiscard]] double
    GetXMaxBound() const {
        return m_xmax_;
    }

    [[nodiscard]] double
    GetYMinBound() const {
        return m_ymin_;
    }

    [[nodiscard]] double
    GetYMaxBound() const {
        return m_ymax_;
    }

    const Point<double> &
    GetNw() {
        return m_pt_nw_;
    }

    const Point<double> &
    GetNe() {
        return m_pt_ne_;
    }

    const Point<double> &
    GetSw() {
        return m_pt_sw_;
    }

    const Point<double> &
    GetSe() {
        return m_pt_se_;
    }

    [[nodiscard]] bool
    ContainsPoint(Point<double> pt) const {
        return ((pt.x >= m_xmin_) && (pt.x <= m_xmax_) && (pt.y >= m_ymin_) && (pt.y <= m_ymax_));
    }

    [[nodiscard]] bool
    IntersectsAabb(Aabb aabb) const {
        return !((aabb.GetXMaxBound() < m_xmin_) || (aabb.GetXMinBound() > m_xmax_) || (aabb.GetYMaxBound() < m_ymin_) || (aabb.GetYMinBound() > m_ymax_));
    }
};

class QuadTree {
#if defined(BUILD_TEST)
public:
#endif
    // Arbitrary constant to indicate how many elements can be stored in this quad tree node
    const int m_child_type_nw_ = 1;
    const int m_child_type_ne_ = 2;
    const int m_child_type_sw_ = 3;
    const int m_child_type_se_ = 4;

    // Axis-aligned bounding box stored as a center with half-dimensions
    // to represent the boundaries of this quad tree
    Aabb m_boundary_;

    static TreeParam m_param_;  // see strct.h for definition

    // Points in this quad tree node
    std::shared_ptr<Node> m_node_;
    std::shared_ptr<OnGpis> m_gp_;

    bool m_leaf_;
    bool m_max_depth_reached_;
    bool m_root_limit_reached_;

    int32_t m_num_nodes_;

    // Children
    QuadTree *m_north_west_;
    QuadTree *m_north_east_;
    QuadTree *m_south_west_;
    QuadTree *m_south_east_;

    QuadTree *m_tree_;

    explicit QuadTree(Aabb boundary, QuadTree *p = nullptr);

    QuadTree(Aabb boundary, QuadTree *ch, int child_type);

    void
    Subdivide();  // Create four children that fully divide this quad into four quads of equal area

    void
    SubdivideExcept(int child_type);

    void
    DeleteChildren();

    bool
    InsertToParent(std::shared_ptr<Node> n);

    bool
    InsertToParent(std::shared_ptr<Node> n, std::unordered_set<QuadTree *> &quads);

    void
    UpdateCount();

    void
    SetParent(QuadTree *const p) {
        m_tree_ = p;
    }

#if defined(BUILD_TEST)
public:
#else
protected:
#endif
    [[nodiscard]] QuadTree *
    GetParent() const {
        return m_tree_;
    }

    [[nodiscard]] bool
    IsLeaf() const {
        return m_leaf_;
    }  // leaf is true if children are initialized

    [[nodiscard]] bool
    IsEmpty() const {
        return (m_node_ == nullptr);
    }  // empty if the data node is null

    [[nodiscard]] bool
    IsEmptyLeaf() const {
        return (m_leaf_ & (m_node_ == nullptr));  // true if no data node no child
    }

public:
    // Methods
    QuadTree()
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
          m_tree_(nullptr) {}

    explicit QuadTree(Point<double> center);

    ~QuadTree() { DeleteChildren(); }

    [[nodiscard]] bool
    IsRoot() const {
        if (m_tree_) return false;
        else
            return true;
    }

    QuadTree *
    GetRoot();

    // Note: Call this function ONLY BEFORE creating an instance of tree
    static void
    SetTreeParam(TreeParam par) {
        m_param_ = par;
    };

    bool
    Insert(std::shared_ptr<Node> n);

    bool
    Insert(std::shared_ptr<Node> n, std::unordered_set<QuadTree *> &quads);

    bool
    IsNotNew(std::shared_ptr<Node> n);

    bool
    Update(std::shared_ptr<Node> n);

    bool
    Update(std::shared_ptr<Node> n, std::unordered_set<QuadTree *> &quads);

    bool
    Remove(std::shared_ptr<Node> n, std::unordered_set<QuadTree *> &quads);

    void
    Update(std::shared_ptr<OnGpis> gp);

    [[nodiscard]] std::shared_ptr<OnGpis>
    GetGp() const {
        return m_gp_;
    }

    bool
    Remove(std::shared_ptr<Node> n);

    void
    CollectTrees(bool (*qualify)(const QuadTree *tree), std::vector<QuadTree *> &trees);

    void
    QueryRange(Aabb range, std::vector<std::shared_ptr<Node>> &nodes);

    void
    QueryNonEmptyLevelC(Aabb range, std::vector<QuadTree *> &quads);

    void
    QueryNonEmptyLevelC(Aabb range, std::vector<QuadTree *> &quads, std::vector<double> &sqdst);

    [[nodiscard]] int32_t
    GetNodeCount() const {
        return m_num_nodes_;
    }  // Useless

    Point<double>
    GetCenter() {
        return m_boundary_.GetCenter();
    }

    [[nodiscard]] double
    GetHalfLength() const {
        return m_boundary_.GetHalfLength();
    }

    Point<double>
    GetNw() {
        return m_boundary_.GetNw();
    }

    Point<double>
    GetNe() {
        return m_boundary_.GetNe();
    }

    Point<double>
    GetSw() {
        return m_boundary_.GetSw();
    }

    Point<double>
    GetSe() {
        return m_boundary_.GetSe();
    }

    void
    GetAllChildrenNonEmptyNodes(std::vector<std::shared_ptr<Node>> &nodes);

    friend std::ostream &
    operator<<(std::ostream &os, QuadTree *a_tree);
};
