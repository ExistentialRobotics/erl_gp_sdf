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
 *          Huang Zonghao<ac@hzh.io>
 */

#include "strct.h"
#include <memory>
#include <vector>

Node::Node(Point<double> pos, double val, double pose_sig, Point<double> grad, double grad_sig, NodeType n) {
    m_pos_ = pos;
    m_val_ = val;
    m_grad_ = grad;
    m_pose_sig_ = pose_sig;
    m_grad_sig_ = grad_sig;
    m_nt_ = n;
}

Node::Node(Point<double> pos, NodeType nt)
    : m_grad_(Point<double>()),
      m_val_(double(0.0)),
      m_pose_sig_(double(0.0)),
      m_grad_sig_(double(0.0)) {
    m_pos_ = pos;
    m_nt_ = nt;
}

Node::Node()
    : m_val_(double(0.0)),
      m_pose_sig_(double(0.0)),
      m_grad_sig_(double(0.0)) {
    m_nt_ = NodeType::kNone;
}

void
Node::UpdateData(double val, double pose_sig, Point<double> grad, double grad_sig, NodeType n) {
    m_val_ = val;
    m_grad_ = grad;
    m_pose_sig_ = pose_sig;
    m_grad_sig_ = grad_sig;
    m_nt_ = n;
}

void
Node::UpdateNoise(double pose_sig, double grad_sig) {
    m_pose_sig_ = pose_sig;
    m_grad_sig_ = grad_sig;
}

Node3::Node3(Point3<double> pos, double val, double pose_sig, Point3<double> grad, double grad_sig, NodeType n) {
    m_pos_ = pos;
    m_val_ = val;
    m_grad_ = grad;
    m_pose_sig_ = pose_sig;
    m_grad_sig_ = grad_sig;
    m_nt_ = n;
}

Node3::Node3(Point3<double> pos, NodeType nt)
    : m_grad_(Point3<double>()),
      m_val_(double(0.0)),
      m_pose_sig_(double(0.0)),
      m_grad_sig_(double(0.0)) {
    m_pos_ = pos;
    m_nt_ = nt;
}

Node3::Node3()
    : m_val_(double(0.0)),
      m_pose_sig_(double(0.0)),
      m_grad_sig_(double(0.0)) {
    m_nt_ = NodeType::kNone;
}

void
Node3::UpdateData(double val, double pose_sig, Point3<double> grad, double grad_sig, NodeType n) {
    m_val_ = val;
    m_grad_ = grad;
    m_pose_sig_ = pose_sig;
    m_grad_sig_ = grad_sig;
    m_nt_ = n;
}

void
Node3::UpdateNoise(double pose_sig, double grad_sig) {
    m_pose_sig_ = pose_sig;
    m_grad_sig_ = grad_sig;
}
