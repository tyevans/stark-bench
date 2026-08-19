"""Wiring: the only layer allowed to know every other one.

A module belongs here when its job is to choose concrete things and hand
them to code that must not choose for itself -- which agent a config name
means, which store a DSN points at. Nothing may import this layer, so
anything placed here is unreachable from the code it wires, which is the
property that makes the wiring safe to change.
"""
