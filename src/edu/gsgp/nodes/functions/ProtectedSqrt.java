/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package edu.gsgp.nodes.functions;

import edu.gsgp.nodes.Node;

/**
 * Protected Square Root
 */
public class ProtectedSqrt extends Function{
    public ProtectedSqrt() { 
        super();
    }

    @Override
    public int getArity() { return 1; }
    
    @Override
    public double eval(double[] inputs) {
        double tmp = Math.abs(arguments[0].eval(inputs));
        return Math.sqrt(tmp);
    }

    @Override
    public int getNumNodes() {
        return arguments[0].getNumNodes() + 1;
    }
    
    @Override
    public Function softClone() {
        return new ProtectedSqrt();
    }
    
    @Override
    public String toString() {
        return "ProtectedSqrt(" + arguments[0].toString() + ")";
    }
    
    @Override
    public Node clone(Node parent) {
        ProtectedSqrt newNode = new ProtectedSqrt();
        for(int i = 0; i < getArity(); i++) newNode.arguments[i] = arguments[i].clone(newNode);
        newNode.parent = parent;
        newNode.parentArgPosition = parentArgPosition;
        return newNode;
    }
}
