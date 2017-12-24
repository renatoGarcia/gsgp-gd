/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package edu.gsgp.nodes.functions;

import edu.gsgp.nodes.Node;

/**
 * Protected Logarithm
 */
public class ProtectedLog extends Function{
    public ProtectedLog() { 
        super();
    }

    @Override
    public int getArity() { return 1; }
    
    @Override
    public double eval(double[] inputs) {
        double tmp = Math.abs(arguments[0].eval(inputs));
        if(tmp == 0)
        {
            return 0;
        }
        else
        {
            return Math.log(tmp);
        }
    }

    @Override
    public int getNumNodes() {
        return arguments[0].getNumNodes() + 1;
    }
    
    @Override
    public Function softClone() {
        return new ProtectedLog();
    }
    
    @Override
    public String toString() {
        return "ProtectedLog(" + arguments[0].toString() + ")";
    }
    
    @Override
    public Node clone(Node parent) {
        ProtectedLog newNode = new ProtectedLog();
        for(int i = 0; i < getArity(); i++) newNode.arguments[i] = arguments[i].clone(newNode);
        newNode.parent = parent;
        newNode.parentArgPosition = parentArgPosition;
        return newNode;
    }
}
