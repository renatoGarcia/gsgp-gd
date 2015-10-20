/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package sgp.population.selector;

import java.util.ArrayList;
import java.util.Arrays;
import sgp.MersenneTwister;
import sgp.population.Individual;
import sgp.population.Population;

/**
 *
 * @author luiz
 */
public class TournamentSelector implements IndividualSelector{
    private int tournamentSize;

    public TournamentSelector(int tournamentSize) {
        this.tournamentSize = tournamentSize;
    }
    
    @Override
    public Individual selectIndividual(Population population, MersenneTwister rnd){
        int popSize = population.size();
        ArrayList<Integer> indexes = new ArrayList<>();
        for(int i = 0; i < popSize; i++) indexes.add(i);
        Individual[] tournament = new Individual[tournamentSize];
        for(int i = 0; i < tournamentSize; i++){
            tournament[i] = population.getIndividual(indexes.remove(rnd.nextInt(indexes.size())));
        }
        Arrays.sort(tournament);
        return tournament[0];
    }
}
