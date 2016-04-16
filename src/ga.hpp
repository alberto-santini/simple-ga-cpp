#ifndef GA_H
#define GA_H

#include "pcg/pcg_random.hpp"
#include <functional>
#include <assert.h>
#include <utility>
#include <random>
#include <set>

namespace ga {
    namespace helpers {
        template<typename T>
        T get_random_and_remove(std::vector<T>& v, pcg32& rng) {
            assert(v.size() > 0);
            
            std::uniform_int_distribution<> dist(0, v.size() - 1);
            int position = dist(rng);
            
            T t = v[position];
            v.erase(v.begin() + position);
            
            return t;
        }
    }
    
    template<typename Individual>
    class GA {
    private:
        struct IndividualComparator {
            bool operator()(const Individual& lhs, const Individual& rhs) const {
                if(lhs.is_feasible() == rhs.is_feasible()) { return rhs.fitness() < lhs.fitness(); }
                if(lhs.is_feasible()) { return true; }
                return false;
            }
        };
        
    public:
        class Builder;
        class Params;
        class Visitor;
        
        using Population = std::set<Individual, IndividualComparator>;
        using PopulationIterator = typename std::set<Individual>::iterator;
        using CrossoverOrMutation = std::pair<PopulationIterator, PopulationIterator>;
            
        GA() : params(Params::Builder().build()), visitor(Visitor::Builder().build()) {}
        GA(auto p, auto v, auto g, auto l, auto f, auto c, auto m) :
            params(p),
            visitor(v),
            generate_new_individual(g),
            local_search(l),
            fix(f),
            crossover(c),
            mutation(m),
            rng(pcg_extras::seed_seq_from<std::random_device>{}) {}
            
        const Population& pop() const { return population; }
        Population solve() {
            generate_initial_population();
            visitor.after_initial_population_generation(*this);
            
            perform_local_search();
            visitor.after_initial_local_search(*this);
            
            for(auto gen = 0; gen < params.max_generations; ++gen) {
                visitor.at_beginning_iteration(*this, gen);
                
                auto com = get_crossover_or_mutation_sets();
                visitor.after_reproduction_sets_generation(*this, gen, com);
                
                perform_crossover_and_mutation(com);
                visitor.after_crossover_and_mutation(*this, gen);
                
                assert(population.size() >= static_cast<typename Population::size_type>(params.pop_size));
                
                fix_unfeasible_individuals();
                visitor.after_fixing_unfeasible_individuals(*this, gen);
                
                perform_local_search();
                visitor.after_local_search(*this, gen);
                
                remove_bad_individuals();
                visitor.after_removing_bad_individuals(*this, gen);
            }
            
            return population;
        }
        
    private:
        Params params;
        Visitor visitor;
        
        std::function<Individual(void)> generate_new_individual;
        std::function<Individual(Individual)> local_search;
        std::function<Individual(Individual)> fix;
        std::function<std::pair<Individual, Individual>(std::pair<Individual, Individual>)> crossover;
        std::function<Individual(const Individual&)> mutation;
        
        Population population;
        
        mutable pcg32 rng;
        
        void generate_initial_population() {
            auto n = 0;
            while(n < params.pop_size) {
                auto res = population.insert(generate_new_individual());
                if(res.second) { ++n; }
            }
        }
        
        void perform_local_search() {
            Population new_population;
            for(const auto& i : population) {
                new_population.insert(local_search(i));
            }
            std::swap(population, new_population);
        }
        
        void fix_unfeasible_individuals() {
            Population new_population;
            
            for(const auto& i : population) {
                if(i.is_feasible()) {
                    new_population.insert(i);
                } else {
                    new_population.insert(fix(i));
                }
            }
            std::swap(population, new_population);
        }
        
        std::vector<CrossoverOrMutation> get_crossover_or_mutation_sets() const {
            std::vector<CrossoverOrMutation> com;
            
            std::vector<PopulationIterator> iters(population.size());
            std::iota(iters.begin(), iters.end(), population.begin());
            
            assert(std::next(iters.back()) == population.end());
            
            std::uniform_real_distribution<double> dist(0, 1);
            
            for(auto i = 0; i < params.reproduction_sets; ++i) {
                if(dist(rng) < params.prob_mutation) {
                    assert(iters.size() >= 1);
                    
                    auto el = helpers::get_random_and_remove(iters, rng);
                    com.push_back(std::make_pair(el, population.end()));
                } else {
                    assert(iters.size() >= 2);
                    
                    auto el1 = helpers::get_random_and_remove(iters, rng);
                    auto el2 = helpers::get_random_and_remove(iters, rng);
                    com.push_back(std::make_pair(el1, el2));
                }
            }
            
            return com;
        }
        
        void perform_crossover_and_mutation(const std::vector<CrossoverOrMutation>& com) {
            for(const auto& c : com) {
                if(c.second == population.end()) {
                    population.insert(mutation(*c.first));
                } else {        
                    auto new_individuals = crossover(std::make_pair(*c.first, *c.second));
                    population.insert(new_individuals.first);
                    population.insert(new_individuals.second);
                }
            }
        }
        
        void remove_bad_individuals() {
            auto it = population.begin();
            int kept_individuals = 0;
            Population new_population;
            
            while(it != population.end() && kept_individuals < params.pop_size) {
                if((*it).is_feasible()) {
                    new_population.insert(*it);
                    ++it;
                    ++kept_individuals;
                }
            }
            
            std::swap(population, new_population);
        }
    };
    
    template<typename Individual>
    class GA<Individual>::Builder {
        Params params = typename Params::Builder().build();
        Visitor visitor = typename Visitor::Builder().build();
        
        std::function<Individual(void)> generate_new_individual
            = [] () { return Individual(); };
        std::function<Individual(Individual)> local_search
            = [] (auto i) { return i; };
        std::function<Individual(Individual)> fix
            = [] (auto i) { return i; };
        std::function<std::pair<Individual, Individual>(std::pair<Individual, Individual>)> crossover
            = [] (auto p) { return p; };
        std::function<Individual(const Individual&)> mutation
            = [] (const auto& i) { return i; };
        
    public:
        Builder() {}
        Builder& set_params(auto p) { params = p; return *this; }
        Builder& set_visitor(auto v) { visitor = v; return *this; }
        Builder& set_generator(auto g) { generate_new_individual = g; return *this; }
        Builder& set_local_search_operator(auto l) { local_search = l; return *this; }
        Builder& set_fix_operator(auto f) { fix = f; return *this; }
        Builder& set_crossover_operator(auto c) { crossover = c; return *this; }
        Builder& set_mutation_operator(auto m) { mutation = m; return *this; }
        GA<Individual> build() {
            return GA<Individual>(
                params,
                visitor,
                generate_new_individual,
                local_search,
                fix,
                crossover,
                mutation
            );
        }
    };
    
    template<typename Individual>
    class GA<Individual>::Params {
    public:
        class Builder;
        
        int pop_size;
        int max_generations;
        int reproduction_sets;
        double prob_mutation;
        
        Params(int ps, int mi, int rs, double pm) :
            pop_size(ps),
            max_generations(mi),
            reproduction_sets(rs),
            prob_mutation(pm)
        {
            assert(pop_size > 0);
            assert(max_generations > 0);
            assert(reproduction_sets > 0 && reproduction_sets <= pop_size / 2);
            assert(0 <= prob_mutation && prob_mutation <= 1);
        }
    };
    
    template<typename Individual>
    class GA<Individual>::Params::Builder {
        int pop_size = 30;
        int max_generations = 1000;
        int reproduction_sets = 10;
        double prob_mutation = 0.05;
        
    public:
        Builder() {}
        Builder& set_pop_size(auto p) { pop_size = p; return *this; }
        Builder& set_max_generations(auto m) { max_generations = m; return *this; }
        Builder& set_reproduction_sets(auto r) { reproduction_sets = r; return *this; }
        Builder& set_prob_mutation(auto p) { prob_mutation = p; return *this; }
        GA<Individual>::Params build() {
            return GA<Individual>::Params(
                pop_size,
                max_generations,
                reproduction_sets,
                prob_mutation
            );
        }
    };
    
    template<typename Individual>
    struct GA<Individual>::Visitor {
        class Builder;
        
        std::function<void(const GA<Individual>&)> after_initial_population_generation;
        std::function<void(const GA<Individual>&)> after_initial_local_search;
        std::function<void(const GA<Individual>&, int)> at_beginning_iteration;
        std::function<void(const GA<Individual>&, int, const std::vector<CrossoverOrMutation>&)> after_reproduction_sets_generation;
        std::function<void(const GA<Individual>&, int)> after_crossover_and_mutation;
        std::function<void(const GA<Individual>&, int)> after_fixing_unfeasible_individuals;
        std::function<void(const GA<Individual>&, int)> after_local_search;
        std::function<void(const GA<Individual>&, int)> after_removing_bad_individuals;
        
        Visitor(auto aipg, auto ails, auto abi, auto arsg, auto acm, auto afui, auto als, auto arbi) :
            after_initial_population_generation(aipg),
            after_initial_local_search(ails),
            at_beginning_iteration(abi),
            after_reproduction_sets_generation(arsg),
            after_crossover_and_mutation(acm),
            after_fixing_unfeasible_individuals(afui),
            after_local_search(als),
            after_removing_bad_individuals(arbi) {}
    };
    
    template<typename Individual>
    class GA<Individual>::Visitor::Builder {
        std::function<void(const GA<Individual>&)> after_initial_population_generation
            = [] (const auto& _) {};
        std::function<void(const GA<Individual>&)> after_initial_local_search
            = [] (const auto& _) {};
        std::function<void(const GA<Individual>&, int)> at_beginning_iteration
            = [] (const auto& _1, auto _2) {};
        std::function<void(const GA<Individual>&, int, const std::vector<CrossoverOrMutation>&)> after_reproduction_sets_generation
            = [] (const auto& _1, auto _2, const auto& _3) {};
        std::function<void(const GA<Individual>&, int)> after_crossover_and_mutation
            = [] (const auto& _1, auto _2) {};
        std::function<void(const GA<Individual>&, int)> after_fixing_unfeasible_individuals
            = [] (const auto& _1, auto _2) {};
        std::function<void(const GA<Individual>&, int)> after_local_search
            = [] (const auto& _1, auto _2) {};
        std::function<void(const GA<Individual>&, int)> after_removing_bad_individuals
            = [] (const auto& _1, auto _2) {};
        
    public:
        Builder() {}
        Builder& set_after_initial_population_generation(auto a) { after_initial_population_generation = a; return *this; }
        Builder& set_after_initial_local_search(auto a) { after_initial_local_search = a; return *this; }
        Builder& set_at_beginning_iteration(auto a) { at_beginning_iteration = a; return *this; }
        Builder& set_after_reproduction_sets_generation(auto a) { after_reproduction_sets_generation = a; return *this; }
        Builder& set_after_crossover_and_mutation(auto a) { after_crossover_and_mutation = a; return *this; }
        Builder& set_after_fixing_unfeasible_individuals(auto a) { after_fixing_unfeasible_individuals = a; return *this; }
        Builder& set_after_local_search(auto a) { after_local_search = a; return *this; }
        Builder& set_after_removing_bad_individuals(auto a) { after_removing_bad_individuals = a; return *this; }
        GA<Individual>::Visitor build() {
            return GA<Individual>::Visitor(
                after_initial_population_generation,
                after_initial_local_search,
                at_beginning_iteration,
                after_reproduction_sets_generation,
                after_crossover_and_mutation,
                after_fixing_unfeasible_individuals,
                after_local_search,
                after_removing_bad_individuals
            );
        }
    };
}

#endif