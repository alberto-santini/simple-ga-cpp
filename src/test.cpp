#include "ga.hpp"
#include "pcg/pcg_random.hpp"

#include <cmath>
#include <string>
#include <random>
#include <iostream>

static constexpr double EPS = 0.0001;

struct Individual {
    double x, y;

    Individual() {}
    Individual(double x, double y) : x(x), y(y) {}
    bool is_feasible() const { return std::abs(x + y - 1) < EPS; }
    double fitness() const { return - x - std::pow(y, 2) + std::sin(x + y); }
};

std::ostream& operator<<(std::ostream& out, const Individual& i) {
    out << "(" << i.x << "," << i.y << "):" << i.fitness() << (i.is_feasible() ? "" : "*");
    return out;
}

class Generator {
    pcg32& rng;
    
public:
    Generator(pcg32& rng) : rng(rng) {}
    Individual operator()() const {
        std::uniform_real_distribution<double> dist(0, 1);
        auto x = dist(rng);
        auto y = 1 - x;
        
        return Individual(x, y);
    }
};

class LocalSearch {
public:
    Individual operator()(Individual i) const {
        auto i_fit = i.fitness();
        auto best = i;
        for(auto dx = -0.2; dx <= 0.2; dx += 0.05) {
            Individual new_i(i.x + dx, i.y - dx);
            if(new_i.is_feasible() && new_i.fitness() > i_fit) { best = new_i; }
        }
        for(auto dy = -0.2; dy <= 0.2; dy += 0.05) {
            Individual new_i(i.x - dy, i.y + dy);
            if(new_i.is_feasible() && new_i.fitness() > i_fit) { best = new_i; }
        }
        return best;
    }
};

class Fix {
public:
    Individual operator()(Individual i) const {
        if(i.is_feasible()) { return i; }
        
        auto hdist = (i.x + i.y - 1) / 2;
        return Individual(i.x - hdist, i.y - hdist);
    }
};

class Crossover {
public:
    auto operator()(auto p) const {
        return std::make_pair(Individual(p.first.x + 0.1, p.second.y - 0.1), Individual(p.second.x + 0.2, p.first.y - 0.2));
    }
};

class Mutation {
    pcg32& rng;
    
public:
    Mutation(pcg32& rng) : rng(rng) {}
    Individual operator()(Individual i) const {
        std::uniform_real_distribution<double> dist(-0.1, 0.1);
        auto x = i.x + dist(rng);
        auto y = i.y + dist(rng);
        
        return Individual(x, y);
    }
};
int main() {
    using namespace ga;
    
    pcg_extras::seed_seq_from<std::random_device> seed_source;
    pcg32 rng(seed_source);
    
    auto print_pop = [] (const std::string& header, const auto& ga) -> void {
        std::cout << header << std::endl;
        for(const auto& i : ga.pop()) {
            std::cout << i << std::endl;
        }
        std::cout << std::endl;
    };
    
    auto print_rep = [] (const std::string& header, const auto& ga, const auto& com) -> void {
        std::cout << header << std::endl;
        for(const auto& c : com) {
            if(c.second == ga.pop().end()) {
                std::cout << "Mutation of " << *c.first << std::endl;
            } else {
                std::cout << "Crossover between " << *c.first << " and " << *c.second << std::endl;
            }
        }
        std::cout << std::endl;
    };
    
    GA<Individual>::Params params = GA<Individual>::Params::Builder()
        .set_pop_size(15)
        .set_max_generations(100)
        .set_reproduction_sets(2)
        .set_prob_mutation(0.2)
        .build();
    
    GA<Individual>::Visitor visitor = GA<Individual>::Visitor::Builder()
        .set_after_initial_population_generation([&] (const auto& ga) { print_pop("Initial population:", ga); })
        .set_after_initial_local_search([&] (const auto& ga) { print_pop("After local search:", ga); })
        .set_at_beginning_iteration([&] (const auto& ga, int gen) { print_pop("At generation " + std::to_string(gen) + ":", ga); })
        .set_after_reproduction_sets_generation([&] (const auto& ga, int gen, const auto& com) { print_rep("Reproduction sets:", ga, com); })
        .set_after_crossover_and_mutation([&] (const auto& ga, int gen) { print_pop("After crossover and mutation:", ga); })
        .set_after_fixing_unfeasible_individuals([&] (const auto& ga, int gen) { print_pop("After fixing:", ga); })
        .set_after_local_search([&] (const auto& ga, int gen) { print_pop("After local search:", ga); })
        .set_after_removing_bad_individuals([&] (const auto& ga, int gen) { print_pop("After removing bad individuals:", ga); })
        .build();

    GA<Individual> ga = GA<Individual>::Builder()
        .set_params(params)
        .set_visitor(visitor)
        .set_generator(Generator(rng))
        .set_local_search_operator(LocalSearch())
        .set_fix_operator(Fix())
        .set_crossover_operator(Crossover())
        .set_mutation_operator(Mutation(rng))
        .build();

    ga.solve();
    
    return 0;
}