### Load libraries ###
using CSV, DataFrames, Dates;
df = DataFrames;
using Colors, ColorBrewer;
using Plots, StatsPlots;
plt = Plots;
using CategoricalArrays, StatsBase, Statistics, Distributions, HypothesisTests, Distances;
using GLM, DecisionTree, LIBSVM, XGBoost, Flux;


### Import/set data ###
## our test is based on successful fish sigh-ups from youtube video "Introduction to Bayesian data analysis - part 1".
## first set our data parameters.
# successful sign-ups
x = 6;

# total in campaign
n = 16;

# set prior parameter values
@time prior = rand(Distributions.Beta(1.0, 1.0), 10^5);

# run simulation of likelihood
@time likelihood = [rand(Distributions.Binomial(n, p)) for p in prior];

# posterior distribution
@time posterior = df.DataFrame(likelihood = likelihood, prior = prior);
@time posterior[!, :posterior_flag] = ifelse.(posterior[:likelihood] .== x, 1, 0);

@time print(df.by(posterior, :posterior_flag, [:prior] =>
    x -> (count=length(x[:prior]), percent=length(x[:prior])/df.size(posterior)[1])));

@time p1 = plt.histogram(prior;
    border="black", color="green", bins=30,
    title="distribution of prior parameter p", xlabel="p", ylabel="frequency",
    legend=false);

@time p2 = @df posterior[posterior[:posterior_flag] .== 1, :] plt.histogram(:prior;
    border="black", color="blue", bins=30,
    title="distribution of posterior parameter p", xlabel="p", ylabel="frequency",
    legend=false);

plt.plot(p1, p2, layout=(2, 1))