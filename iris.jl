### Load libraries ###
using CSV, DataFrames, Dates, RDatasets;
df = DataFrames;
using Colors, ColorBrewer;
using Plots, StatsPlots;
plt = Plots;
using CategoricalArrays, StatsBase, Statistics, Distributions, HypothesisTests;
using GLM, DecisionTree, LIBSVM, XGBoost, Flux;


### Import data ###
iris = RDatasets.dataset("datasets", "iris");
iris = df.DataFrame(iris);
println(iris[1:5, :]);


### Conduct analysis ###
## aggregate.
@time print(df.aggregate(iris, :Species, mean))

## test by function.
@time print(df.by(iris, :Species, [:PetalLength, :PetalWidth, :SepalLength, :SepalWidth] =>
    x -> (count=length(x[:PetalLength]), mean_petal_length=mean(x[:PetalLength]), mean_petal_width=mean(x[:PetalWidth]))))

@time print(df.by(iris, :Species, [:PetalLength, :PetalWidth] =>
    x -> (mean_petal_length=mean(x[:PetalLength]), mean_petal_width=mean(x[:PetalWidth]))))

## test plotting.
@df iris andrewsplot(:Species, cols(1:4),
    line=(fill=[:blue :red :green]), legend=:topleft)

@df iris plt.scatter(:SepalLength, :SepalWidth,
    group=:Species,
    title="sepal length vs sepal width", xlabel="sepal length", ylabel="sepal width",
    legend=:topleft)

@df iris plt.scatter(:SepalLength, :SepalWidth, :PetalLength;
    group=:Species,
    title="3d scatter plot", xlabel="sepal length", ylabel="sepal width", zlabel="petal length",
    legend=:topright)
